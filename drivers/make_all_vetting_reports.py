"""
Make multipage PDFs needed to vet CDIPS objects of interest.

$ python -u make_all_vetting_reports.py &> logs/vetting_pdf.log &
"""

from glob import glob
import datetime, os, pickle, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cdips.plotting import vetting_pdf as vp
from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.vetting import centroid_analysis as cdva
from cdips.vetting import make_vetting_multipg_pdf as mvmp

from numpy import array as nparr
from astropy.io import fits
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier


def _get_supprow(sourceid, supplementstatsdf):

    mdf = supplementstatsdf.loc[supplementstatsdf['lcobj']==sourceid]

    return mdf


def make_all_vetting_reports(tfa_sr_paths, lcbasedir, resultsdir, cdips_df,
                             supplementstatsdf, pfdf, toidf, k13_notes_df,
                             sector=6, cdipsvnum=1):

    for tfa_sr_path in tfa_sr_paths:

        sourceid = int(tfa_sr_path.split('gaiatwo')[1].split('-')[0].lstrip('0'))
        mdf = cdips_df[cdips_df['source_id']==sourceid]
        if len(mdf) != 1:
            errmsg = 'expected exactly 1 source match in CDIPS cat'
            raise AssertionError(errmsg)

        hdul = fits.open(tfa_sr_path)
        hdr = hdul[0].header
        cam, ccd = hdr['CAMERA'], hdr['CCD']
        hdul.close()

        lcname = (
            'hlsp_cdips_tess_ffi_'
            'gaiatwo{zsourceid}-{zsector}-cam{cam}-ccd{ccd}_'
            'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            cam=cam,
            ccd=ccd,
            zsourceid=str(sourceid).zfill(22),
            zsector=str(sector).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )

        lcpath = os.path.join(
            lcbasedir,
            'cam{}_ccd{}'.format(cam, ccd),
            lcname
        )

        # logic: even if you know it's nottransit, it's a TCE. therefore, the
        # pdf will be made. i just dont want to have to look at it. put it in a
        # separate directory.
        outpath = os.path.join(
            resultsdir,'pdfs',
            'vet_'+os.path.basename(tfa_sr_path).replace('.fits','.pdf')
        )
        nottransitpath = os.path.join(
            resultsdir,'nottransitpdfs',
            'vet_'+os.path.basename(tfa_sr_path).replace('.fits','.pdf')
        )

        supprow = _get_supprow(sourceid, supplementstatsdf)
        suppfulldf = supplementstatsdf

        pfrow = pfdf.loc[pfdf['source_id']==sourceid]
        if len(pfrow) != 1:
            errmsg = 'expected exactly 1 source match in period find df'
            raise AssertionError(errmsg)

        if not os.path.exists(outpath) and not os.path.exists(nottransitpath):
            mvmp.make_vetting_multipg_pdf(tfa_sr_path, lcpath, outpath, mdf,
                                          sourceid, supprow, suppfulldf, pfdf,
                                          pfrow, toidf, sector, k13_notes_df)
        else:
            print('found {}, continue'.format(outpath))


def main(sector=None, cdips_cat_vnum=None):

    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'vetting/'
        'sector-{}'.format(sector)
    )
    dirs = [resultsdir, os.path.join(resultsdir,'pdfs'),
            os.path.join(resultsdir,'pkls'),
            os.path.join(resultsdir,'nottransitpdfs'),
            os.path.join(resultsdir,'nottransitpkls')]
    for _d in dirs:
        if not os.path.exists(_d):
            os.mkdir(_d)

    tfasrdir = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                'CDIPS_LCS/sector-{}_TFA_SR'.format(sector))
    lcbasedir =  ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                  'CDIPS_LCS/sector-{}/'.format(sector))

    cdips_df = ccl.get_cdips_catalog(ver=cdips_cat_vnum)
    cddf = ccl.get_cdips_pub_catalog(ver=cdips_cat_vnum)
    uniqpath = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/OC_MG_FINAL_v{}_uniq.csv'.
        format(cdips_cat_vnum)
    )
    udf = pd.read_csv(uniqpath, sep=';')

    subdf = udf[['unique_cluster_name', 'k13_name_match', 'how_match',
                 'have_name_match', 'have_mwsc_id_match', 'is_known_asterism',
                 'not_in_k13', 'why_not_in_k13', 'cluster']]

    fdf = cdips_df.merge(subdf, on='cluster', how='left')

    cdips_df = cdips_df.sort_values(by='source_id')
    cddf = cddf.sort_values(by='source_id')
    fdf = fdf.sort_values(by='source_id')

    np.testing.assert_array_equal(cddf['source_id'], cdips_df['source_id'])
    np.testing.assert_array_equal(cddf['source_id'], fdf['source_id'])

    cddf['dist'] = cdips_df['dist']
    cddf['is_known_asterism'] = fdf['is_known_asterism']
    cddf['why_not_in_k13'] = fdf['why_not_in_k13']

    del cdips_df, udf

    supppath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
                'cdips_lc_stats/sector-{}/'.format(sector)+
                'supplemented_cdips_lc_statistics.txt')
    supplementstatsdf = pd.read_csv(supppath, sep=';')

    pfpath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
              'cdips/results/cdips_lc_periodfinding/'
              'sector-{}/'.format(sector)+
              'initial_period_finding_results_supplemented.csv')
    pfdf = pd.read_csv(pfpath, sep=';')

    toipath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
              'cdips/data/toi-plus-2019-10-19.csv')
    toidf = pd.read_csv(toipath)

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/A+A/558/A53')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    k13_notes_df = catalogs[2].to_pandas()
    for c in k13_notes_df.columns:
        k13_notes_df[c] = k13_notes_df[c].str.decode('utf-8')

    # reconstructive_tfa/RunTFASR.sh applied the threshold cutoff on TFA_SR
    # lightcurves. use whatever is in `tfasrdir` to determine which sources to
    # make pdfs for.
    tfa_sr_paths = glob(os.path.join(tfasrdir, '*_llc.fits'))

    make_all_vetting_reports(tfa_sr_paths, lcbasedir, resultsdir, cddf,
                             supplementstatsdf, pfdf, toidf, k13_notes_df,
                             sector=sector)


if __name__ == "__main__":

    main(sector=9, cdips_cat_vnum=0.4)
