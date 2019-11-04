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
from cdips.vetting import (
    centroid_analysis as cdva,
    make_vetting_multipg_pdf as mvmp,
    initialize_vetting_report_information as ivri
)

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

    (tfa_sr_paths, lcbasedir, resultsdir,
    cddf, supplementstatsdf, pfdf, toidf,
    k13_notes_df, sector) = (
            ivri.initialize_vetting_report_information(sector, cdips_cat_vnum)
    )

    make_all_vetting_reports(tfa_sr_paths, lcbasedir, resultsdir, cddf,
                             supplementstatsdf, pfdf, toidf, k13_notes_df,
                             sector=sector)


if __name__ == "__main__":

    main(sector=9, cdips_cat_vnum=0.4)
