"""
System-specific path definitions needed to generate vetting reports.
"""
from glob import glob
import os
import numpy as np, pandas as pd

from cdips.utils import collect_cdips_lightcurves as ccl
from astroquery.vizier import Vizier

def initialize_vetting_report_information(
    sector, cdips_cat_vnum,
    baseresultsdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/vetting/'
):

    resultsdir = os.path.join(baseresultsdir, 'sector-{}'.format(sector))

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
              'cdips/data/csv-file-toi-plus-2019-12-05.csv')
    toidf = pd.read_csv(toipath, comment='#')

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

    return (tfa_sr_paths, lcbasedir, resultsdir, cddf, supplementstatsdf, pfdf,
            toidf, k13_notes_df, sector)
