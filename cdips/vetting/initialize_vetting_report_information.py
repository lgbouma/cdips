"""
System-specific path definitions needed to generate vetting reports.
"""
from glob import glob
import os
import numpy as np, pandas as pd

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils.catalogs import get_toi_catalog, get_exofop_toi_catalog

def initialize_vetting_report_information(
    sector, cdips_cat_vnum,
    baseresultsdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/vetting/'
):

    resultsdir = os.path.join(baseresultsdir, f'sector-{sector}')

    dirs = [resultsdir, os.path.join(resultsdir,'pdfs'),
            os.path.join(resultsdir,'pkls'),
            os.path.join(resultsdir,'nottransitpdfs'),
            os.path.join(resultsdir,'nottransitpkls')]
    for _d in dirs:
        if not os.path.exists(_d):
            os.mkdir(_d)

    lcbasedir =  ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                  f'CDIPS_LCS/sector-{sector}/')

    cdips_df = ccl.get_cdips_catalog(ver=cdips_cat_vnum)

    cdips_df = cdips_df.sort_values(by='source_id')

    supppath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
                f'cdips_lc_stats/sector-{sector}/'
                'supplemented_cdips_lc_statistics.txt')
    try:
        supplementstatsdf = pd.read_csv(supppath, sep=';')
    except FileNotFoundError:
        print(f'WRN! Did not find supppath {supppath}')
        supplementstatsdf = None

    pfpath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
              'cdips/results/cdips_lc_periodfinding/'
              f'sector-{sector}/'
              'initial_period_finding_results_with_limit.csv')
    try:
        with open(pfpath) as f:
            first_line = f.readline()
        _temp = first_line.split(';')
        sep = ',' if len(_temp) == 1 else ';'
        pfdf = pd.read_csv(pfpath, sep=sep)
    except FileNotFoundError:
        print(f'WRN! Did not find pfpath {pfpath}')
        pfdf = None

    try:
        toidf = get_toi_catalog()
    except:
        toidf = get_exofop_toi_catalog()

    try:
        lc_paths = pfdf[pfdf.abovelimit == 1].lcpath
    except AttributeError:
        print(f'WRN! Did not get any light curves!')
        lc_paths = None

    return (lc_paths, lcbasedir, resultsdir, cdips_df, supplementstatsdf, pfdf,
            toidf, sector)
