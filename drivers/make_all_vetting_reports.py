"""
Make multipage PDFs needed to vet CDIPS objects of interest.

$ python -u make_all_vetting_reports.py &> logs/vetting_pdf.log &
"""

from cdips.vetting import (
    initialize_vetting_report_information as ivri,
    make_all_vetting_reports as mavp
)

import os
from glob import glob

def main(sector=None, cdips_cat_vnum=None):
    #
    # enforce that TFA-SR worked through file counts.
    # directory containing both TFA-SR light curves, as well as just ordinary
    # light curves copied over without TFA-SR applied.
    #
    tfasrdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}_TFA_SR'.
        format(sector)
    )
    pfdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding/sector-{}'.
        format(sector)
    )
    with open(os.path.join(pfdir, 'n_above_and_below_limit.txt'), 'r') as f:
        l = f.readlines()
    N_above = int(l[0].split('|')[0].split('=')[1])
    N_below = int(l[0].split('|')[1].split('=')[1])
    N_SR_lightcurves = len(glob(os.path.join(tfasrdir, '*.fits')))
    if N_above != N_SR_lightcurves:
        errmsg = (
            'Expected {} light curves in the TFA_SR directory. Got {}.'.
            format(N_above, N_SR_lightcurves)
        )

    #
    # make the vetting reports
    #
    (tfa_sr_paths, lcbasedir, resultsdir,
    cddf, supplementstatsdf, pfdf, toidf,
    k13_notes_df, sector) = (
            ivri.initialize_vetting_report_information(sector, cdips_cat_vnum)
    )

    mavp.make_all_vetting_reports(
        tfa_sr_paths, lcbasedir, resultsdir, cddf, supplementstatsdf, pfdf,
        toidf, k13_notes_df, sector=sector, show_rvs=True
    )

if __name__ == "__main__":
    main(sector=12, cdips_cat_vnum=0.4)
