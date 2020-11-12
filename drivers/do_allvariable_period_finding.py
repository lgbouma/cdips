"""
do_allvariable_period_finding.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Given a list of Gaia source_ids, make systematics-corrected multi-sector light
curves, and run periodograms for general variability classification (not just
planet-finding).

Usage:
$ python -u do_allvariable_period_finding.py &> logs/ic2602_allvariable.log &
"""

import numpy as np, pandas as pd

import cdips.utils.lcutils as lcu
import cdips.lcproc.detrend as dtr
import cdips.lcproc.mask_orbit_edges as moe

from wotan.slide_clipper import slide_clip

def main():

    sourcelist_path = (
        '/home/lbouma/proj/cdips/tests/data/test_pca_ic2602_examples.csv'
    )

    df = pd.read_csv(sourcelist_path, comment='#', names=['source_id'])

    for s in list(df.source_id):

        do_allvariable_period_finding(s)


def do_allvariable_period_finding(source_id):

    lcpaths = lcu.find_cdips_lc_paths(source_id)

    #
    # detrend systematics. each light  curve yields tuples of:
    #   primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs
    #
    dtr_infos = []
    for lcpath in lcpaths:
        dtr_info = dtr.detrend_systematics(lcpath)
        dtr_infos.append(dtr_info)

    #
    # stitch all available light curves
    #
    timelist = [d[1]['TMID_BJD'] for d in dtr_infos]
    maglist = [d[1][f'PCA{d[2]}'] for d in dtr_infos]
    magerrlist = [d[1][f'IRE{d[2]}'] for d in dtr_infos]

    time, flux, fluxerr = lcu.stitch_light_curves(timelist, maglist, magerrlist)

    #
    # mask orbit edges
    #
    s_time, s_flux = moe.mask_orbit_start_and_end(
        time, flux, raise_expectation_error=False, orbitgap=0.7
    )

    #
    # remove outliers with windowed stdevn removal
    #
    window_length = 1.5 # days
    s_flux = slide_clip(s_time, s_flux, window_length, low=3, high=3,
                        method='mad', center='median')

    #
    # run the period search
    #
    #TODO

    #
    # save output.
    #

    #
    # make summary plots.
    #


if __name__ == "__main__":
    main()
