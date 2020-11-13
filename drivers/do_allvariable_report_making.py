"""
do_allvariable_report_making.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Given a list of Gaia source_ids, make systematics-corrected multi-sector light
curves, run periodograms for general variability classification (not just
planet-finding), and make an associated report.

Usage:
$ python -u do_allvariable_report_making.py &> logs/ic2602_allvariable.log &
"""

import pickle, os
import numpy as np, pandas as pd

import cdips.utils.lcutils as lcu
import cdips.lcproc.detrend as dtr
import cdips.lcproc.mask_orbit_edges as moe
from cdips.plotting.allvar_report import make_allvar_report

from wotan.slide_clipper import slide_clip

def main():

    runid = 'ic2602_examples'
    sourcelist_path = (
        f'/home/lbouma/proj/cdips/tests/data/test_pca_{runid}.csv'
    )

    # the plot and linked pickles go here
    outdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/allvariability_reports'
    outdir = os.path.join(outdir, runid)
    for d in [outdir,
              os.path.join(outdir, 'data'),
              os.path.join(outdir, 'reports')
    ]:
        if not os.path.exists(d):
            os.mkdir(d)

    df = pd.read_csv(sourcelist_path, comment='#', names=['source_id'])

    for s in list(df.source_id):

        try:
            do_allvariable_report_making(s, outdir=outdir)
        except Exception as e:
            print(f'ERROR! {e}')
            pass


def do_allvariable_report_making(source_id, outdir=None):

    picklepath = os.path.join(outdir, 'data', f'{source_id}_allvar.pkl')

    if not os.path.exists(picklepath):

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
        ap = dtr_infos[0][2]
        timelist = [d[1]['TMID_BJD'] for d in dtr_infos]
        maglist = [d[1][f'PCA{ap}'] for d in dtr_infos]
        magerrlist = [d[1][f'IRE{ap}'] for d in dtr_infos]

        extravecdict = {}
        extravecdict[f'IRM{ap}'] = [d[1][f'IRM{ap}'] for d in dtr_infos]
        for i in range(0,7):
            extravecdict[f'CBV{i}'] = [d[3][i, :] for d in dtr_infos]

        time, flux, fluxerr, vec_dict = lcu.stitch_light_curves(
            timelist, maglist, magerrlist, extravecdict
        )

        #
        # mask orbit edges
        #
        s_time, s_flux, inds = moe.mask_orbit_start_and_end(
            time, flux, raise_expectation_error=False, orbitgap=0.7,
            return_inds=True
        )
        s_fluxerr = fluxerr[inds]

        #
        # remove outliers with windowed stdevn removal
        #
        window_length = 1.5 # days
        s_flux = slide_clip(s_time, s_flux, window_length, low=3, high=3,
                            method='mad', center='median')

        ap = dtr_infos[0][2]
        allvardict = {
            'source_id': source_id,
            'ap': ap,
            'TMID_BJD': time,
            f'PCA{ap}': flux,
            f'IRE{ap}': fluxerr,
            'STIME': s_time,
            f'SPCA{ap}': s_flux,
            f'SPCAE{ap}': s_flux,
            'dtr_infos': dtr_infos,
            'vec_dict': vec_dict
        }

        with open(picklepath , 'wb') as f:
            pickle.dump(allvardict, f)

    with open(picklepath, 'rb') as f:
        allvardict = pickle.load(f)

    #
    # make summary plots.
    #
    plotdir = os.path.join(outdir, 'reports')
    make_allvar_report(allvardict, plotdir)


if __name__ == "__main__":
    main()
