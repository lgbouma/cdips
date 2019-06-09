"""
[environment: phtess2]

Randomly select N=100 (or 1000) TFA CDIPS light curves.
(Optionally select them to be only the LCs with strong LS FAPs already?)

Then inject a 0.25% = 2.5 mmag central-transit planet with periods in the range
1–12 d.  Try recovering via TLS (a) without detrending, (b) with detrending.
(For each light curve, do say 10 experiments).

For case (b), first-pass try using the robust Huber spline (with w = 0.3 d) and
also the sliding biweight (w = 0.25 d). For a sliding biweight, if w/T14 > 2.2,
most (> 98%) of the flux integral is preserved. So anything between w=0.25 days
to w=0.5 days should be good...
"""

import numpy as np, pandas as pd
import os, textwrap
from glob import glob
from datetime import datetime
from astropy.io import fits

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.lcproc import mask_orbit_edges as moe
from skim_cream import plot_initial_period_finding_results

lcdirectory = (
    '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/'.
    format(sectornum)
)
resultsdirectory = (
    '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/detrend_checks'
)

def main():

    ##########################################
    np.random.seed(42)

    N_periods_per_lc = 10
    N_lcs = 100

    P_lower = 1        # 1 day
    P_upper = 10       # 10 days
    inj_depth = 2.5e-3 # 2.5 mmag central transit

    dtr_dict = {
        'type':'none', # none, hspline, or biweight
        'window':0
    }
    ##########################################

    _lcpaths = get_lc_paths(N_lcs=N_lcs)
    lcpaths = np.tile(lcpaths, N_periods_per_lc)

    inj_periods = np.random.uniform(P_lower, P_upper, N_lcs*N_periods_per_lc)
    inj_epochs = inj_periods * np.random.uniform(0, 1, N_lcs*N_periods_per_lc)
    inj_depths = np.ones_like(inj_periods)*inj_depth

    # each job is one injection / recovery experiment, and associated result.
    for lcpath, inj_period, inj_epoch, inj_depth in zip(
        lcpaths, inj_periods, inj_epochs, inj_depths
    ):
        #FIXME: you will need full batman param set
        inj_dict = {
            'period':inj_period,
            'epoch':inj_epoch,
            'depth':inj_depth
        }

        inj_recov_worker(lcpath, inj_dict, dtr_dict)

    # merge and save results
    csvpaths = glob(os.path.join(resultsdirectory,'worker_output','*.csv'))
    df = pd.concat((pd.read_csv(f) for f in csvpaths))
    outpath = os.path.join(resultsdirectory,
                           'detrend_check_type-{}_window-{}.csv'.
                           format(dtr_dict['type'],dtr_dict['window']))
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))



def inj_recov_worker(lcpath, inj_dict, dtr_dict):
    # output is a csv file with (a) injected params, (b) whether injected
    # signals was recovered in first peak, (c) whether it was in first three
    # peaks.

    source_id, tfa_time, tfa_mag, xcc, ycc, ra, dec, tmag = (
        get_lc_data(lcpath, tfa_aperture='TFA2')
    )

    if np.all(pd.isnull(tfa_mag)):

        outpath = os.path.join(
            resultsdirectory,'worker_output',
            str(source_id)+"_"+inj_dict['period']+".csv"
        )
        if os.path.exists(outpath):
            print('found {}'.format(outpath))
            return

        t = {
            'source_id':source_id,
            'tmag':tmag,
            'allnan':True,
            'recovered_as_best_peak':False,
            'recovered_in_topthree_peaks':False
        }
        outd = {**t, **inj_dict, **dtr_dict}
        outdf = pd.DataFrame(outd)

        outdf.to_csv(outpath, index=False)
        print('made {}'.format(outpath))

        return

    #
    # otherwise, begin the injection recovery + detrending experiment.
    #

    #FIXME FIXME FIXME TODO TODO TODO -- mybe use the old cbp injection code
    # (?), else just rewrite -- not too hard lol!
    # inject
    inject_signal()

    # detrend
    detrend_lightcurve()

    # period find
    find_dips()
    r = run_periodograms(source_id, tfa_time, tfa_mag)

    # check if u got dips, save output
    t = {
        'source_id':source_id,
        'tmag':tmag,
        'allnan':False,
        'recovered_as_best_peak':recovered_as_best_peak,
        'recovered_in_topthree_peaks':recovered_in_topthree_peaks
    }
    outd = {**t, **inj_dict, **dtr_dict}
    outdf = pd.DataFrame(outd)

    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


    return


def get_lc_paths(N_lcs=100):

    lcglob = 'cam?_ccd?/*_llc.fits'

    lcpaths = glob(os.path.join(lcdirectory, lcglob))
    np.random.shuffle(lcpaths)

    lcpaths = np.random.choice(lcpaths,size=N_lcs)

    return lcpaths

def inject_signal():
    pass

def detrend_lightcurve():
    pass

def find_dips():
    pass

def assess_statistics():
    pass
