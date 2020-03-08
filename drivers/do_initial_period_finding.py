"""
Given >~10k LCs per galactic field, do the period finding (and optionally
detrend if the LCs are variable).

do_initial_period_finding SETS THE SDE LIMITS. currently implemented is at
least 12 everywhere, and 15 if you're in a ratty region of period space.

run as (from phtess2 usually):
$ python -u do_initial_period_finding.py &> logs/sector6_initial_period_finding.log &
"""
from astropy.stats import LombScargle
from transitleastsquares import transitleastsquares

from astrobase import lcmath

import multiprocessing as mp

import numpy as np, pandas as pd
import os, textwrap
from glob import glob
from datetime import datetime
from astropy.io import fits
from copy import deepcopy

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils import lcutils as lcu
from cdips.lcproc import mask_orbit_edges as moe
from cdips.lcproc import detrend as dtr
from skim_cream import plot_initial_period_finding_results

from astrobase.lcmath import sigclip_magseries

# dependencies are hidden; this makes them explicit
import pygam
import wotan

DEBUG = False
nworkers = mp.cpu_count()

def main():

    do_initial_period_finding(
        sectornum=12, nworkers=nworkers, maxworkertasks=1000,
        outdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding',
        OC_MG_CAT_ver=0.4
    )


def run_periodograms_and_detrend(source_id, time, mag, period_min=0.5,
                                 period_max=27, orbitgap=1, expected_norbits=2,
                                 orbitpadding=6/(24), detrend_if_variable=True,
                                 ls_fap_cutoff=1e-5, tfa_time=None,
                                 tfa_mag=None):
    """
    kwargs:

        time, mag : time and magnitude vector of IRM (raw) light-curve.

        tfa_time, tfa_mag: ditto for the TFA light-curve.

        orbitpadding (float): amount of time to clip near TESS perigee to
        remove ramp signals. 12 data points = 6 hours = 0.25 days (and must
        give in units of days).

        detrend_if_variable (bool): if Lomb-Scargle finds a peak with FAP <
        1e-5 in the TFA, the star is "variable". Detrend LC with robust
        penalized B-splines (Eilers & Marx 1996), which are B-splines with knot
        length automatically determined via cross-validation. (Additional knots
        give smaller residuals on the training data, but bigger errors when
        tested on the entire dataset).  I use the Wotan implementation
        (Hippke+2019), which is a wrapper to the pyGAM spline fitter (Serven &
        Brummitt 2018), with $2\sigma$ clipping of outliers from the fit
        residuals at each iteration.  The maximum number of splines by default
        is 50, which for TESS data (total time ~=25 days) is commensurate with
        0.5 day periodic signal.

        In injection-recovery tests (/tests/detrend_checks.py),  the result was
        that the additional trending was only helpful in detecting planets
        whenever substantial non-gaussian variability existed in the TFA light
        curve.
    """
    #
    # Lomb scargle w/ uniformly weighted points.
    #
    ls = LombScargle(tfa_time, tfa_mag, tfa_mag*1e-3)
    freq, power = ls.autopower(minimum_frequency=1/period_max,
                               maximum_frequency=1/period_min)
    ls_fap = ls.false_alarm_probability(power.max())
    ls_period = 1/freq[np.argmax(power)]

    if detrend_if_variable and ls_fap > ls_fap_cutoff:
        # If light-curve was not variable in TFA form (FAP > cutoff), don't run
        # wotan, just search the TFA LC.
        mag = deepcopy(tfa_mag)
        time = tfa_time
    else:
        # If the light-curve is still variable in TFA-form, then default to just
        # running wotan on the raw light-curve.
        pass

    #
    # prepare flux for transit least squares.
    #
    f_x0 = 1e4
    m_x0 = 10
    flux = f_x0 * 10**( -0.4 * (mag - m_x0) )
    flux /= np.nanmedian(flux)

    #
    # ignore the times near the edges of orbits for TLS.
    #
    bls_time, bls_flux = moe.mask_orbit_start_and_end(
        time, flux, raise_expectation_error=False
    )

    #
    # sig clip asymmetric [40,4] (40 is the dip-side).
    #
    bls_time, bls_flux, _ = sigclip_magseries(bls_time, bls_flux,
                                              np.ones_like(bls_flux)*1e-4,
                                              magsarefluxes=True,
                                              sigclip=[40,4], iterative=True)

    #
    # detrend if required.
    #
    detrended = False
    if detrend_if_variable and ls_fap < ls_fap_cutoff:
        if DEBUG:
            print('fap<cutoff for sourceid : {} '.format(source_id))
        bls_flux, _ = dtr.detrend_flux(bls_time, bls_flux)
        detrended = True

    model = transitleastsquares(bls_time, bls_flux)
    results = model.power(use_threads=1, show_progress_bar=False,
                          R_star_min=0.1, R_star_max=10, M_star_min=0.1,
                          M_star_max=5.0, period_min=period_min,
                          period_max=period_max, n_transits_min=1,
                          transit_template='default', oversampling_factor=5)

    tls_sde = results.SDE
    tls_period = results.period
    tls_t0 = results.T0
    tls_depth = results.depth
    tls_duration = results.duration

    r = [source_id, ls_period, ls_fap, tls_period, tls_sde, tls_t0, tls_depth,
        tls_duration, detrended]

    return r


def get_lc_data(lcpath, mag_aperture='IRM2'):

    return lcu.get_lc_data(lcpath, mag_aperture=mag_aperture)


def periodfindingworker(lcpath):

    source_id, time, mag, xcc, ycc, ra, dec, _, tfa_mag = get_lc_data(lcpath)

    if np.all(pd.isnull(mag)):
        r = [source_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
             False, xcc, ycc, ra, dec]

    else:
        r = run_periodograms_and_detrend(source_id, time, mag,
                                         tfa_time=time, tfa_mag=tfa_mag)
        r.append(xcc)
        r.append(ycc)
        r.append(ra)
        r.append(dec)

    return r


def make_log_result(results, N_lcs):
    def log_result(return_value):
        results.append(return_value)
        if N_lcs >= 100:
            if len(results) % (N_lcs//100) == 0:
                print('{}: {:.0%} done'.format(datetime.utcnow().isoformat(),
                                               len(results)/N_lcs))
    return log_result


def do_initial_period_finding(
    sectornum=6,
    nworkers=52,
    maxworkertasks=1000,
    outdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding',
    OC_MG_CAT_ver=None
):

    lcdirectory = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/'.
        format(sectornum)
    )
    lcglob = 'cam?_ccd?/*_llc.fits'

    np.random.seed(42)
    lcpaths = glob(os.path.join(lcdirectory, lcglob))
    np.random.shuffle(lcpaths)

    if DEBUG:
        # when debugging, period find on fewer LCs
        lcpaths = np.random.choice(lcpaths,size=100)

    outdir = os.path.join(outdir, 'sector-{}'.format(sectornum))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(outdir, 'initial_period_finding_results.csv')

    if not os.path.exists(outpath):
        tasks = [(x) for x in lcpaths]
        N_lcs = len(lcpaths)

        print('%sZ: %s files to run initial periodograms on' %
              (datetime.utcnow().isoformat(), len(lcpaths)))

        # pool and run jobs
        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
        results = []
        for task in tasks:
            pool.apply_async(periodfindingworker, args=[task],
                             callback=make_log_result(results, N_lcs))
        pool.close()
        pool.join()

        df = pd.DataFrame(
            results,
            columns=['source_id', 'ls_period', 'ls_fap', 'tls_period',
                     'tls_sde', 'tls_t0', 'tls_depth', 'tls_duration',
                     'pspline_detrended', 'xcc', 'ycc', 'ra', 'dec']
        )
        df = df.sort_values(by='source_id')

        df.to_csv(outpath, index=False)
        print('made {}'.format(outpath))

    else:
        print('found periodfinding results, loading them')
        df = pd.read_csv(outpath)

    # you want some idea of what references, and what clusters are most
    # important.
    outpath = os.path.join(
        outdir, 'initial_period_finding_results_supplemented.csv'
    )
    if not os.path.exists(outpath):
        cd = ccl.get_cdips_catalog(ver=OC_MG_CAT_ver)

        df['source_id'] = df['source_id'].astype(np.int64)
        mdf = df.merge(cd, how='left', on='source_id')

        # ";" sep needed b/c reference is ","-separated
        mdf.to_csv(outpath, index=False, sep=';')
        print('made {}'.format(outpath))
    else:
        print('found supplemented periodfinding results, loading them')
        mdf = pd.read_csv(outpath, sep=';')

    u_ref, u_ref_count = np.unique(mdf['reference'], return_counts=True)
    u_cluster, u_cluster_count = np.unique(
        np.array(mdf['cluster']).astype(str), return_counts=True)
    outpath = os.path.join(outdir, 'which_references_and_clusters_matter.txt')

    if not os.path.exists(outpath):
        with open(outpath, mode='w') as f:

            txt = (
            """
            ==========================================
            of {ntotal} total CDIPS lightcurves made (including nans)

            top 20 referred sources:
            {top20ref}

            top 20 referred sources counts:
            {top20refcounts}

            top 20 referred sources counts fraction:
            {top20refcountsfrac}

            ==========

            top 20 clusters quoted:
            {top20cluster}

            top 20 clusters counts:
            {top20clustercounts}

            top 20 clusters counts fraction:
            {top20clustercountsfrac}
            """
            ).format(
                ntotal=len(mdf),
                top20ref=repr(u_ref[np.argsort(u_ref_count)[::-1]][:20]),
                top20refcounts=repr(u_ref_count[np.argsort(u_ref_count)[::-1]][:20]),
                top20refcountsfrac=repr(u_ref_count[np.argsort(u_ref_count)[::-1]][:20]/len(mdf)),
                top20cluster=repr(u_cluster[np.argsort(u_cluster_count)[::-1]][:20]),
                top20clustercounts=repr(u_cluster_count[np.argsort(u_cluster_count)[::-1]][:20]),
                top20clustercountsfrac=repr(u_cluster_count[np.argsort(u_cluster_count)[::-1]][:20]/len(mdf))
            )
            f.write(textwrap.dedent(txt))
        print('made {}'.format(outpath))
    else:
        pass

    # plot results distribution
    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'cdips_lc_periodfinding/'
        'sector-{}'.format(sectornum)
    )
    initpfresultspath = (
        os.path.join(resultsdir, 'initial_period_finding_results.csv')
    )
    df = pd.read_csv(initpfresultspath)

    # SET MANUAL SNR LIMITS, based on plot_initial_period_finding_results
    if sectornum == 6:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 15
        df['limit'][(df['tls_period']<6.1) & (df['tls_period']>5.925)] = 20
        df['limit'][(df['tls_period']<5.925) & (df['tls_period']>5.75)] = 15
        df['limit'][(df['tls_period']<9) & (df['tls_period']>8.75)] = 15
        df['limit'][(df['tls_period']<12.05) & (df['tls_period']>11.8)] = 15
        df['limit'][(df['tls_period']>20) & (df['tls_period']<25)] = 18
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 7:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 16
        df['limit'][(df['tls_period']<1.95) & (df['tls_period']>1.85)] = 18
        df['limit'][(df['tls_period']>21) & (df['tls_period']<25)] = 18
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 8:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 16
        df['limit'][(df['tls_period']<6.6) & (df['tls_period']>6.3)] = 15
        df['limit'][(df['tls_period']>22)] = 40
        df['limit'][(df['tls_period']<15) & (df['tls_period']>14.5)] = 15
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 9:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 16
        df['limit'][(df['tls_period']>21)] = 18
        df['limit'][(df['tls_period']<13.6) & (df['tls_period']>13.0)] = 25
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 10:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 16
        df['limit'][(df['tls_period']<1.2) & (df['tls_period']>1)] = 14
        df['limit'][(df['tls_period']>21)] = 20
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 11:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 17
        df['limit'][(df['tls_period']<1.2) & (df['tls_period']>1)] = 14.5
        df['limit'][(df['tls_period']<14.9) & (df['tls_period']>13.5)] = 25
        df['limit'][(df['tls_period']<14.4) & (df['tls_period']>14.0)] = 45
        df['limit'][(df['tls_period']<14.9/2) & (df['tls_period']>13.5/2)] = 14
        df['limit'][(df['tls_period']<14.9/3) & (df['tls_period']>13.5/3)] = 14
        df['limit'][(df['tls_period']>24)] = 30
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 12:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 17
        df['limit'][(df['tls_period']<1.2) & (df['tls_period']>1)] = 14.5
        df['limit'][(df['tls_period']<8.8) & (df['tls_period']>8.4)] = 19
        df['limit'][(df['tls_period']<15.5) & (df['tls_period']>13.4)] = 19
        df['limit'][(df['tls_period']<18.9) & (df['tls_period']>18.3)] = 19
        df['limit'][(df['tls_period']>21)] = 30
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    else:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 15
        df['limit'][(df['tls_period']<6.1) & (df['tls_period']>5.75)] = 15
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    df['pspline_detrended'] = df['pspline_detrended'].astype(int)

    outpath = os.path.join(
        resultsdir, 'initial_period_finding_results_with_limit.csv'
    )
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))

    plot_initial_period_finding_results(df, resultsdir)

    if sectornum not in [6,7,8,9,10,11,12]:
        raise NotImplementedError(
            'you need to manually set SNR limits for this sector!'
        )


if __name__ == "__main__":
    main()
