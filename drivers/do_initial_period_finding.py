"""
given >~10k LCs per galactic field, do the period finding.

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

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils import lcutils as lcu
from cdips.lcproc import mask_orbit_edges as moe
from skim_cream import plot_initial_period_finding_results

def main():

    do_initial_period_finding(
        sectornum=6, nworkers=52, maxworkertasks=1000,
        outdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding',
        OC_MG_CAT_ver=0.3
    )


def run_periodograms(source_id, tfa_time, tfa_mag, period_min=0.5,
                     period_max=27, orbitgap=1, expected_norbits=2,
                     orbitpadding=6/(24)):
    """
    kwargs:

        orbitpadding (float): amount of time to clip near TESS perigee to
        remove ramp signals. 12 data points = 6 hours = 0.25 days (and must
        give in units of days).
    """

    #
    # Lomb scargle w/ uniformly weighted points.
    #
    ls = LombScargle(tfa_time, tfa_mag, tfa_mag*1e-3)
    freq, power = ls.autopower(minimum_frequency=1/period_max,
                               maximum_frequency=1/period_min)
    ls_fap = ls.false_alarm_probability(power.max())
    ls_period = 1/freq[np.argmax(power)]

    #
    # Transit least squares, ditto.
    #
    f_x0 = 1e4
    m_x0 = 10
    tfa_flux = f_x0 * 10**( -0.4 * (tfa_mag - m_x0) )
    tfa_flux /= np.nanmedian(tfa_flux)

    #
    # ignore the times near the edges of orbits for TLS.
    #
    bls_times, bls_flux = moe.mask_orbit_start_and_end(tfa_time, tfa_flux)

    model = transitleastsquares(bls_times, bls_flux)
    results = model.power(use_threads=1, show_progress_bar=False,
                          R_star_min=0.1, R_star_max=10, M_star_min=0.1,
                          M_star_max=5.0, period_min=period_min,
                          period_max=period_max, n_transits_min=2,
                          transit_template='default', oversampling_factor=4)

    tls_sde = results.SDE
    tls_period = results.period
    tls_t0 = results.T0
    tls_depth = results.depth
    tls_duration = results.duration

    r = [source_id, ls_period, ls_fap, tls_period, tls_sde, tls_t0, tls_depth,
        tls_duration]

    return r


def get_lc_data(lcpath, tfa_aperture='TFA2'):

    return lcu.get_lc_data(lcpath, tfa_aperture='TFA2')


def periodfindingworker(lcpath):

    source_id, tfa_time, tfa_mag, xcc, ycc, ra, dec, _ = get_lc_data(lcpath)

    if np.all(pd.isnull(tfa_mag)):
        r = [source_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
             xcc, ycc, ra, dec]

    else:
        r = run_periodograms(source_id, tfa_time, tfa_mag)
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
    OC_MG_CAT_ver=0.3
):

    lcdirectory = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/'.
        format(sectornum)
    )
    lcglob = 'cam?_ccd?/*_llc.fits'

    np.random.seed(42)
    lcpaths = glob(os.path.join(lcdirectory, lcglob))
    np.random.shuffle(lcpaths)

    #lcpaths = np.random.choice(lcpaths,size=1000) #NOTE for debugging

    outdir = os.path.join(outdir, 'sector-{}'.format(sectornum))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(outdir, 'initial_period_finding_results.csv')

    if not os.path.exists(outpath):
        tasks = [(x) for x in lcpaths]
        N_lcs = len(lcpaths)

        print('%sZ: %s files to run initial periodograms on' %
              (datetime.utcnow().isoformat(), len(lcpaths)))

        pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

        # fire up the pool of workers
        #results = pool.map(periodfindingworker, tasks)
        #results = pool.map_async(periodfindingworker, tasks, make_log_result,)
        results = []
        for task in tasks:
            pool.apply_async(periodfindingworker, args=[task],
                             callback=make_log_result(results, N_lcs))

        # wait for the processes to complete work
        pool.close()
        pool.join()

        df = pd.DataFrame(
            results,
            columns=['source_id', 'ls_period', 'ls_fap', 'tls_period', 'tls_sde',
                     'tls_t0', 'tls_depth', 'tls_duration', 'xcc', 'ycc', 'ra',
                     'dec']
        )
        df = df.sort_values(by='source_id')

        df.to_csv(outpath, index=False)
        print('made {}'.format(outpath))

    else:
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

        mdf.to_csv(outpath, index=False)
        print('made {}'.format(outpath))
    else:
        mdf = pd.read_csv(outpath)

    u_ref, u_ref_count = np.unique(mdf['reference'], return_counts=True)
    u_cluster, u_cluster_count = np.unique(
        np.array(mdf['cluster']).astype(str), return_counts=True)
    outpath = os.path.join(outdir, 'which_references_and_clusters_matter.txt')

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

    # SET SNR LIMITS. at least 12 everywhere. 15 if you're in a ratty region of
    # period space.
    df['limit'] = np.ones(len(df))*12
    df['limit'][df['tls_period']<1] = 15
    df['limit'][(df['tls_period']<6.1) & (df['tls_period']>5.75)] = 15
    df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    outpath = os.path.join(
        resultsdir, 'initial_period_finding_results_with_limit.csv'
    )
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))

    plot_initial_period_finding_results(df, resultsdir)




if __name__ == "__main__":
    main()
