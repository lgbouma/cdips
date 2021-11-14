"""
Given >~10k LCs per galactic field, do the period finding (and detrend, since
the LCs should be variable).

Contents:

do_initial_period_finding: runs period-finding, and sets the detection limits
adapted based on the results.

periodfindingworker: given lcpath, run periodograms and detrend

    run_periodograms_and_detrend: given source_id, time, mag, find planet.

get_tls_sde_versus_period_detection_boundary: given TLS SDE and TLS periods for
"plausible planet detections", define the detection boundary.

run as (from phtess2 usually):
$ python -u do_initial_period_finding.py &> logs/sector6_initial_period_finding.log &
"""

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############

from astropy.timeseries import LombScargle
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
from cdips.testing import check_dependencies
from skim_cream import plot_initial_period_finding_results

from astrobase.lcmath import sigclip_magseries

from wotan import slide_clip

nworkers = mp.cpu_count()

def main():

    check_dependencies()

    do_initial_period_finding(
        sectornum=14, nworkers=nworkers, maxworkertasks=1000,
        outdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding',
        OC_MG_CAT_ver=0.6
    )
    msg = (
        """
        After running, you need to manually tune the SNR distribution for which
        you consider objects, in `do_initial_period_finding`.
        """
    )
    LOGINFO(msg)


def run_periodograms_and_detrend(source_id, time, mag, dtr_dict,
                                 period_min=0.5, period_max=27, orbitgap=1,
                                 expected_norbits=2, orbitpadding=6/(24),
                                 dtr_method='best'):
    """
    Given a source_id, time, and magnitude time-series, this function runs
    clean_rotationsignal_tess_singlesector_light_curve to remove rotation
    signals (via masking orbit edges, sigma slide clip, detrending, and
    re-sigma slide clipping).  "Detrending" here means the "best" method
    currently known, which is the notch + locor combination.
    This was demonstrated through injection-recovery tests
    (/tests/test_injrecov_with_detrending.py)

    kwargs:

        time, mag : time and magnitude vector of light-curve.  PCA is
        preferred, since common instrumental systematics are removed.

        dtr_dict : E.g.,
            {'method':'best', 'break_tolerance':0.5, 'window_length':0.5}

        orbitpadding (float): amount of time to clip near TESS perigee to
        remove ramp signals. 12 data points = 6 hours = 0.25 days (and must
        give in units of days).

        dtr_method (str): any of ['notch', 'locor', 'pspline', 'best'].

    Returns: list of
        r = [source_id, ls_period, ls_fap, ls_amplitude, tls_period, tls_sde,
             tls_t0, tls_depth, tls_duration, tls_distinct_transit_count,
             tls_odd_even, dtr_method]
    """

    lsp_options = {'period_min':0.1, 'period_max':20}

    search_time, search_flux, dtr_stages_dict = (
        dtr.clean_rotationsignal_tess_singlesector_light_curve(
            time, mag, magisflux=False, dtr_dict=dtr_dict,
            lsp_dict=None, maskorbitedge=True, lsp_options=lsp_options,
            verbose=False
        )
    )

    # retrieve LS periodogram information
    ls_period = dtr_stages_dict['lsp_dict']['ls_period']
    ls_amplitude = dtr_stages_dict['lsp_dict']['ls_amplitude']
    ls_fap = dtr_stages_dict['lsp_dict']['ls_fap']

    # run the TLS periodogram
    model = transitleastsquares(search_time, search_flux)
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
    tls_distinct_transit_count = results.distinct_transit_count
    tls_odd_even = results.odd_even_mismatch

    dtr_method = dtr_stages_dict['dtr_method_used']

    r = [source_id, ls_period, ls_fap, ls_amplitude, tls_period, tls_sde,
         tls_t0, tls_depth, tls_duration, tls_distinct_transit_count,
         tls_odd_even, dtr_method]

    return r


def periodfindingworker(lcpath):

    APNAME = 'PCA1'
    source_id, time, mag, xcc, ycc, ra, dec, _, tfa_mag = (
        lcu.get_lc_data(lcpath, mag_aperture=APNAME, tfa_aperture='TFA1')
    )

    dtr_dict = {'method':'best', 'break_tolerance':0.5, 'window_length':0.5}

    if np.all(pd.isnull(mag)):
        r = [source_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan,
             False, xcc, ycc, ra, dec]

    else:
        r = run_periodograms_and_detrend(
            source_id, time, mag, dtr_dict
        )
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
                LOGINFO(f'period finding: {len(results)/N_lcs:.0%} done')
    return log_result


def do_initial_period_finding(
    sectornum=None,
    nworkers=None,
    maxworkertasks=1000,
    outdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding',
    OC_MG_CAT_ver=None
):

    check_dependencies()

    lcdirectory = (
        f'/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{sectornum}/'
    )
    lcglob = 'cam?_ccd?/*_llc.fits'

    np.random.seed(42)
    lcpaths = glob(os.path.join(lcdirectory, lcglob))
    np.random.shuffle(lcpaths)

    if DEBUG:
        # when debugging, period find on fewer LCs
        lcpaths = np.random.choice(lcpaths,size=100)

    outdir = os.path.join(outdir, f'sector-{sectornum}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(outdir, 'initial_period_finding_results.csv')

    if not os.path.exists(outpath):
        tasks = [(x) for x in lcpaths]
        N_lcs = len(lcpaths)

        LOGINFO(f'{len(lcpaths)} files to run initial periodograms on')

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
            columns=['source_id', 'ls_period', 'ls_fap', 'ls_amplitude', 'tls_period',
                     'tls_sde', 'tls_t0', 'tls_depth', 'tls_duration',
                     'tls_distinct_transit_count', 'tls_odd_even',
                     'dtr_method', 'xcc', 'ycc', 'ra', 'dec']
        )
        df = df.sort_values(by='source_id')

        df.to_csv(outpath, index=False)
        LOGINFO(f'made {outpath}')

    else:
        LOGINFO('found periodfinding results, loading them')
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
        LOGINFO('made {}'.format(outpath))
    else:
        LOGINFO('found supplemented periodfinding results, loading them')
        mdf = pd.read_csv(outpath, sep=';')

    if 'reference' in mdf:
        u_ref, u_ref_count = np.unique(mdf['reference'], return_counts=True)
    elif 'reference_id' in mdf:
        u_ref, u_ref_count = np.unique(mdf['reference_id'], return_counts=True)
    else:
        u_ref, u_ref_count = ('N/A', 0)
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
        LOGINFO('made {}'.format(outpath))
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

    limit, abovelimit = get_tls_sde_versus_period_detection_boundary(
        tls_sde, tls_period
    )

    # SET MANUAL SNR LIMITS, based on plot_initial_period_finding_results
    if sectornum == 1:
        df['limit'] = np.ones(len(df))*9
        df['limit'][df['tls_period']<1] = 14
        df['limit'][df['tls_period']>11] = 20
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 2:
        df['limit'] = np.ones(len(df))*8.5
        df['limit'][df['tls_period']<1] = 14
        df['limit'][df['tls_period']>11] = 30
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 3:
        df['limit'] = np.ones(len(df))*8.5
        df['limit'][df['tls_period']<1] = 11
        df['limit'][df['tls_period']>15] = 20
        df['limit'][(df['tls_period']<2.05) & (df['tls_period']>1.95)] = 15
        df['limit'][(df['tls_period']<2.55) & (df['tls_period']>2.45)] = 15
        df['limit'][(df['tls_period']<6.1) & (df['tls_period']>5.75)] = 15
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 4:
        df['limit'] = np.ones(len(df))*10
        df['limit'][df['tls_period']<1] = 13
        df['limit'][(df['tls_period']<9.1) & (df['tls_period']>8.85)] = 22
        df['limit'][(df['tls_period']>21)] = 40
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 5:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 15
        df['limit'][(df['tls_period']<14) & (df['tls_period']>11.5)] = 18
        df['limit'][(df['tls_period']>21)] = 30
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 6:
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

    elif sectornum == 13:
        df['limit'] = np.ones(len(df))*10
        df['limit'][df['tls_period']<1] = 15
        df['limit'][(df['tls_period']<1.5) & (df['tls_period']>1)] = 12
        df['limit'][(df['tls_period']<7.8) & (df['tls_period']>7.2)] = 15
        df['limit'][(df['tls_period']>21)] = 30
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    elif sectornum == 14:
        df['limit'] = np.ones(len(df))*10
        df['limit'][df['tls_period']<1] = 15
        df['limit'][(df['tls_period']<1.5) & (df['tls_period']>1)] = 12
        df['limit'][(df['tls_period']<4.5) & (df['tls_period']>4.44)] = 20
        df['limit'][(df['tls_period']<12.0) & (df['tls_period']>5.0)] = 13
        df['limit'][(df['tls_period']<16.0) & (df['tls_period']>12.0)] = 25
        df['limit'][(df['tls_period']<20.3) & (df['tls_period']>18.0)] = 20
        df['limit'][(df['tls_period']>24)] = 30
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    else:
        df['limit'] = np.ones(len(df))*12
        df['limit'][df['tls_period']<1] = 15
        df['limit'][(df['tls_period']<6.1) & (df['tls_period']>5.75)] = 15
        df['abovelimit'] = np.array(df['tls_sde']>df['limit']).astype(int)

    outpath = os.path.join(
        resultsdir, 'initial_period_finding_results_with_limit.csv'
    )
    df.to_csv(outpath, index=False)
    LOGINFO('made {}'.format(outpath))

    plot_initial_period_finding_results(df, resultsdir)

    if sectornum not in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
        raise NotImplementedError(
            'you need to manually set SNR limits for this sector!'
        )


def get_tls_sde_versus_period_detection_boundary(
    tls_sde, tls_period, make_plots=False
):
    """
    Given np.ndarrays of tls_sde and tls_period, return an array "limit", which
    represents the boundary in SDE versus Period space between "above
    threshold" and "below threshold" planets.  Also, return the array
    "abovelimit", which is just the boolean array `tls_sde > limit`.

    The construction of the boundary is performed by sorting on tls_period
    """

    assert np.all(np.isfinite(tls_period))
    assert np.all(np.isfinite(tls_sde))
    assert len(tls_sde) == len(tls_period)

    N_lcs = len(tls_period)

    WRN_THRESHOLD = 1e4

    if N_lcs < WRN_THRESHOLD:
        msg = (
            f'Only got {N_lcs} light curves for definition of TLS versus '
            'SDE boundary. Will likely often default to baseline window.'
        )
        print(msg)

    # the cleanest window for single-sector TESS data is typically
    # from 2 to 10 days. use that to set the "bare minimum" threshold, as the
    # 95th percentile of the TLS SDE..
    base_window = (tls_period > 2) & (tls_period < 10)
    sde_95 = np.percentile(tls_sde[base_window], 95)

    # ~2e4 to 1e5 LCs total typical per sector.
    N_bins = int(N_lcs / 100)

    hist, bin_edges = np.histogram(np.log10(tls_period), N_bins)

    sde_boundary = []
    midpoints = []
    N_cutoff = 50

    hist_50 = np.percentile(hist, 50)
    hist_75 = np.percentile(hist, 75)
    hist_90 = np.percentile(hist, 90)
    hist_95 = np.percentile(hist, 95)
    hist_98 = np.percentile(hist, 98)

    for lo,hi,count in zip(bin_edges[0:-1], bin_edges[1:], hist):

        midpoint = (lo+hi)/2

        _sel = (
            (np.log10(tls_period) >= lo)
            &
            (np.log10(tls_period) < hi)
        )

        these_sdes = tls_sde[_sel]

        # you want a higher cutoff in regions with more points.
        if count > hist_98:
            this_sde = np.percentile(these_sdes, 99.8)
        elif count > hist_95:
            this_sde = np.percentile(these_sdes, 99.5)
        elif count > hist_90:
            this_sde = np.percentile(these_sdes, 99)
        elif count > hist_75:
            this_sde = np.percentile(these_sdes, 98)
        elif count > hist_50:
            this_sde = np.percentile(these_sdes, 97)
        else:
            this_sde = np.percentile(these_sdes, 80)

        midpoints.append(midpoint)
        if len(these_sdes) > N_cutoff:
            sde_boundary.append(
                np.max([
                    sde_95, this_sde
                ])
            )
        else:
            sde_boundary.append(
                sde_95
            )
    sde_boundary = np.array(sde_boundary)
    period_bin_edges = 10**bin_edges

    fn_log10period_to_limit = interp1d(
        midpoints, sde_boundary, kind='quadratic', bounds_error=False,
        fill_value='extrapolate'
    )

    interp_log10period = np.linspace(
        min(np.log10(tls_period)), max(np.log10(tls_period)), 1000
    )
    interp_limit = fn_log10period_to_limit(interp_log10period)

    limit = fn_log10period_to_limit(np.log10(tls_period))
    abovelimit = np.array(tls_sde > limit).astype(bool)

    if make_plots:
        # assessment plots

        # 1d period histogram
        plt.hist(np.log10(tls_period), bins=N_bins)
        plt.xlabel('log10 tls period [d]')
        plt.ylabel('count')
        plt.savefig('temp_tlsperiod_log10hist.png', dpi=400)
        plt.close('all')

        # 2d period vs SDE histogram
        from fast_histogram import histogram1d, histogram2d
        from astropy.visualization.mpl_normalize import ImageNormalize
        from astropy.visualization import LogStretch
        norm = ImageNormalize(vmin=0.1, vmax=50, stretch=LogStretch())
        fig = plt.figure()
        fig.set_dpi(600)
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        ax.scatter_density(np.log10(tls_period), tls_sde, cmap='Greys', norm=norm)
        plt.xlabel('log10 tls period [d]')
        plt.ylabel('SDE')
        fig.savefig('temp_log10period_log10sde_scatterdensity.png', dpi=600)
        plt.close('all')

        # scatter period vs SDE with boundary
        #plt.plot(bin_edges, sde_boundary, c='orange', lw=0.5, zorder=3)
        title = f'N={len(tls_sde)}. N_above={len(tls_sde[abovelimit])}'
        plt.plot(interp_log10period, interp_limit, c='orange', lw=0.5, zorder=3)
        plt.scatter(np.log10(tls_period), tls_sde,
                    s=0.5, c='k', zorder=2, marker='.', linewidths=0)
        plt.scatter(np.log10(tls_period[abovelimit]), tls_sde[abovelimit],
                    s=4, c='C0', zorder=3, marker='.', linewidths=0)
        plt.xlabel('log10 tls period [d]')
        plt.ylabel('SDE')
        plt.title(title)
        plt.savefig('temp_period_sde_scatter_boundary.png', dpi=400)
        plt.close('all')

        # hist of above limit
        plt.hist(np.log10(tls_period[abovelimit]), bins=50)
        plt.xlabel('log10 tls period [d]')
        plt.ylabel('count')
        plt.savefig('temp_tlsperiod_log10hist_abovelimit.png', dpi=400)
        plt.close('all')

    return limit, abovelimit.astype(int)




if __name__ == "__main__":
    main()
