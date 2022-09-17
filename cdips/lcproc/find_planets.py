"""
Contents:
    run_periodograms_and_detrend:
        given star_id, time, mag, find a single planet.
    detrend_and_iterative_tls:
        given star_id, time, mag, find many planets.
    plot_detrend_check:
        visualize detrending from run_periodograms_and_detrend.
    plot_planet_finding_results:
        viz BLS or TLS results from run_periodograms_and_detrend.
    plot_iterative_planet_finding_results:
        viz TLS results, from detrend_and_iterative_tls
Helpers:
    _p2p_rms
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

from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask, cleaned_array
from astrobase.periodbase import kbls

import os, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from cdips.lcproc import detrend as dtr

import multiprocessing as mp
nworkers = mp.cpu_count()

def detrend_and_iterative_tls(
    star_id, time, mag, dtr_dict,
    search_method='tls',
    max_tls_iterations=3,
    period_min=0.5, period_max=27,
    R_star=1.0, M_star=1.0,
    R_star_min=0.1, R_star_max=5,
    M_star_min=0.1, M_star_max=3.0,
    n_transits_min=3, oversampling_factor=5,
    transit_depth_min=10e-6, # 10ppm
    orbitgap=1,
    expected_norbits=2, orbitpadding=6/(24),
    dtr_method='best', n_threads=1,
    magisflux=False,
    cachepath=None, verbose=False,
    slide_clip_lo=20, slide_clip_hi=3
    ):
    """
    This function is a clone of run_periodograms_and_detrend, but TLS is
    performed iteratively up to a maximum number of attempts, or a S/N cutoff.

    kwargs:

        as in run_periodograms_and_detrend, except for `max_tls_iterations`,
        which is self-explanatory.

    Returns:
        "outdicts": keys are TLS iterations (plus dtr_stages_dict).  Each TLS
        iteration value contains
            (r, tls_results, search_time, search_flux, dy), for
            r = [star_id, ls_period, ls_fap, ls_amplitude, tls_period, tls_sde,
                 tls_t0, tls_depth, tls_duration, tls_distinct_transit_count,
                 tls_odd_even, dtr_method]
    """

    assert search_method == 'tls'

    # if this has been run before, load from a cache.  notch+TLS are slow.
    if isinstance(cachepath, str):
        assert cachepath.endswith(".pkl")
        if os.path.exists(cachepath):
            LOGINFO(f"Found {cachepath}, loading results.")
            with open(cachepath, 'rb') as f:
                d = pickle.load(f)
            return d

    dtrcachepath = cachepath.replace(".pkl", "_dtrcache.pkl")
    lsp_options = {'period_min':0.1, 'period_max':20}
    if os.path.exists(dtrcachepath):
        LOGINFO(f"Found {dtrcachepath}, loading results.")
        with open(dtrcachepath, 'rb') as f:
            d = pickle.load(f)
        search_time, search_flux, dtr_stages_dict = (
            d['search_time'], d['search_flux'], d['dtr_stages_dict']
        )
    else:
        # otherwise, run the detrending; cache results
        search_time, search_flux, dtr_stages_dict = (
            dtr.clean_rotationsignal_tess_singlesector_light_curve(
                time, mag, magisflux=magisflux, dtr_dict=dtr_dict,
                lsp_dict=None, maskorbitedge=True, lsp_options=lsp_options,
                verbose=verbose, slide_clip_lo=slide_clip_lo,
                slide_clip_hi=slide_clip_hi
            )
        )
        outdict = {
            'search_time':search_time,
            'search_flux':search_flux,
            'dtr_stages_dict':dtr_stages_dict
        }
        with open(dtrcachepath, 'wb') as f:
            pickle.dump(outdict, f)
            LOGINFO(f"Wrote {dtrcachepath}")

    # retrieve LS periodogram information
    ls_period = dtr_stages_dict['lsp_dict']['ls_period']
    ls_amplitude = np.abs(dtr_stages_dict['lsp_dict']['ls_amplitude'])
    ls_fap = dtr_stages_dict['lsp_dict']['ls_fap']

    results = {}

    dtr_method = dtr_stages_dict['dtr_method_used']
    count = 0
    outdicts = {}
    outdicts['dtr_stages_dict'] = dtr_stages_dict

    # run the TLS periodograms
    while count < max_tls_iterations:

        dy = np.ones_like(search_time)*_p2p_rms(search_flux)

        if verbose:
            LOGINFO(f'Running TLS iteration {count} '
                    f'w/ {len(search_time)} points')

        model = transitleastsquares(
            search_time, search_flux, dy=dy, verbose=verbose
        )
        results = model.power(
            use_threads=n_threads, show_progress_bar=verbose,
            R_star=R_star, R_star_min=R_star_min, R_star_max=R_star_max,
            M_star=M_star, M_star_min=M_star_min, M_star_max=M_star_max,
            period_min=period_min, period_max=period_max,
            n_transits_min=n_transits_min, transit_template='default',
            transit_depth_min=transit_depth_min,
            oversampling_factor=oversampling_factor
        )

        r = {
            'star_id': star_id,
            'dtr_method': dtr_method,
            'search_method': search_method,
            #
            'ls_period': ls_period,
            'ls_fap': ls_fap,
            'ls_amplitude': ls_amplitude,
            #
            'tls_period': results.get('period', None),
            'tls_sde': results.get('SDE', None),
            'tls_snr': results.get('snr', None),
            'tls_t0': results.get('T0', None),
            'tls_depth': results.get('depth', None),
            'tls_duration': results.get('duration', None),
            'tls_distinct_transit_count': results.get('distinct_transit_count', None),
            'tls_odd_even': results.get('odd_even_mismatch', None),
        }

        outdict = {
            'r':r,
            'tls_results': results,
            'search_time': search_time,
            'search_flux': search_flux,
            'dy':dy,
        }

        outdicts[count] = outdict

        if r['tls_sde'] < 10:
            LOGINFO('Found TLS_SDE < 10. Halting periodogram search.')
            break
        else:
            LOGINFO(f'Found TLS_SDE = {r["tls_sde"]:.1f}. Continuing search.')

        # iteratively mask out transits.
        intransit = transit_mask(
            search_time, results.period, 2*results.duration, results.T0
        )
        search_flux = search_flux[~intransit]
        search_time = search_time[~intransit]
        search_time, search_flux = cleaned_array(search_time, search_flux)

        count += 1

    if isinstance(cachepath, str):
        with open(cachepath, 'wb') as f:
            pickle.dump(outdicts, f)
            LOGINFO(f"Wrote {cachepath}")

    return outdicts



def run_periodograms_and_detrend(
    star_id, time, mag, dtr_dict,
    search_method='tls',
    period_min=0.5, period_max=27,
    R_star=1.0, M_star=1.0,
    R_star_min=0.1, R_star_max=5,
    M_star_min=0.1, M_star_max=3.0,
    n_transits_min=1, oversampling_factor=5,
    transit_depth_min=10e-6, # 10ppm
    orbitgap=1,
    expected_norbits=2, orbitpadding=6/(24),
    dtr_method='best', n_threads=1,
    return_extras=False, magisflux=False,
    cachepath=None, verbose=False,
    slide_clip_lo=20, slide_clip_hi=3
    ):
    """
    Given a star_id, time, and magnitude time-series, this function runs
    clean_rotationsignal_tess_singlesector_light_curve to remove rotation
    signals (via masking orbit edges, sigma slide clip, detrending, and
    re-sigma slide clipping).  "Detrending" here means the "best" method
    currently known, which is the notch + locor combination (which was
    demonstrated via /tests/test_injrecov_with_detrending.py). TLS or BLS is
    then run on the flux residuals from the detrending (not e.g., the BIC
    time-series), depending on the `search_method`.

    Note: TLS runtime is ~1 minute for single-sector TESS volumes over
    plausible grids, and ~100 minutes for all-quarter Kepler volumes (cf the
    Hippke paper).  The speedup from multithreading is sub-linear (factor of
    5.5x going from 1->16 cores, in my testing).

    Note: The uncertainties are evaluated using the p2p_rms, per the discussion
    at https://github.com/hippke/tls/issues/102.

    kwargs:

        star_id: used for bookkeeping, can be any int/str.

        time, mag : time and magnitude vector of light-curve.  PCA is
        preferred, since common instrumental systematics are removed.

        dtr_dict : E.g.,
            {'method':'best', 'break_tolerance':0.5, 'window_length':0.5}
            {'method':'notch', 'break_tolerance':0.5, 'window_length':0.3}

        search_method : 'tls' or 'bls', for transitleastsqaures or the
        astrobase Kovacs+02 multithreaded BLS implementation.

        orbitpadding (float): amount of time to clip near TESS perigee to
        remove ramp signals. 12 data points = 6 hours = 0.25 days (and must
        give in units of days).

        dtr_method (str): any of ['notch', 'locor', 'pspline', 'best'].

        return_extras (bool): default False.  If True, returns the search_time,
        search_flux, and dtr_stages_dict along with the summary row.

        magisflux (bool): default False

        cachepath (str): if a pickle file path is passed, detrending and TLS
        results will be cached to this pickle file.  The detrending results
        will also be cached to a separate (internal) pickle file for speed-up.

        (period_min, period_max, R_star_min,
        R_star_max, M_star_min, M_star_max, n_transits_min,
        oversampling_factor, transit_depth_min): ...
            These are all TLS kwargs.

    Returns:
        If return_extras is False:
            r = [star_id, ls_period, ls_fap, ls_amplitude, tls_period, tls_sde,
                 tls_t0, tls_depth, tls_duration, tls_distinct_transit_count,
                 tls_odd_even, dtr_method]
        else:
            tuple of: r, search_time, search_flux, dtr_stages_dict.
            `dtr_stages_dict` contains np.ndarrays from each step of the
            detrending process, and Lomb-Scargle information from the
            pre-requisite rotation period check.
    """

    assert search_method in ['bls', 'tls']

    # if this has been run before, load from a cache.  notch+TLS are slow.
    if isinstance(cachepath, str):
        assert cachepath.endswith(".pkl")
        if os.path.exists(cachepath):
            LOGINFO(f"Found {cachepath}, loading results.")
            with open(cachepath, 'rb') as f:
                d = pickle.load(f)
            if not return_extras:
                return d['r']
            return (d['r'], d['search_time'], d['search_flux'],
                    d['dtr_stages_dict'])
    else:
        cachedir = os.path.join(os.path.expanduser("~"), ".cdips")
        if not os.path.exists(cachedir): os.mkdir(cachedir)
        cachepath = os.path.join(cachedir, f"{star_id}.pkl")

    dtrcachepath = cachepath.replace(".pkl", "_dtrcache.pkl")
    lsp_options = {'period_min':0.1, 'period_max':20}
    if os.path.exists(dtrcachepath):
        LOGINFO(f"Found {dtrcachepath}, loading results.")
        with open(dtrcachepath, 'rb') as f:
            d = pickle.load(f)
        search_time, search_flux, dtr_stages_dict = (
            d['search_time'], d['search_flux'], d['dtr_stages_dict']
        )
    else:
        # otherwise, run the detrending; cache results
        search_time, search_flux, dtr_stages_dict = (
            dtr.clean_rotationsignal_tess_singlesector_light_curve(
                time, mag, magisflux=magisflux, dtr_dict=dtr_dict,
                lsp_dict=None, maskorbitedge=True, lsp_options=lsp_options,
                verbose=verbose, slide_clip_lo=slide_clip_lo,
                slide_clip_hi=slide_clip_hi
            )
        )
        outdict = {
            'search_time':search_time,
            'search_flux':search_flux,
            'dtr_stages_dict':dtr_stages_dict
        }
        with open(dtrcachepath, 'wb') as f:
            pickle.dump(outdict, f)
            LOGINFO(f"Wrote {dtrcachepath}")

    # retrieve LS periodogram information
    ls_period = dtr_stages_dict['lsp_dict']['ls_period']
    ls_amplitude = np.abs(dtr_stages_dict['lsp_dict']['ls_amplitude'])
    ls_fap = dtr_stages_dict['lsp_dict']['ls_fap']

    results = {}
    blsdict = {}
    if search_method == 'tls':
        # run the TLS periodogram
        dy = np.ones_like(search_time)*_p2p_rms(search_flux)
        model = transitleastsquares(
            search_time, search_flux, dy=dy, verbose=verbose
        )
        results = model.power(
            use_threads=n_threads, show_progress_bar=verbose,
            R_star=R_star, M_star=M_star,
            R_star_min=R_star_min, R_star_max=R_star_max,
            M_star_min=M_star_min, M_star_max=M_star_max,
            period_min=period_min, period_max=period_max,
            n_transits_min=n_transits_min, transit_template='default',
            transit_depth_min=transit_depth_min,
            oversampling_factor=oversampling_factor
        )

    elif search_method == 'bls':
        dy = np.ones_like(search_time)*_p2p_rms(search_flux)
        blsdict = kbls.bls_parallel_pfind(
            search_time, search_flux, dy,
            magsarefluxes=True, startp=period_min, endp=period_max,
            mintransitduration=0.01, maxtransitduration=0.12, nphasebins=200,
            autofreq=True, nbestpeaks=5, nworkers=n_threads, get_stats=True
        )

        # the stats functions in bls_parallel_pfind have issues, especially
        # that the t0 fit is usally wrong.  a possible TODO here is to use the
        # TLS stats functions to fix this.  [NOTE patching astrobase would be
        # the best long-term solution]
        from transitleastsquares.stats import (
            final_T0_fit, intransit_stats
        )

        powers = [p['bestpower'] for p in blsdict['blsresult']]
        assert (
            blsdict['bestperiod']
            ==
            blsdict['blsresult'][np.argmax(powers)]['bestperiod']
        )
        init_tdur = blsdict['blsresult'][np.argmax(powers)]['transduration']*24
        init_depth = 1-blsdict['blsresult'][np.argmax(powers)]['transdepth']
        init_period = blsdict['bestperiod']
        raise NotImplementedError(
            "astrobase BLS t0-finder still needs patching."
        )

        # TODO:
        # follow the logic of "final_T0_fit" in transitleastsquares, and
        # implement your own version of this to get the best-fit T0 over a grid
        # from [t_min, t_min+period].
        # NOTE: after, could reimplement transitleastsquares.stats functions
        # too... to get the odd-even stats!

    dtr_method = dtr_stages_dict['dtr_method_used']

    r = {
        'star_id': star_id,
        'dtr_method': dtr_method,
        'search_method': search_method,
        #
        'ls_period': ls_period,
        'ls_fap': ls_fap,
        'ls_amplitude': ls_amplitude,
        #
        'tls_period': results.get('period', None),
        'tls_sde': results.get('SDE', None),
        'tls_snr': results.get('snr', None),
        'tls_t0': results.get('T0', None),
        'tls_depth': results.get('depth', None),
        'tls_duration': results.get('duration', None),
        'tls_distinct_transit_count': results.get('distinct_transit_count', None),
        'tls_odd_even': results.get('odd_even_mismatch', None),
    }

    if 'stats' in blsdict:
        r['bls_period'] = blsdict.get('bestperiod', None)
        r['bls_t0'] = blsdict['stats'][0].get('epoch', None)
        r['bls_duration'] = blsdict['stats'][0].get('transitduration', None)
        r['bls_snr'] = blsdict['stats'][0].get('snr', None)
        r['bls_depth'] = blsdict['stats'][0].get('transitdepth', None)
        r['bls_npoints_in_transit'] = blsdict['stats'][0].get('npoints_in_transit', None)
    else:
        r['bls_period'] = None
        r['bls_t0'] = None
        r['bls_duration'] = None
        r['bls_snr'] = None
        r['bls_depth'] = None
        r['bls_npoints_in_transit'] = None

    outdict = {
        'r':r,
        'tls_results': results,
        'bls_results': blsdict,
        'search_time':search_time,
        'search_flux':search_flux,
        'dy':dy,
        'dtr_stages_dict':dtr_stages_dict,
    }

    if isinstance(cachepath, str):
        with open(cachepath, 'wb') as f:
            pickle.dump(outdict, f)
            LOGINFO(f"Wrote {cachepath}")

    if not return_extras:
        return r
    else:
        return r, search_time, search_flux, dtr_stages_dict


def plot_detrend_check(star_id, outdir, dtr_dict, dtr_stages_dict,
                       r=None, instrument='tess', cdipslcpath=None,
                       overwrite=False):
    """
    Plots of raw and detrended flux from run_periodograms_and_detrend, with
    vertical lines showing the preferred TLS/BLS period.
    """

    assert instrument in ['tess', 'kepler']

    if 'lsp_dict' in dtr_stages_dict:
        dtr_dict['ls_period'] = dtr_stages_dict['lsp_dict']['ls_period']
        dtr_dict['ls_amplitude'] = dtr_stages_dict['lsp_dict']['ls_amplitude']
        dtr_dict['ls_fap'] = dtr_stages_dict['lsp_dict']['ls_fap']

    time, flux = dtr_stages_dict['time'], dtr_stages_dict['flux']
    sel0 = dtr_stages_dict['sel0']
    clipped_flux = dtr_stages_dict['clipped_flux']
    trend_flux = dtr_stages_dict['trend_flux']
    search_time = dtr_stages_dict['search_time'],
    search_flux = dtr_stages_dict['search_flux']

    if cdipslcpath is not None:
        hdrlist =  (
            'CDCLSTER,CDIPSAGE,TESSMAG,phot_bp_mean_mag,phot_rp_mean_mag'.
            split(',')
        )
        infodict = iu.get_header_keyword_list(lcpath, hdrlist)
        cluster = infodict['CDCLSTER']
        age = float(infodict['CDIPSAGE'])
        tmag = float(infodict['TESSMAG'])
        bpmrp = (
            float(infodict['phot_bp_mean_mag']) -
            float(infodict['phot_rp_mean_mag'])
        )
        titlestr = (
            f"{star_id}: {cluster[:16]}, logt={age:.2f}, "
            f"T={tmag:.2f}, Bp-Rp={bpmrp:.2f}"
        )
        outname = (
            f'tmag{tmag:.2f}_'+
            str(star_id)+
            "_method-{}".format(dtr_dict['method'])+
            '_windowlength-{}'.format(dtr_dict['window_length'])+
            ".png"
        )

    else:
        outname =  (
            f"{star_id}_detrending"
            "_method-{}".format(dtr_dict['method'])+
            '_windowlength-{}'.format(dtr_dict['window_length'])+
            ".png"
        )

    # for kepler, visualize quarter-by-quarter.
    # for tess, just do everything at once.
    if instrument == 'kepler':
        from cdips.paths import DATADIR
        csvpath = os.path.join(
            DATADIR, 'spacecraft', 'time_to_kepler_quarter.csv'
        )
        time_df = pd.read_csv(csvpath)
        segment_id = np.array(time_df.quarter.apply(lambda x: str(x).zfill(2)))
        segment_start = np.array(time_df.tstart)
        segment_stop = np.array(time_df.tstop)

    elif instrument == 'tess':
        segment_id = ['tess']
        segment_start = [np.nanmin(time)]
        segment_stop = [np.nanmax(time)]

    for _segid, _start, _stop in zip(segment_id, segment_start, segment_stop):

        if instrument == 'kepler': _segid = f"Q{_segid}"
        outname =  (
            f"{star_id}"
            f"_{_segid}_detrending"
            f"_method-{dtr_dict['method']}"
            f"_windowlength-{dtr_dict['window_length']}"
            ".png"
        )

        outpng = os.path.join(outdir, outname)

        if os.path.exists(outpng) and not overwrite:
            LOGINFO(f"Found {outpng}, continue.")
            continue

        f,axs = plt.subplots(nrows=2, sharex=True, figsize=(14,7))
        # lower: "raw" data; upper: sigma-clipped
        norm_y = lambda y: 1e3*(y-1)

        axs[0].scatter(
            time, norm_y(clipped_flux), c='black', s=1, zorder=2,
            rasterized=True
        )
        axs[0].scatter(
            time, norm_y(flux), c='red', s=1, zorder=1, rasterized=True
        )
        axs[0].plot(
            time[sel0], norm_y(trend_flux), c='C0'
        )

        axs[1].scatter(
            search_time, norm_y(search_flux), c='black', s=1,
            zorder=2, rasterized=True
        )

        titlestr = (
            f"{star_id} ({_segid})"
        )

        if r is not None:
            search_method = r['search_method']
            period_val = r[f'{search_method}_period']
            t0_val = r[f'{search_method}_t0']
            tdur_val = r[f'{search_method}_duration']

            midtimes = t0_val + np.arange(-2000,2000,1)*period_val

            for _f, ax in zip([norm_y(flux), norm_y(search_flux)], axs):

                med = np.nanmedian(_f)
                std = np.nanstd(_f)
                y_diff = np.nanpercentile(_f,90) - np.nanpercentile(_f,10)

                y_lo = med - 1.4*y_diff
                y_hi = med + 1.1*y_diff

                ax.vlines(midtimes, y_lo, y_hi, color='orangered',
                          linestyle='--', zorder=1, lw=1, alpha=0.3)
                ax.set_ylim((y_lo, y_hi))

            txtstr = f"P: {period_val:.5f}, t0: {t0_val:.4f}"
            titlestr = titlestr + ', ' + txtstr

        #axs[1].legend(loc='best',fontsize='xx-small')
        axs[0].set_ylabel('pdcsap [ppt]')
        axs[0].set_title(titlestr, fontsize='x-small')
        axs[1].set_ylabel('notch [ppt]')
        axs[1].set_xlabel('time [days]')

        axs[0].set_xlim([_start, _stop])
        axs[1].set_xlim([_start, _stop])

        f.tight_layout()
        f.savefig(outpng, dpi=300)
        LOGINFO(f"Made {outpng}")


def plot_planet_finding_results(
    star_id, search_method, outdir, cachepath, dtr_dict, instrument='kepler',
    overwrite=0
    ):
    """
    Plots of phase-folded flux from run_periodograms_and_detrend, with other
    diagnostics (including odd-evens).
    """

    assert instrument in ['tess', 'kepler']
    assert search_method in ['bls', 'tls']
    assert os.path.exists(cachepath)

    with open(cachepath, 'rb') as f:
        d = pickle.load(f)

    r = d['r']
    if search_method == 'bls':
        pgr = d['bls_results']
        blsr = d['bls_results']
    if search_method == 'tls':
        pgr = d['tls_results']
        tlsr = d['tls_results']

    search_time = d['search_time']
    search_flux = d['search_flux']
    dtr_stages_dict = d['dtr_stages_dict']

    if dtr_stages_dict is not None:
        time, flux = dtr_stages_dict['time'], dtr_stages_dict['flux']
        sel0 = dtr_stages_dict['sel0']
        clipped_flux = dtr_stages_dict['clipped_flux']
        trend_flux = dtr_stages_dict['trend_flux']
        search_time = dtr_stages_dict['search_time']
        search_flux = dtr_stages_dict['search_flux']

    # for kepler, show quarter roll times.
    if instrument == 'kepler':
        from cdips.paths import DATADIR
        csvpath = os.path.join(
            DATADIR, 'spacecraft', 'time_to_kepler_quarter.csv'
        )
        time_df = pd.read_csv(csvpath)
        segment_id = np.array(time_df.quarter.apply(lambda x: str(x).zfill(2)))
        segment_start = np.array(time_df.tstart)
        segment_stop = np.array(time_df.tstop)

    elif instrument == 'tess':
        segment_id = ['tess']
        segment_start = [np.nanmin(time)]
        segment_stop = [np.nanmax(time)]

    outname =  (
        f"{star_id}_{search_method}"
        f"_method-{dtr_dict['method']}"
        f"_windowlength-{dtr_dict['window_length']}"
        ".png"
    )

    outpng = os.path.join(outdir, outname)

    if os.path.exists(outpng):
        LOGINFO(f"Found {outpng}, continue.")
        return 1

    fig = plt.figure(figsize=(14,15))
    axd = fig.subplot_mosaic(
        """
        000
        112
        342
        562
        """
    )

    # ax0: light curve
    # ax1: tls periodogram
    # ax2: text on right
    # ax3: primary transit
    # ax4: occultation
    # ax5: odd
    # ax6: even
    ax0 = axd['0']
    ax1 = axd['1']
    ax2 = axd['2']
    ax3 = axd['3']
    ax4 = axd['4']
    ax5 = axd['5']
    ax6 = axd['6']

    # ax0
    ax = ax0
    norm_y = lambda y: 1e3*(y-1)

    ax.scatter(
        search_time, norm_y(search_flux), c='black', s=2, zorder=2,
        rasterized=True, linewidths=0, marker='.'
    )

    if search_method == 'tls':
        period, t0, dur = tlsr['period'], tlsr['T0'], tlsr['duration']
    elif search_method == 'bls':
        period, t0, dur = r['bls_period'], r['bls_t0'], r['bls_duration']
    tdur_by_period = dur / period

    if np.isnan(period):
        LOGWARNING(f'Got NaN {search_method} period for {star_id}.  Escaping.')
        return 0
    midtimes = t0 + np.arange(-2000,2000,1)*period

    _f = norm_y(search_flux)
    med = np.nanmedian(_f)
    std = np.nanstd(_f)
    y_diff = np.nanpercentile(_f,90) - np.nanpercentile(_f,10)
    y_lo = med - 1.8*y_diff
    y_hi = med + 1.4*y_diff
    ax.vlines(midtimes, y_lo, y_hi, color='orangered',
              linestyle='--', zorder=1, lw=0.5, alpha=0.3)
    ax.set_ylim((y_lo, y_hi))

    ax.set_xlim((np.nanmin(search_time)-10, np.nanmax(search_time)+10))

    ax.update({'ylabel':'notch [ppt]', 'xlabel': 'time [days]'})

    # ax1
    ax = ax1

    if search_method == 'tls':
        powerkey = 'power'
        bestperiodkey = 'period'
    if search_method == 'bls':
        powerkey = 'lspvals'
        bestperiodkey = 'bestperiod'

    ax.plot(pgr['periods'], pgr[powerkey], c='k', lw=0.5)
    ax.axvline(pgr[bestperiodkey], alpha=0.4, lw=2, c='C0',
               label=f'{search_method} (Porb)')
    for n in [2,3,4,5,6,7]:
        ax.axvline(n * pgr[bestperiodkey], alpha=0.4, lw=0.5, ls="--", c='C0')
        ax.axvline(pgr[bestperiodkey] / n, alpha=0.4, lw=0.5, ls="--", c='C0')

    ls_period = r['ls_period']
    ax.axvline(ls_period, alpha=0.4, lw=2, c='C1', label='LS (Prot)')
    for n in [2,3,4]:
        ax.axvline(n * ls_period, alpha=0.4, lw=0.5, ls="--", c='C1')
        ax.axvline(ls_period / n, alpha=0.4, lw=0.5, ls="--", c='C1')

    ax.legend(loc='best', fontsize='x-small')
    ax.update({'ylabel':'power', 'xlabel': 'period [days]',
               'xlim':[0.99*min(pgr['periods']), 1.01*max(pgr['periods'])],
               'xscale': 'log'})

    # ax2
    ax = ax2

    if search_method == 'tls':
        txt = (
            f"{star_id}\n"
            f"dtr method: {dtr_stages_dict['dtr_method_used']}\n"
            f"search method: {search_method}\n"
            f"windowlength: {dtr_dict['window_length']} days\n"
            f"LS period (raw): {ls_period:.3f} days\n"
            f"\n"
            f"TLS results:\n"
            f"SDE: {tlsr['SDE']:.1f}\n"
            f"SNR: {tlsr['snr']:.1f}\n"
            f"P: {tlsr['period']:.4f} ± {tlsr['period_uncertainty']:.4f} days\n"
            f"t0: {tlsr['T0']:.4f}\n"
            f"dur: {24*tlsr['duration']:.1f} hr\n"
            f"δ: {1e3*(1-tlsr['depth_mean'][0]):.2f} ± {1e3*(tlsr['depth_mean'][1]):.2f} ppt\n"
            f"δeven: {1e3*(1-tlsr['depth_mean_even'][0]):.2f} ± {1e3*(tlsr['depth_mean_even'][1]):.2f} ppt\n"
            f"δodd: {1e3*(1-tlsr['depth_mean_odd'][0]):.2f} ± {1e3*(tlsr['depth_mean_odd'][1]):.2f} ppt\n"
            f"odd-even sig: {tlsr['odd_even_mismatch']:.1f}σ\n"
            f"Rp/R*: {tlsr['rp_rs']:.4f}\n"
            f"Ntra: {tlsr['transit_count']}\n"
            f"Nobs: {tlsr['distinct_transit_count']}\n"
        )
    elif search_method == 'bls':
        txt = (
            f"{star_id}\n"
            f"dtr method: {dtr_stages_dict['dtr_method_used']}\n"
            f"search method: {search_method}\n"
            f"windowlength: {dtr_dict['window_length']} days\n"
            f"LS period (raw): {ls_period:.3f} days\n"
            f"\n"
            f"BLS results:\n"
            f"SNR: {r['bls_snr']:.1f}\n"
            f"P: {r['bls_period']:.4f} days\n"
            f"t0: {r['bls_t0']:.4f}\n"
            f"dur: {24*r['bls_duration']:.1f} hr\n"
            f"δ: {1e3*(r['bls_depth']):.2f} ppt\n"
            f"δeven: TODO ppt\n"
            f"δodd: TODO ppt\n"
            f"odd-even sig: TODO σ\n"
            f"Ntra: {r['bls_npoints_in_transit']}\n"
        )

    ax.text(0, 0.5, txt, ha='left', va='center', fontsize='x-large', zorder=2,
            transform=ax.transAxes)
    ax.set_axis_off()

    # ax3: primary transit
    from astrobase.checkplot.png import _make_phased_magseries_plot
    from astrobase.lcmath import phase_magseries
    ax = ax3

    phasebin = 1e-3
    minbinelems = 2
    phasems = 2.0
    phasebinms = 6.0

    plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
    _make_phased_magseries_plot(ax, 0, search_time, norm_y(search_flux),
                                np.ones_like(search_flux)/1e4, period, t0,
                                True, True, phasebin, minbinelems, plotxlim,
                                search_method, xliminsetmode=False,
                                magsarefluxes=True, phasems=phasems,
                                phasebinms=phasebinms, verbose=True,
                                lowerleftstr='primary',
                                lowerleftfontsize='small')

    if search_method == 'tls':
        model_time = tlsr['model_lightcurve_time']
        model_y = tlsr['model_lightcurve_model']
        phasedlc = phase_magseries(
            model_time, model_y, period, t0, wrap=True, sort=True
        )
        plotphase = phasedlc['phase']
        plotmags = phasedlc['mags']

    elif search_method == 'bls':
        plotphase = blsr['stats'][0]['phases']
        plotmags = blsr['stats'][0]['phasedmags']

    ax.plot(plotphase, norm_y(plotmags), zorder=0, color='gray')
    ax.set_ylabel('flux [ppt]')
    ax.set_ylim((y_lo, y_hi))

    # ax4: occultation
    ax = ax4
    plotxlim=(-2.0*tdur_by_period+0.5,2.0*tdur_by_period+0.5)
    _make_phased_magseries_plot(ax, 0, search_time, norm_y(search_flux),
                                np.ones_like(search_flux)/1e4, period, t0,
                                True, True, phasebin, minbinelems, plotxlim,
                                search_method, xliminsetmode=False,
                                magsarefluxes=True, phasems=phasems,
                                phasebinms=phasebinms, verbose=True,
                                lowerleftstr='secondary',
                                lowerleftfontsize='small')
    ax.plot(plotphase, norm_y(plotmags), zorder=0, color='gray')
    ax.set_ylabel('flux [ppt]')
    ax.set_ylim((y_lo, y_hi))

    # ax5: odd
    ax = ax5

    all_midtimes = t0 + period*np.arange(-10000,10000,1)
    obsd_midtimes = all_midtimes[
        (all_midtimes > np.nanmin(search_time))
        &
        (all_midtimes < np.nanmax(search_time))
    ]

    even_midtimes = obsd_midtimes[::2]
    odd_midtimes = obsd_midtimes[1::2]

    delta_t = 0.245*period
    even_windows = np.array((even_midtimes - delta_t, even_midtimes+delta_t))

    even_mask = np.zeros_like(search_time).astype(bool)
    for even_window in even_windows.T:
        even_mask |= np.array(
            (search_time > np.min(even_window)) & (search_time < np.max(even_window))
        )
    odd_mask = ~even_mask

    plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
    _make_phased_magseries_plot(ax, 0, search_time[odd_mask],
                                norm_y(search_flux[odd_mask]),
                                np.ones_like(search_flux[odd_mask])/1e4,
                                period, t0, True, True, phasebin, minbinelems,
                                plotxlim, search_method, xliminsetmode=False,
                                magsarefluxes=True, phasems=phasems,
                                phasebinms=phasebinms, verbose=True,
                                lowerleftstr='odd', lowerleftfontsize='small')
    ax.plot(plotphase, norm_y(plotmags), zorder=0, color='gray')
    ax.set_ylabel('flux [ppt]')
    ax.set_ylim((y_lo, y_hi))

    # ax6: even
    ax = ax6
    plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
    _make_phased_magseries_plot(ax, 0, search_time[even_mask],
                                norm_y(search_flux[even_mask]),
                                np.ones_like(search_flux[even_mask])/1e4,
                                period, t0, True, True, phasebin, minbinelems,
                                plotxlim, search_method, xliminsetmode=False,
                                magsarefluxes=True, phasems=phasems,
                                phasebinms=phasebinms, verbose=True,
                                lowerleftstr='even', lowerleftfontsize='small')
    ax.plot(plotphase, norm_y(plotmags), zorder=0, color='gray')
    ax.set_ylabel('flux [ppt]')
    ax.set_ylim((y_lo, y_hi))

    fig.tight_layout()
    fig.savefig(outpng, dpi=300)
    LOGINFO(f"Made {outpng}")


def plot_iterative_planet_finding_results(
    star_id, search_method, outdir, cachepath, dtr_dict, instrument='kepler',
    overwrite=False
    ):
    """
    Plots of phase-folded flux from detrend_and_iterative_tls, with other
    diagnostics (including odd-evens).
    """

    assert instrument in ['tess', 'kepler']
    assert search_method in ['tls']
    assert os.path.exists(cachepath)

    with open(cachepath, 'rb') as f:
        d = pickle.load(f)

    dtr_stages_dict = d['dtr_stages_dict']

    iter_keys = sorted([int(k) for k in d.keys() if k != 'dtr_stages_dict'])
    n_iter = max(iter_keys) + 1

    # for kepler, show quarter roll times.
    if instrument == 'kepler':
        from cdips.paths import DATADIR
        csvpath = os.path.join(
            DATADIR, 'spacecraft', 'time_to_kepler_quarter.csv'
        )
        time_df = pd.read_csv(csvpath)
        segment_id = np.array(time_df.quarter.apply(lambda x: str(x).zfill(2)))
        segment_start = np.array(time_df.tstart)
        segment_stop = np.array(time_df.tstop)

    elif instrument == 'tess':
        segment_id = ['tess']
        segment_start = [np.nanmin(time)]
        segment_stop = [np.nanmax(time)]

    outname =  (
        f"{star_id}_{search_method}"
        f"_method-{dtr_dict['method']}"
        f"_windowlength-{dtr_dict['window_length']}"
        ".pdf"
    )

    outpdf = os.path.join(outdir, outname)

    if os.path.exists(outpdf) and not overwrite:
        LOGINFO(f"Found {outpdf}, continue.")
        return 1

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(outpdf) as pdf:

        for iter_key in iter_keys:

            LOGINFO(f"{star_id}: beginning iter {iter_key} of PDF report.")

            r = d[iter_key]['r']
            pgr = d[iter_key]['tls_results']
            tlsr = d[iter_key]['tls_results']

            search_time = d[iter_key]['search_time']
            search_flux = d[iter_key]['search_flux']

            time, flux = dtr_stages_dict['time'], dtr_stages_dict['flux']
            sel0 = dtr_stages_dict['sel0']
            clipped_flux = dtr_stages_dict['clipped_flux']
            trend_flux = dtr_stages_dict['trend_flux']
            search_time = dtr_stages_dict['search_time']
            search_flux = dtr_stages_dict['search_flux']

            fig = plt.figure(figsize=(14,15))
            axd = fig.subplot_mosaic(
                """
                000
                112
                342
                562
                """
            )

            # ax0: light curve
            # ax1: tls periodogram
            # ax2: text on right
            # ax3: primary transit
            # ax4: occultation
            # ax5: odd
            # ax6: even
            ax0 = axd['0']
            ax1 = axd['1']
            ax2 = axd['2']
            ax3 = axd['3']
            ax4 = axd['4']
            ax5 = axd['5']
            ax6 = axd['6']

            # ax0
            ax = ax0
            norm_y = lambda y: 1e3*(y-1)

            ax.scatter(
                search_time, norm_y(search_flux), c='black', s=2, zorder=2,
                rasterized=True, linewidths=0, marker='.'
            )

            period, t0, dur = tlsr['period'], tlsr['T0'], tlsr['duration']
            tdur_by_period = dur / period

            if np.isnan(period):
                LOGWARNING(
                    f'Got NaN {search_method} period for {star_id}. '
                    'Escaping.'
                )
                return 0
            midtimes = t0 + np.arange(-2000,2000,1)*period

            _f = norm_y(search_flux)
            med = np.nanmedian(_f)
            std = np.nanstd(_f)
            y_diff = np.nanpercentile(_f,90) - np.nanpercentile(_f,10)
            y_lo = med - 1.8*y_diff
            y_hi = med + 1.4*y_diff
            ax.vlines(midtimes, y_lo, y_hi, color='orangered',
                      linestyle='--', zorder=1, lw=0.5, alpha=0.3)
            ax.set_ylim((y_lo, y_hi))

            ax.set_xlim((np.nanmin(search_time)-10, np.nanmax(search_time)+10))

            ax.update({'ylabel':'notch [ppt]', 'xlabel': 'time [days]'})

            # ax1
            ax = ax1

            powerkey = 'power'
            bestperiodkey = 'period'
            ax.plot(pgr['periods'], pgr[powerkey], c='k', lw=0.5)
            ax.axvline(pgr[bestperiodkey], alpha=0.4, lw=2, c='C0',
                       label=f'{search_method} (Porb)')
            for n in [2,3,4,5,6,7]:
                ax.axvline(n * pgr[bestperiodkey], alpha=0.4, lw=0.5,
                           ls="--", c='C0')
                ax.axvline(pgr[bestperiodkey] / n, alpha=0.4, lw=0.5,
                           ls="--", c='C0')

            ls_period = r['ls_period']
            ax.axvline(ls_period, alpha=0.4, lw=2, c='C1', label='LS (Prot)')
            for n in [2,3,4]:
                ax.axvline(n * ls_period, alpha=0.4, lw=0.5, ls="--", c='C1')
                ax.axvline(ls_period / n, alpha=0.4, lw=0.5, ls="--", c='C1')

            ax.legend(loc='best', fontsize='x-small')
            ax.update({'ylabel':'power', 'xlabel': 'period [days]',
                       'xlim':[0.99*min(pgr['periods']), 1.01*max(pgr['periods'])],
                       'xscale': 'log'})

            # ax2
            ax = ax2

            txt = (
                f"{star_id}\n"
                f"dtr method: {dtr_stages_dict['dtr_method_used']}\n"
                f"search method: {search_method}\n"
                f"iteration: {iter_key}\n"
                f"windowlength: {dtr_dict['window_length']} days\n"
                f"LS period (raw): {ls_period:.3f} days\n"
                f"\n"
                f"TLS results:\n"
                f"SDE: {tlsr['SDE']:.1f}\n"
                f"SNR: {tlsr['snr']:.1f}\n"
                f"P: {tlsr['period']:.4f} ± {tlsr['period_uncertainty']:.4f} days\n"
                f"t0: {tlsr['T0']:.4f}\n"
                f"dur: {24*tlsr['duration']:.1f} hr\n"
                f"δ: {1e3*(1-tlsr['depth_mean'][0]):.2f} ± {1e3*(tlsr['depth_mean'][1]):.2f} ppt\n"
                f"δeven: {1e3*(1-tlsr['depth_mean_even'][0]):.2f} ± {1e3*(tlsr['depth_mean_even'][1]):.2f} ppt\n"
                f"δodd: {1e3*(1-tlsr['depth_mean_odd'][0]):.2f} ± {1e3*(tlsr['depth_mean_odd'][1]):.2f} ppt\n"
                f"odd-even sig: {tlsr['odd_even_mismatch']:.1f}σ\n"
                f"Rp/R*: {tlsr['rp_rs']:.4f}\n"
                f"Ntra: {tlsr['transit_count']}\n"
                f"Nobs: {tlsr['distinct_transit_count']}\n"
            )

            ax.text(0, 0.5, txt, ha='left', va='center', fontsize='x-large',
                    zorder=2, transform=ax.transAxes)
            ax.set_axis_off()

            # ax3: primary transit
            from astrobase.checkplot.png import _make_phased_magseries_plot
            from astrobase.lcmath import phase_magseries
            ax = ax3

            phasebin = 1e-3
            minbinelems = 2
            phasems = 2.0
            phasebinms = 6.0

            plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
            _make_phased_magseries_plot(ax, 0, search_time,
                                        norm_y(search_flux),
                                        np.ones_like(search_flux)/1e4, period,
                                        t0, True, True, phasebin, minbinelems,
                                        plotxlim, search_method,
                                        xliminsetmode=False,
                                        magsarefluxes=True, phasems=phasems,
                                        phasebinms=phasebinms, verbose=True,
                                        lowerleftstr='primary',
                                        lowerleftfontsize='small')

            model_time = tlsr['model_lightcurve_time']
            model_y = tlsr['model_lightcurve_model']
            phasedlc = phase_magseries(
                model_time, model_y, period, t0, wrap=True, sort=True
            )
            plotphase = phasedlc['phase']
            plotmags = phasedlc['mags']

            ax.plot(plotphase, norm_y(plotmags), zorder=0, color='gray')
            ax.set_ylabel('flux [ppt]')
            ax.set_ylim((y_lo, y_hi))

            # ax4: occultation
            ax = ax4
            plotxlim=(-2.0*tdur_by_period+0.5,2.0*tdur_by_period+0.5)
            _make_phased_magseries_plot(ax, 0, search_time,
                                        norm_y(search_flux),
                                        np.ones_like(search_flux)/1e4, period,
                                        t0, True, True, phasebin, minbinelems,
                                        plotxlim, search_method,
                                        xliminsetmode=False,
                                        magsarefluxes=True, phasems=phasems,
                                        phasebinms=phasebinms, verbose=True,
                                        lowerleftstr='secondary',
                                        lowerleftfontsize='small')
            ax.plot(plotphase, norm_y(plotmags), zorder=0, color='gray')
            ax.set_ylabel('flux [ppt]')
            ax.set_ylim((y_lo, y_hi))

            # ax5: odd
            ax = ax5

            all_midtimes = t0 + period*np.arange(-10000,10000,1)
            obsd_midtimes = all_midtimes[
                (all_midtimes > np.nanmin(search_time))
                &
                (all_midtimes < np.nanmax(search_time))
            ]

            even_midtimes = obsd_midtimes[::2]
            odd_midtimes = obsd_midtimes[1::2]

            delta_t = 0.245*period
            even_windows = np.array(
                (even_midtimes - delta_t, even_midtimes+delta_t)
            )

            even_mask = np.zeros_like(search_time).astype(bool)
            for even_window in even_windows.T:
                even_mask |= np.array(
                    (search_time > np.min(even_window)) &
                    (search_time < np.max(even_window))
                )
            odd_mask = ~even_mask

            plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
            _make_phased_magseries_plot(ax, 0, search_time[odd_mask],
                                        norm_y(search_flux[odd_mask]),
                                        np.ones_like(search_flux[odd_mask])/1e4,
                                        period, t0, True, True, phasebin,
                                        minbinelems, plotxlim, search_method,
                                        xliminsetmode=False,
                                        magsarefluxes=True, phasems=phasems,
                                        phasebinms=phasebinms, verbose=True,
                                        lowerleftstr='odd',
                                        lowerleftfontsize='small')
            ax.plot(plotphase, norm_y(plotmags), zorder=0, color='gray')
            ax.set_ylabel('flux [ppt]')
            ax.set_ylim((y_lo, y_hi))

            # ax6: even
            ax = ax6
            plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
            _make_phased_magseries_plot(ax, 0, search_time[even_mask],
                                        norm_y(search_flux[even_mask]),
                                        np.ones_like(search_flux[even_mask])/1e4,
                                        period, t0, True, True, phasebin,
                                        minbinelems, plotxlim, search_method,
                                        xliminsetmode=False,
                                        magsarefluxes=True, phasems=phasems,
                                        phasebinms=phasebinms, verbose=True,
                                        lowerleftstr='even',
                                        lowerleftfontsize='small')
            ax.plot(plotphase, norm_y(plotmags), zorder=0, color='gray')
            ax.set_ylabel('flux [ppt]')
            ax.set_ylim((y_lo, y_hi))

            fig.tight_layout()

            # finish this *page*
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close()

        # finish the multi-page pdf report
        LOGINFO(f"Made {outpdf}")



###########
# helpers #
###########
def _p2p_rms(flux):
    import cdips.utils.lcutils as lcu
    return lcu.p2p_rms(flux)
