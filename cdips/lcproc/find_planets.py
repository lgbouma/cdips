"""
Contents:
    run_periodograms_and_detrend: given star_id, time, mag, find planet.
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

import os, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from cdips.lcproc import detrend as dtr

import multiprocessing as mp
nworkers = mp.cpu_count()

def run_periodograms_and_detrend(
    star_id, time, mag, dtr_dict,
    period_min=0.5, period_max=27,
    R_star_min=0.1, R_star_max=5,
    M_star_min=0.1, M_star_max=3.0,
    n_transits_min=1, oversampling_factor=5,
    orbitgap=1,
    expected_norbits=2, orbitpadding=6/(24),
    dtr_method='best', n_threads=1,
    return_extras=False, magisflux=False,
    cachepath=None, verbose=False
    ):
    """
    Given a star_id, time, and magnitude time-series, this function runs
    clean_rotationsignal_tess_singlesector_light_curve to remove rotation
    signals (via masking orbit edges, sigma slide clip, detrending, and
    re-sigma slide clipping).  "Detrending" here means the "best" method
    currently known, which is the notch + locor combination (which was
    demonstrated via /tests/test_injrecov_with_detrending.py). TLS is then run
    on the flux residuals from the detrending (not e.g., the BIC time-series).

    kwargs:

        star_id: used for bookkeeping, can be any int/str.

        time, mag : time and magnitude vector of light-curve.  PCA is
        preferred, since common instrumental systematics are removed.

        dtr_dict : E.g.,
            {'method':'best', 'break_tolerance':0.5, 'window_length':0.5}

        orbitpadding (float): amount of time to clip near TESS perigee to
        remove ramp signals. 12 data points = 6 hours = 0.25 days (and must
        give in units of days).

        dtr_method (str): any of ['notch', 'locor', 'pspline', 'best'].

        return_extras (bool): default False.  If True, returns the search_time,
        search_flux, and dtr_stages_dict along with the summary row.

        magisflux (bool): default False

        cachepath (str): if a pickle file path is passed, results (tuple case
        described below) will be cached to this pickle file.

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

    # if this has been run before, load from a cache.  notch is pretty slow.
    if isinstance(cachepath, str):

        assert cachepath.endswith(".pkl")

        if os.path.exists(cachepath):

            LOGINFO(f"Found {cachepath}, loading results.")

            with open(cachepath, 'rb') as f:
                d = pickle.load(f)

            if return_extras:
                return (
                    d['r'], d['search_time'], d['search_flux'],
                    d['dtr_stages_dict']
                )
            else:
                return d['r']

    # otherwise, run the detrending
    lsp_options = {'period_min':0.1, 'period_max':20}
    search_time, search_flux, dtr_stages_dict = (
        dtr.clean_rotationsignal_tess_singlesector_light_curve(
            time, mag, magisflux=magisflux, dtr_dict=dtr_dict,
            lsp_dict=None, maskorbitedge=True, lsp_options=lsp_options,
            verbose=verbose
        )
    )

    # retrieve LS periodogram information
    ls_period = dtr_stages_dict['lsp_dict']['ls_period']
    ls_amplitude = np.abs(dtr_stages_dict['lsp_dict']['ls_amplitude'])
    ls_fap = dtr_stages_dict['lsp_dict']['ls_fap']

    # run the TLS periodogram
    model = transitleastsquares(search_time, search_flux, verbose=verbose)
    results = model.power(use_threads=n_threads, show_progress_bar=verbose,
                          R_star_min=R_star_min, R_star_max=R_star_max,
                          M_star_min=M_star_min, M_star_max=M_star_max,
                          period_min=period_min, period_max=period_max,
                          n_transits_min=n_transits_min,
                          transit_template='default',
                          oversampling_factor=oversampling_factor)

    dtr_method = dtr_stages_dict['dtr_method_used']

    r = {
        'star_id': star_id,
        'ls_period': ls_period,
        'ls_fap': ls_fap,
        'ls_amplitude': ls_amplitude,
        'tls_period': results.period,
        'tls_sde': results.SDE,
        'tls_snr': results.snr,
        'tls_t0': results.T0,
        'tls_depth': results.depth,
        'tls_duration': results.duration,
        # The number of transits with intransit data points
        'tls_distinct_transit_count': results.distinct_transit_count,
        'tls_odd_even': results.odd_even_mismatch,
        'dtr_method': dtr_method
    }

    if isinstance(cachepath, str):
        outdict = {
            'r':r,
            'search_time':search_time,
            'search_flux':search_flux,
            'dtr_stages_dict':dtr_stages_dict
        }
        with open(cachepath, 'wb') as f:
            pickle.dump(
                outdict, f
            )
            LOGINFO(f"Wrote {cachepath}")

    if not return_extras:
        return r
    else:
        return r, search_time, search_flux, dtr_stages_dict


def plot_detrend_check(star_id, outdir, dtr_dict, dtr_stages_dict,
                       r=None, instrument='tess', cdipslcpath=None):

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
            str(star_id)+
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
        segment_id = time_df.quarter
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
            f"_{_segid}"
            f"_method-{dtr_dict['method']}"
            f"_windowlength-{dtr_dict['window_length']}"
            ".png"
        )

        outpng = os.path.join(outdir, outname)

        if os.path.exists(outpng):
            LOGINFO(f"Found {outpng}, continue.")
            continue

        f,axs = plt.subplots(nrows=2, sharex=True, figsize=(12,7))
        # lower: "raw" data; upper: sigma-clipped
        axs[0].scatter(
            time, clipped_flux, c='black', s=1, zorder=2,
            rasterized=True
        )
        axs[0].scatter(
            time, flux, c='red', s=1, zorder=1, rasterized=True
        )
        axs[0].plot(time[sel0], trend_flux, c='C0')
        axs[1].scatter(
            search_time, search_flux, c='black', s=1,
            zorder=2, rasterized=True, label='searched flux'
        )

        titlestr = (
            f"{star_id}"
        )

        if r is not None:
            period_val = r['tls_period']
            t0_val = r['tls_t0']
            tdur_val = r['tls_duration']

            midtimes = t0_val + np.arange(-2000,2000,1)*period_val

            for ax in axs:
                ylim = ax.get_ylim()
                ax.vlines(midtimes, min(ylim), max(ylim), color='orangered',
                          linestyle='--', zorder=1, lw=2, alpha=0.3)
                ax.set_ylim((min(ylim), max(ylim)))

            txtstr = f"P: {period_val:.5f}, t0: {t0_val:.4f}"
            titlestr = titlestr + ' ' + txtstr

        axs[1].legend(loc='best',fontsize='xx-small')
        axs[0].set_ylabel(f'flux')
        axs[0].set_title(titlestr, fontsize='x-small')
        axs[1].set_ylabel('flattened')
        axs[1].set_xlabel('time [days]')

        axs[0].set_xlim([_start, _stop])
        axs[1].set_xlim([_start, _stop])

        f.savefig(outpng, dpi=300)
        LOGINFO(f"Made {outpng}")
