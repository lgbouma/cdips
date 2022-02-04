"""
Contents:
    run_periodograms_and_detrend: given source_id, time, mag, find planet.
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

import os
import numpy as np, pandas as pd
from cdips.lcproc import detrend as dtr

import multiprocessing as mp
nworkers = mp.cpu_count()

def run_periodograms_and_detrend(source_id, time, mag, dtr_dict,
                                 period_min=0.5, period_max=27, orbitgap=1,
                                 expected_norbits=2, orbitpadding=6/(24),
                                 dtr_method='best', return_extras=False,
                                 magisflux=False):
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

        return_extras (bool): default False.  If True, returns the search_time,
        search_flux, and dtr_stages_dict along with the summary row.

        magisflux (bool): default False

    Returns:
        If return_extras is False:
            r = [source_id, ls_period, ls_fap, ls_amplitude, tls_period, tls_sde,
                 tls_t0, tls_depth, tls_duration, tls_distinct_transit_count,
                 tls_odd_even, dtr_method]
        else:
            tuple of: r, search_time, search_flux, dtr_stages_dict.
            `dtr_stages_dict` contains np.ndarrays from each step of the
            detrending process, and Lomb-Scargle information from the
            pre-requisite rotation period check.
    """

    lsp_options = {'period_min':0.1, 'period_max':20}

    search_time, search_flux, dtr_stages_dict = (
        dtr.clean_rotationsignal_tess_singlesector_light_curve(
            time, mag, magisflux=magisflux, dtr_dict=dtr_dict,
            lsp_dict=None, maskorbitedge=True, lsp_options=lsp_options,
            verbose=False
        )
    )

    # retrieve LS periodogram information
    ls_period = dtr_stages_dict['lsp_dict']['ls_period']
    ls_amplitude = np.abs(dtr_stages_dict['lsp_dict']['ls_amplitude'])
    ls_fap = dtr_stages_dict['lsp_dict']['ls_fap']

    # run the TLS periodogram
    model = transitleastsquares(search_time, search_flux, verbose=False)
    results = model.power(use_threads=1, show_progress_bar=False,
                          R_star_min=0.1, R_star_max=5, M_star_min=0.1,
                          M_star_max=3.0, period_min=period_min,
                          period_max=period_max, n_transits_min=1,
                          transit_template='default', oversampling_factor=5)

    dtr_method = dtr_stages_dict['dtr_method_used']

    r = {
        'source_id': source_id,
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

    if not return_extras:
        return r
    else:
        return r, search_time, search_flux, dtr_stages_dict



