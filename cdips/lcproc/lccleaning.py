"""
cleaning.py - Luke Bouma (bouma.luke@gmail)

Contents:

    basic_cleaning: sliding sigma clipper, with nan-masking

    iterativegp_cleaning: fit GP to LC while iteratively excluding outliers

    _rotation_period: Lomb-Scargle wrapper.

"""
#############
## LOGGING ##
#############

import logging
from cdips import log_sub, log_fmt, log_date_fmt

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
import os, shutil, socket
from os.path import join
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits

from datetime import datetime
from glob import glob
from copy import deepcopy

from numpy import array as nparr, all as npall, isfinite as npisfinite

from astropy.timeseries import LombScargle

# NOTE: some code duplication with cdips.testing.check_dependencies
from wotan import version
wotanversion = version.WOTAN_VERSIONING
wotanversiontuple = tuple(wotanversion.split('.'))
assert int(wotanversiontuple[0]) >= 1
assert int(wotanversiontuple[1]) >= 9

def _rotation_period(time, flux,
                     lsp_options={'period_min':0.1, 'period_max':20}):

    # Drop NaNs & median normalize.
    mask = np.isnan(flux) | np.isnan(time)
    time = time[~mask].astype(float)
    flux = flux[~mask].astype(float)

    flux_median = np.median(flux)
    flux /= flux_median

    # Calculate the periodogram.
    period_min = lsp_options['period_min']
    period_max = lsp_options['period_max']

    ls = LombScargle(time, flux, flux*1e-3)
    freq, power = ls.autopower(
        minimum_frequency=1/period_max, maximum_frequency=1/period_min
    )
    ls_fap = ls.false_alarm_probability(power.max())
    best_freq = freq[np.argmax(power)]
    ls_period = 1/best_freq
    theta = ls.model_parameters(best_freq)
    ls_amplitude = theta[1]

    lsp_dict = {}
    lsp_dict['ls_period'] = ls_period
    lsp_dict['ls_amplitude'] = np.abs(ls_amplitude)
    lsp_dict['ls_fap'] = ls_fap

    return lsp_dict


def iterativegp_cleaning(time, flux,
                         nll, gp_params, mu,
                         N_iter=3,
                         sigma_clip=3,
                         clipwindow=3,
                         verbose=True):
    """
    Fit GP to LC while iteratively excluding outliers.
    Call only really makes sense in context of lcproc/nuance_planet_search.py

    NB: The main point of this routine isn't really to "clean" the light curve,
    it's to obtain reasonable GP parameters.

    Args:
        time (np.ndarray): Array of time values for the light curve.
        flux (np.ndarray): Array of flux values for the light curve.
        clipwindow (int): Number of points around outliers to also drop.

    Returns:
        search_time, search_flux, dtr_stages_dict (tuple):
    """

    from nuance.utils import minimize, sigma_clip_mask
    outlier_mask = np.ones_like(time).astype(bool)

    for _ in range(N_iter):
        residuals = flux - mu(gp_params)
        outlier_mask = (
            outlier_mask &
            sigma_clip_mask(residuals, sigma=sigma_clip,
                            window=clipwindow)
        )
        gp_params = minimize(nll, gp_params)

    search_time = time[outlier_mask]
    search_flux = flux[outlier_mask]

    dtr_stages_dict = {
        # initial time and flux.
        'time': time,
        'flux': flux,
        # times and fluxes used
        'search_time': search_time,
        'search_flux': search_flux
    }
    return search_time, search_flux, gp_params, dtr_stages_dict



def basic_cleaning(time, flux,
                   lsp_options={'period_min':0.1, 'period_max':20},
                   verbose=True,
                   slide_clip_lo=20,
                   slide_clip_hi=3,
                   clip_window=3):
    """
    NaN-mask and sliding sigma-clip a light curve.  Run LS on it as well.

    Args:
        time (np.ndarray): Array of time values for the light curve.
        flux (np.ndarray): Array of flux values for the light curve.

    Returns:
        search_time, search_flux, dtr_stages_dict (tuple):
    """

    mask = np.isnan(flux) | np.isnan(time)
    time = time[~mask].astype(float)
    flux = flux[~mask].astype(float)

    flux_median = np.median(flux)
    flux /= flux_median
    flux_err /= flux_median

    # identify rotation period
    lsp_dict = _rotation_period(time, flux, lsp_options)
    LOGINFO(f"Prot = {lsp_dict['ls_period']:.3f} days")

    #
    # sliding sigma clip asymmetric [20,3]*MAD, about median. use a 3-day
    # window, to give ~100 to 150 data points at 30-minute cadence. mostly
    # avoids big flares, provided restrictive slide_clip_lo and slide_clip_hi
    # are given.
    #
    from wotan import slide_clip
    LOGINFO(f"Slide clipping MAD over window={clip_window}, "
            f"lo={slide_clip_lo}, hi={slide_clip_hi}")
    clipped_flux = slide_clip(
        time, flux, window_length=clip_window, low=slide_clip_lo,
        high=slide_clip_hi, method='mad', center='median'
    )
    sel0 = ~np.isnan(clipped_flux) & (clipped_flux != 0)

    search_time = time[sel0]
    search_flux = clipped_flux[sel0]

    dtr_stages_dict = {
        # initial time and flux.
        'time': time,
        'flux': flux,
        # after initial window sigma_clip on flux, what is left?
        'clipped_flux': clipped_flux,
        # non-nan indices from clipped_flux
        'sel0': sel0,
        # times and fluxes used
        'search_time': search_time,
        'search_flux': search_flux
    }
    if isinstance(lsp_dict, dict):
        # in most cases, cache the LS period, amplitude, and FAP
        dtr_stages_dict['lsp_dict'] = lsp_dict

    return search_time, search_flux, dtr_stages_dict

