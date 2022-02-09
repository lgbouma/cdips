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

from astrobase import lcmath
import numpy as np, pandas as pd
import os, textwrap
from glob import glob
from datetime import datetime

def mask_orbit_start_and_end(time, flux, flux_err=None, orbitgap=1, expected_norbits=2,
                             orbitpadding=6/(24),
                             raise_expectation_error=True,
                             return_inds=False, verbose=True):
    """
    Trim out the times near the edges of orbits.

    args:
        time, flux
    returns:
        time, flux: with `orbitpadding` days trimmed out
    """
    norbits, groups = lcmath.find_lc_timegroups(time, mingap=orbitgap)

    if norbits != expected_norbits:
        wrnmsg = 'got {} orbits, expected {}. groups are {}'.format(
            norbits, expected_norbits, repr(groups))

        if raise_expectation_error:
            raise AssertionError(wrnmsg)
        elif norbits > 0 and not raise_expectation_error and verbose:
            LOGINFO(f'WRN! {wrnmsg}')
        elif norbits > 0 and not raise_expectation_error and not verbose:
            # suppress the warning, just mask the edges
            pass
        else:
            # no matter what, if you don't get any data, raise the assertion
            # error.
            raise AssertionError(wrnmsg)

    sel = np.zeros_like(time).astype(bool)
    for group in groups:
        tg_time = time[group]
        start_mask = (np.min(tg_time), np.min(tg_time) + orbitpadding)
        end_mask = (np.max(tg_time) - orbitpadding, np.max(tg_time))
        sel |= (
            (time > max(start_mask)) & (time < min(end_mask))
        )

    return_time = time[sel]
    return_flux = flux[sel]
    if flux_err is not None:
        return_flux_err = flux_err[sel]

    if not return_inds and flux_err is None:
        return return_time, return_flux
    elif return_inds and flux_err is None:
        return return_time, return_flux, sel
    elif not return_inds and flux_err is not None:
        return return_time, return_flux, return_flux_err
    elif return_inds and flux_err is not None:
        return return_time, return_flux, return_flux_err, sel
