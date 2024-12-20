"""
Contents:
    find_good_peaks
    flag_harmonics
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
import numpy as np
from numpy import array as nparr

def find_good_peaks(period_grid, power_vals, periodepsilon=0.1, nbestpeaks=5,
                    param_dict=None):
    """Find distinct peaks in a periodogram.

    This function identifies the `nbestpeaks` distinct peaks in a periodogram
    generated from a given set of periods and their associated power values.
    It avoids counting broad peaks multiple times by grouping them based on a
    fractional separation threshold `periodepsilon`.

    This implementation is adapted from:
    https://github.com/waqasbhatti/astrobase

    Args:
        period_grid (numpy.ndarray): Array of searched periods.
        power_vals (numpy.ndarray): Corresponding power values for each period.
        periodepsilon (float): Fractional threshold for considering periods
            distinct peaks. Peaks differing by less than this fraction are
            grouped together. Defaults to 0.1.
        nbestpeaks (int): Number of best peaks to return. Defaults to 5.
        param_dict (dict, optional): Dictionary with keys 't0', 'tdur', and 'P',
            each an ndarray of the same length as `period_grid`. These
            parameters, if provided, are included in the returned dictionary.

    Returns:
        dict: Dictionary containing:
            - 'bestperiod' (float): Best identified period.
            - 'bestpowerval' (float): Power value of the best period.
            - 'nbestpeaks' (int): Number of best peaks found.
            - 'nbestpowervals' (list): Power values of the best peaks.
            - 'nbestperiods' (list): Periods of the best peaks.
            - 'nbestparams' (dict): Associated parameters for the best peaks.
            - 'powervals' (numpy.ndarray): Input power values.
            - 'periods' (numpy.ndarray): Input periods.
    """

    if isinstance(param_dict, dict):
        assert 't0' in param_dict
        assert len(param_dict['t0']) == len(period_grid)

    finitepeakind = np.isfinite(power_vals) & np.isfinite(period_grid)
    finpower = power_vals[finitepeakind]
    finperiods = period_grid[finitepeakind]

    if isinstance(param_dict, dict):
        fint0 = param_dict['t0'][finitepeakind]
        fintdur = param_dict['tdur'][finitepeakind]
        finP = param_dict['P'][finitepeakind]
    else:
        fint0, fintdur, finP = None, None, None


    # make sure that finpower has finite values before we work on it
    try:

        bestperiodind = np.argmax(finpower)

    except ValueError:

        LOGERROR('no finite periodogram values '
                 'for this mag series, skipping...')
        return {'bestperiod':np.nan,
                'bestpowerval':np.nan,
                'nbestpeaks':nbestpeaks,
                'nbestpowervals':None,
                'nbestperiods':None,
                'nbestparams':None,
                'powervals':None,
                'periods':periods}

    sortedpowerind = np.argsort(finpower)[::-1]
    sortedperiods = finperiods[sortedpowerind]
    sortedpowervals = finpower[sortedpowerind]

    if isinstance(param_dict, dict):
        sortedt0 = fint0[sortedpowerind]
        sortedtdur = fintdur[sortedpowerind]
        sortedP = finP[sortedpowerind]
    else:
        sortedt0, sortedtdur, sortedP = None, None, None

    # now get the nbestpeaks
    nbestperiods, nbestpowervals, peakcount = (
        [finperiods[bestperiodind]],
        [finpower[bestperiodind]],
        1
    )
    prevperiod = sortedperiods[0]

    if isinstance(param_dict, dict):
        nbestparams = {
            't0': [fint0[bestperiodind]],
            'tdur': [fintdur[bestperiodind]],
            'P': [finP[bestperiodind]]
        }
    else:
        nbestparams = None

    # find the best nbestpeaks in the power and their periods
    for i, (period, powerval) in enumerate(zip(sortedperiods, sortedpowervals)):

        if peakcount == nbestpeaks:
            break
        perioddiff = abs(period - prevperiod)
        bestperiodsdiff = [abs(period - x) for x in nbestperiods]

        # this ensures that this period is different from the last
        # period and from all the other existing best periods by
        # periodepsilon to make sure we jump to an entire different peak
        # in the periodogram
        if (perioddiff > (periodepsilon*prevperiod) and
            all(x > (periodepsilon*period) for x in bestperiodsdiff)):
            nbestperiods.append(period)
            nbestpowervals.append(powerval)
            if isinstance(param_dict, dict):
                nbestparams['t0'].append(sortedt0[i])
                nbestparams['tdur'].append(sortedtdur[i])
                nbestparams['P'].append(sortedP[i])
            peakcount = peakcount + 1

        prevperiod = period

    if nbestparams is not None:
        nbestparamsout = {k:nparr(v) for k,v in nbestparams.items()}
    else:
        nbestparamsout = None

    return {'bestperiod':finperiods[bestperiodind],
            'bestpowerval':finpower[bestperiodind],
            'nbestpeaks':nbestpeaks,
            'nbestpowervals':nparr(nbestpowervals),
            'nbestperiods':nparr(nbestperiods),
            'nbestparams':nbestparamsout,
            'powervals':power_vals,
            'periods':period_grid}


def flag_harmonics(period_list: np.ndarray,
                   prot: float,
                   tolerance: float = 0.01) -> np.ndarray:
    """Flag periods that are harmonics of a known rotation period.

    A period is considered a harmonic if it is close to an integer multiple
    of `prot` within a fractional tolerance.

    Args:
        period_list (numpy.ndarray): Array of candidate periods.
        prot (float): The known rotation period.
        tolerance (float): Fractional tolerance for identifying harmonics.
            Defaults to 0.01.

    Returns:
        numpy.ndarray: Boolean array indicating which periods are harmonics.
    """
    ratios = period_list / prot
    nearest_int = np.round(ratios)
    fractional_diff = np.abs(ratios - nearest_int)
    return fractional_diff <= tolerance
