from astrobase import lcmath
import numpy as np, pandas as pd
import os, textwrap
from glob import glob
from datetime import datetime

def mask_orbit_start_and_end_given_lcpaths(lcpaths):
    #TODO
    raise NotImplementedError

def mask_orbit_start_and_end(time, flux, orbitgap=1, expected_norbits=2,
                             orbitpadding=6/(24), raise_expectation_error=True):
    """
    Ignore the times near the edges of orbits.

    args:
        time, flux
    returns:
        time, flux: with `orbitpadding` days trimmed out
    """
    norbits, groups = lcmath.find_lc_timegroups(time, mingap=orbitgap)

    if norbits != expected_norbits:
        errmsg = 'got {} orbits, expected {}. groups are {}'.format(
            norbits, expected_norbits, repr(groups))

        if raise_expectation_error:
            raise AssertionError(errmsg)
        elif norbits > 0 and not raise_expectation_error:
            print('WRN! {}'.format(errmsg))
        else:
            # no matter what, if you don't get any data, raise the assertion
            # error.
            raise AssertionError(errmsg)

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

    return return_time, return_flux
