"""
Helper utilities for common coordinate transformations. Some duplications with
astrobase/timeutils, and astrobase/coordutils, but these implementations mostly
outsource it to astropy.

Contents:

    precess_coordinates
"""

import numpy as np, pandas as pd
from numpy import array as nparr

import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time

def precess_gaia_coordinates(gaia_df, new_epoch=2000.0):
    '''
    Precesses target coordinates from Gaia (i.e., using a parallax) from the
    original epoch to a new epoch. Radial velocities are assumed to be zero
    (because they're measured poorly). "The new position of the source is
    determined assuming that it moves in a straight line with constant velocity
    in an inertial frame."

    Adapted from the astropy docs:
        https://docs.astropy.org/en/stable/coordinates/apply_space_motion.html

    [Note that the *equinox* is a currently obsolete concept, formerly tied to
    equatorial coordinate systems centered on the Earth. The ICRS reference
    system, which Gaia uses, has an origin at the solar system barycenter and
    kinematic axes that are fixed and non-rotating. The nutation of the Earth
    does not affect these coordinates. See
    https://www.cosmos.esa.int/web/gaia/faqs#ICRSICRF]

    Parameters
    ----------

    gaia_df: pandas DataFrame of Gaia data.
        It's assumed that you're starting from Gaia data (any data release).
        Keys must include 'ra', 'dec', 'parallax', 'pmra', 'pmdec',
        'ref_epoch'. This can also be a dict.

    new_epoch: float
        Target epoch in Julian year format to precess from origin epoch. This
        is a float, like: 2000.0, 2021.0, etc.

    Returns
    -------

    c, c_new_epoch: astropy Coordinate
        Original and precessed astropy coordinate objects corresponding to new_epoch.
    '''

    c = SkyCoord(ra=nparr(gaia_df['ra']) * u.deg,
                 dec=nparr(gaia_df['dec']) * u.deg,
                 distance=Distance(parallax=nparr(gaia_df['parallax']) * u.mas),
                 pm_ra_cosdec=nparr(gaia_df['pmra']) * u.mas/u.yr,
                 pm_dec=nparr(gaia_df['pmdec']) * u.mas/u.yr,
                 obstime=Time(nparr(gaia_df['ref_epoch']), format='jyear'))

    new_epoch = Time(new_epoch, format='jyear')

    c_new_epoch = c.apply_space_motion(new_epoch)

    return c, c_new_epoch
