"""
1d interpolation functions based off the Pecaut & Mamajek (2013) tables:
http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt

Contents:
    get_interp_mass_from_rstar
    get_interp_rstar_from_teff
    get_interp_BmV_from_Teff
    get_interp_BpmRp_from_Teff
    get_interp_BmV_from_BpmRp
"""
import os
import numpy as np, pandas as pd
from numpy import array as nparr
from scipy.interpolate import interp1d
from scipy import optimize

import cdips as cd
datadir = os.path.join(os.path.dirname(cd.__path__[0]), 'data')

def load_basetable():

    mamajekpath = os.path.join(datadir, 'EEM_dwarf_UBVIJHK_colors_Teff_20210302.txt')
    mamadf = pd.read_csv(
        mamajekpath, comment='#', delim_whitespace=True
    )
    mamadf = mamadf[mamadf.Teff < 41000]
    mamadf = mamadf.reset_index(drop=True)

    return mamadf

def get_interp_mass_from_rstar(rstar):
    """
    Given a stellar radius, use the Pecaut & Mamajek (2013) table from
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt to
    interpolate the stellar mass. Assumes the star is a dwarf, which may not be
    true. This should obviously not be trusted to better than 50%.
    """

    mamadf = load_basetable()
    mamarstar, mamamstar = (
        nparr(mamadf['R_Rsun'])[::-1], nparr(mamadf['Msun'])[::-1]
    )

    #
    # mstar and rstar are not strictly monotonic, at least according to this
    # table. sort the tuples by mstar, since that will be the x-value we will
    # interpolate over. then split back once sorted.
    #
    points = zip(mamamstar, mamarstar)
    points = sorted(points, key=lambda point: point[0])
    mamamstar, mamarstar = zip(*points)
    mamamstar, mamarstar = nparr(mamamstar), nparr(mamarstar)

    # include "isbad" catch because EVEN ONCE SORTED, you can have multivalued
    # rstars. so remove anything where diff not strictly greater than...
    isbad = np.insert(np.diff(mamamstar) == 0, False, 0)

    fn_mass_to_rstar = interp1d(mamamstar[~isbad],
                                mamarstar[~isbad],
                                kind='quadratic',
                                bounds_error=False,
                                fill_value='extrapolate')
    radiusval = rstar
    fn = lambda mass: fn_mass_to_rstar(mass) - radiusval
    mass_guess = (
        mamamstar[np.argmin(np.abs(mamarstar - radiusval))]
    )
    try:
        mass_val = optimize.newton(fn, mass_guess)
    except RuntimeError:
        mass_val = mass_guess
        print('WRN! optimization for mass failed; returning initial guess.')

    mstar = mass_val
    return mstar


def get_interp_rstar_from_teff(teff):
    """
    Given an effective temperature, use the Pecaut & Mamajek (2013) table from
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt to
    interpolate the stellar radius. Assumes the star is a dwarf, which may not
    be true. This should obviously not be trusted to better than 50%.
    """

    mamadf = load_basetable()
    mamarstar, mamamstar, mamateff = (
        nparr(mamadf['R_Rsun'])[::-1],
        nparr(mamadf['Msun'])[::-1],
        nparr(mamadf['Teff'])[::-1]
    )

    #
    # teff and rstar are not strictly monotonic, at least according to this
    # table. sort the tuples by rstar, since that will be the x-value we will
    # interpolate over. then split back once sorted.
    #
    points = zip(mamarstar, mamateff)
    points = sorted(points, key=lambda point: point[0])
    mamarstar, mamateff = zip(*points)
    mamarstar, mamateff = nparr(mamarstar), nparr(mamateff)

    # include "isbad" catch because EVEN ONCE SORTED, you can have multivalued
    # rstars. so remove anything where diff not strictly greater than...
    isbad = np.insert(np.diff(mamarstar) == 0, False, 0)

    fn_rstar_to_teff = interp1d(mamarstar[~isbad],
                                mamateff[~isbad],
                                kind='quadratic',
                                bounds_error=False,
                                fill_value='extrapolate')
    teffval = teff
    fn = lambda rstar: fn_rstar_to_teff(rstar) - teffval
    rstar_guess = (
        mamarstar[np.argmin(np.abs(mamateff - teffval))]
    )
    try:
        rstar_val = optimize.newton(fn, rstar_guess)
    except RuntimeError:
        rstar_val = rstar_guess
        print('WRN! optimization for rstar failed; returning initial guess.')

    rstar = rstar_val
    return rstar


def get_interp_BmV_from_Teff(teff):
    """
    Given an effective temperature (or an array of them), get interpolated B-V
    color.
    """

    mamadf = load_basetable()
    mamadf = mamadf[3:-6] # finite, monotonic BmV

    mamarstar, mamamstar, mamateff, mamaBmV = (
        nparr(mamadf['R_Rsun'])[::-1],
        nparr(mamadf['Msun'])[::-1],
        nparr(mamadf['Teff'])[::-1],
        nparr(mamadf['B-V'])[::-1].astype(float)
    )

    # include "isbad" catch because EVEN ONCE SORTED, you can have multivalued
    # BmVs. so remove anything where diff not strictly greater than...
    isbad = np.insert(np.diff(mamaBmV) == 0, False, 0)

    fn_teff_to_BmV = interp1d(mamateff[~isbad], mamaBmV[~isbad],
                              kind='quadratic', bounds_error=False,
                              fill_value='extrapolate')

    return fn_teff_to_BmV(teff)


def get_interp_BpmRp_from_Teff(teff):
    """
    Given an effective temperature (or an array of them), get interpolated
    Bp-Rp color.
    """

    mamadf = load_basetable()
    mamadf = mamadf[22:-6] # finite, monotonic BpmRp

    mamarstar, mamamstar, mamateff, mamaBpmRp = (
        nparr(mamadf['R_Rsun'])[::-1],
        nparr(mamadf['Msun'])[::-1],
        nparr(mamadf['Teff'])[::-1],
        nparr(mamadf['Bp-Rp'])[::-1].astype(float)
    )

    # include "isbad" catch because EVEN ONCE SORTED, you can have multivalued
    # BmVs. so remove anything where diff not strictly greater than...
    isbad = np.insert(np.diff(mamaBpmRp) == 0, False, 0)

    fn_teff_to_BpmRp = interp1d(mamateff[~isbad], mamaBpmRp[~isbad],
                                kind='quadratic', bounds_error=False,
                                fill_value='extrapolate')

    return fn_teff_to_BpmRp(teff)


def get_interp_BmV_from_BpmRp(BpmRp):
    """
    Given a Bp-Rp color, get expected B-V color.
    """

    mamadf = load_basetable()
    mamadf = mamadf[22:-6] # finite, monotonic BpmRp

    sel = (
        (mamadf['B-V'] != '...')
        &
        (mamadf['Bp-Rp'] != '...')
    )

    mamaBmV, mamaBpmRp = (
        nparr(mamadf['B-V'][sel])[::-1].astype(float),
        nparr(mamadf['Bp-Rp'][sel])[::-1].astype(float)
    )

    # include "isbad" catch because EVEN ONCE SORTED, you can have multivalued
    # BmVs. so remove anything where diff not strictly greater than...
    isbad = np.insert(np.diff(mamaBpmRp) == 0, False, 0)

    fn_BpmRp_to_BmV = interp1d(mamaBpmRp[~isbad], mamaBmV[~isbad],
                               kind='quadratic', bounds_error=False,
                               fill_value='extrapolate')

    return fn_BpmRp_to_BmV(BpmRp)
