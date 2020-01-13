import os
import numpy as np, pandas as pd
from numpy import array as nparr
from scipy.interpolate import interp1d
from scipy import optimize

import cdips as cd
datadir = os.path.join(os.path.dirname(cd.__path__[0]), 'data')

def get_interp_mass_from_rstar(rstar):
    """
    Given a stellar radius, use the Pecaut & Mamajek (2013) table from
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt to
    interpolate the stellar mass. Assumes the star is a dwarf, which may not be
    true. This should obviously not be trusted to better than 50%.
    """

    mamadf = pd.read_csv(
        os.path.join(datadir, 'Mamajek_Rstar_Mstar_Teff_SpT.txt'),
        delim_whitespace=True
    )
    mamarstar, mamamstar = (
        nparr(mamadf['Rsun'])[::-1], nparr(mamadf['Msun'])[::-1]
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

    mamadf = pd.read_csv(
        os.path.join(datadir, 'Mamajek_Rstar_Mstar_Teff_SpT.txt'),
        delim_whitespace=True
    )
    mamarstar, mamamstar, mamateff = (
        nparr(mamadf['Rsun'])[::-1],
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
