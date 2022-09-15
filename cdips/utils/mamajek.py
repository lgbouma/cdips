"""
1d interpolation functions based off the Pecaut & Mamajek (2013) tables:
http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt

Contents:
    get_interp_mass_from_rstar
    get_interp_rstar_from_teff
    get_interp_BmV_from_Teff
    get_interp_BpmRp_from_Teff
    get_interp_BmV_from_BpmRp
    get_interp_BpmRp_from_BmV
    get_interp_SpType_from_teff
    get_SpType_BpmRp_correspondence
    get_SpType_GmRp_correspondence
    given_VmKs_get_Teff
"""
import os
import numpy as np, pandas as pd
from numpy import array as nparr
from scipy.interpolate import interp1d
from scipy import optimize

import cdips as cd
datadir = os.path.join(os.path.dirname(cd.__path__[0]), 'data')

def load_basetable():

    mamajekpath = os.path.join(datadir, 'LITERATURE_DATA',
                               'EEM_dwarf_UBVIJHK_colors_Teff_20220416.txt')
    mamajek_df = pd.read_csv(
        mamajekpath, comment='#', delim_whitespace=True
    )
    mamajek_df = mamajek_df[mamajek_df.Teff < 41000]
    mamajek_df = mamajek_df.reset_index(drop=True)

    return mamajek_df

def get_interp_mass_from_rstar(rstar):
    """
    Given a stellar radius, use the Pecaut & Mamajek (2013) table from
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt to
    interpolate the stellar mass. Assumes the star is a dwarf, which may not be
    true. This should obviously not be trusted to better than 50%.
    """

    mamajek_df = load_basetable()

    sel = (
        (mamajek_df['Msun'] != '...') &
        (mamajek_df['R_Rsun'] != '...') &
        (mamajek_df['Msun'] != '....')
    )
    mamajek_df = mamajek_df[sel] # finite mass and radius.

    mamarstar, mamamstar = (
        nparr(mamajek_df['R_Rsun'])[::-1].astype(float),
        nparr(mamajek_df['Msun'])[::-1].astype(float)
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

    mamajek_df = load_basetable()

    sel = (
        (mamajek_df['Msun'] != '...') &
        (mamajek_df['Msun'] != '....') &
        (mamajek_df['R_Rsun'] != '...') &
        (mamajek_df['Teff'] != '...')
    )
    mamajek_df = mamajek_df[sel] # finite mass, radius, teff.

    mamarstar, mamamstar, mamateff = (
        nparr(mamajek_df['R_Rsun'])[::-1].astype(float),
        nparr(mamajek_df['Msun'])[::-1].astype(float),
        nparr(mamajek_df['Teff'])[::-1].astype(float)
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


def get_interp_SpType_from_teff(teff, verbose=True):
    """
    Given an effective temperature, use the Pecaut & Mamajek (2013) table from
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt to
    interpolate the spectral type. Assumes the star is a dwarf, which may not
    be true. This should obviously not be trusted to better than 1 or 2
    spectral subtypes. Worse if it's pre-main-sequence.
    """

    mamajek_df = load_basetable()

    sel = (
        (mamajek_df['Msun'] != '...') &
        (mamajek_df['Msun'] != '....') &
        (mamajek_df['R_Rsun'] != '...') &
        (mamajek_df['Teff'] != '...')
    )
    mamajek_df = mamajek_df[sel] # finite mass, radius, teff.

    mamateff, mamaspt = (
        nparr(mamajek_df['Teff'])[::-1].astype(float),
        nparr(mamajek_df['SpT'])[::-1]
    )

    closest_spt = mamaspt[np.argmin(np.abs(mamateff - teff))]
    closest_teff = mamateff[np.argmin(np.abs(mamateff - teff))]

    # hotter end
    closest_spt_m1 = mamaspt[np.argmin(np.abs(mamateff - teff))+1]
    closest_teff_m1 = mamateff[np.argmin(np.abs(mamateff - teff))+1]

    # cooler end
    closest_spt_p1 = mamaspt[np.argmin(np.abs(mamateff - teff))-1]
    closest_teff_p1 = mamateff[np.argmin(np.abs(mamateff - teff))-1]

    if verbose:
        print(f'Target Teff: {teff}')
        print(f'Closest Teff: {closest_teff} (SpT = {closest_spt})')
        print(f'Next-highest Teff: {closest_teff_m1} (SpT = {closest_spt_m1})')
        print(f'Next-lowest Teff: {closest_teff_p1} (SpT = {closest_spt_p1})')

    if teff <= closest_teff:

        dT_interval = closest_teff - closest_teff_p1
        dT = closest_teff - teff
        endstr = str( np.round(dT/dT_interval, 1) )[1:]
        outstr = closest_spt[:-1] + endstr + "V"

    else:

        dT_interval = closest_teff_m1 - closest_teff
        dT = teff - closest_teff
        endstr = str( np.round(1-dT/dT_interval, 1) )[1:]
        outstr = closest_spt_m1[:-1] + endstr + "V"

    # clean-up for M-dwarf case, e.g., "M1.5.3V" becomes "M.165V" ==
    # "M.16V" after rounding.
    if np.sum(np.array(list(outstr)) == '.') == 2:
        to_add = float(outstr.split('.')[-1][0]) / 2
        outstr = (
            outstr[:3]
            +
            str(int(
                np.round(float(outstr.split('.')[1]) + to_add, 0))
            )
            +
            "V"
        )

    return outstr


def get_interp_BmV_from_Teff(teff):
    """
    Given an effective temperature (or an array of them), get interpolated B-V
    color.
    """

    mamajek_df = load_basetable()
    sel = (mamajek_df['B-V'] != '...')
    mamajek_df = mamajek_df[sel] # finite, monotonic BmV

    mamarstar, mamamstar, mamateff, mamaBmV = (
        nparr(mamajek_df['R_Rsun'])[::-1],
        nparr(mamajek_df['Msun'])[::-1],
        nparr(mamajek_df['Teff'])[::-1],
        nparr(mamajek_df['B-V'])[::-1].astype(float)
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

    mamajek_df = load_basetable()
    mamajek_df = mamajek_df[22:-6] # finite, monotonic BpmRp
    mamajek_df = mamajek_df[(mamajek_df['Bp-Rp'] != '...')] # finite!

    mamarstar, mamamstar, mamateff, mamaBpmRp = (
        nparr(mamajek_df['R_Rsun'])[::-1],
        nparr(mamajek_df['Msun'])[::-1],
        nparr(mamajek_df['Teff'])[::-1],
        nparr(mamajek_df['Bp-Rp'])[::-1].astype(float)
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

    mamajek_df = load_basetable()
    mamajek_df = mamajek_df[22:-6] # finite, monotonic BpmRp

    sel = (
        (mamajek_df['B-V'] != '...')
        &
        (mamajek_df['Bp-Rp'] != '...')
    )

    mamaBmV, mamaBpmRp = (
        nparr(mamajek_df['B-V'][sel])[::-1].astype(float),
        nparr(mamajek_df['Bp-Rp'][sel])[::-1].astype(float)
    )

    # include "isbad" catch because EVEN ONCE SORTED, you can have multivalued
    # BmVs. so remove anything where diff not strictly greater than...
    isbad = np.insert(np.diff(mamaBpmRp) == 0, False, 0)

    fn_BpmRp_to_BmV = interp1d(mamaBpmRp[~isbad], mamaBmV[~isbad],
                               kind='quadratic', bounds_error=False,
                               fill_value='extrapolate')

    return fn_BpmRp_to_BmV(BpmRp)


def get_interp_BpmRp_from_BmV(BmV):
    """
    Given B-V, get BP-RP
    """

    mamajek_df = load_basetable()
    mamajek_df = mamajek_df[22:-6] # finite, monotonic BpmRp

    sel = (
        (mamajek_df['B-V'] != '...')
        &
        (mamajek_df['Bp-Rp'] != '...')
    )

    mamaBmV, mamaBpmRp = (
        nparr(mamajek_df['B-V'][sel])[::-1].astype(float),
        nparr(mamajek_df['Bp-Rp'][sel])[::-1].astype(float)
    )

    # include "isbad" catch because EVEN ONCE SORTED, you can have multivalued
    # BmVs. so remove anything where diff not strictly greater than...
    isbad = np.insert(np.diff(mamaBpmRp) == 0, False, 0)

    fn_BmV_to_BpmRp = interp1d(mamaBmV[~isbad], mamaBpmRp[~isbad],
                               kind='quadratic', bounds_error=False,
                               fill_value='extrapolate')

    return fn_BmV_to_BpmRp(BmV)


def get_interp_Rstar_from_BpmRp(BpmRp):
    """
    Given a Bp-Rp color, get expected Rstar.
    """

    mamajek_df = load_basetable()
    mamajek_df = mamajek_df[22:-6] # finite, monotonic BpmRp

    sel = (
        (mamajek_df['Bp-Rp'] != '...') &
        (mamajek_df['Msun'] != '...') &
        (mamajek_df['Msun'] != '....') &
        (mamajek_df['R_Rsun'] != '...') &
        (mamajek_df['Teff'] != '...')
    )

    mamaRstar, mamaBpmRp = (
        nparr(mamajek_df['R_Rsun'][sel])[::-1].astype(float),
        nparr(mamajek_df['Bp-Rp'][sel])[::-1].astype(float)
    )

    # include "isbad" catch because EVEN ONCE SORTED, you can have multivalued
    # yvals. so remove anything where diff not strictly greater than...
    isbad = (mamaBpmRp > 4.2)

    fn_BpmRp_to_Rstar = interp1d(mamaBpmRp[~isbad], mamaRstar[~isbad],
                                 kind='quadratic', bounds_error=False,
                                 fill_value='extrapolate')

    return fn_BpmRp_to_Rstar(BpmRp)


def get_SpType_BpmRp_correspondence(
    sptypes=['A0V','F0V','G0V','K2V','K5V','M0V','M3V','M5V'],
    return_absG=False
    ):

    mamajek_df = load_basetable()

    sel = (
        (mamajek_df['Bp-Rp'] != '...')
    )

    sdf = mamajek_df[sel]

    BpmRps = []
    AbsGs = []
    Msuns = []
    for sptype in sptypes:
        BpmRps.append(float(sdf.loc[sdf.SpT==sptype]['Bp-Rp']))
        AbsGs.append(float(sdf.loc[sdf.SpT==sptype]['M_G']))
        Msuns.append(float(sdf.loc[sdf.SpT==sptype]['Msun']))

    sptypes = [s.replace('V','') for s in sptypes]

    if not return_absG:
        return np.array(sptypes), np.array(BpmRps)
    else:
        return np.array(sptypes), np.array(AbsGs), np.array(Msuns)


def get_SpType_GmRp_correspondence(
    sptypes=['A0V','F0V','G0V','K2V','K5V','M0V','M3V','M5V'],
    return_absG=False
    ):

    mamajek_df = load_basetable()

    sel = (
        (mamajek_df['G-Rp'] != '...')
    )

    sdf = mamajek_df[sel]

    GmRps = []
    AbsGs = []
    Msuns = []
    for sptype in sptypes:
        GmRps.append(float(sdf.loc[sdf.SpT==sptype]['G-Rp']))
        AbsGs.append(float(sdf.loc[sdf.SpT==sptype]['M_G']))
        Msuns.append(float(sdf.loc[sdf.SpT==sptype]['Msun']))

    sptypes = [s.replace('V','') for s in sptypes]

    if not return_absG:
        return np.array(sptypes), np.array(GmRps)
    else:
        return np.array(sptypes), np.array(AbsGs), np.array(Msuns)


def get_SpType_Teff_correspondence(
    sptypes=['A0V','F0V','G0V','K2V','K5V','M0V','M3V','M5V'],
    ):

    mamajek_df = load_basetable()

    sel = (
        (mamajek_df['Bp-Rp'] != '...')
    )

    sdf = mamajek_df[sel]

    Teffs = []
    for sptype in sptypes:
        Teffs.append(float(sdf.loc[sdf.SpT==sptype]['Teff']))

    sptypes = [s.replace('V','') for s in sptypes]

    return np.array(sptypes), np.array(Teffs)


def given_VmKs_get_Teff(VmKs):
    """
    Interpolate effective temperatures from the Mamajek table and (V-Ks)0.
    (less reddening dependence than B-V).
    """
    mamajekpath = os.path.join(DATADIR, "literature",
                               "EEM_dwarf_UBVIJHK_colors_Teff_20220416.txt")
    mamajek_df = pd.read_csv(
        mamajekpath, comment='#', delim_whitespace=True
    )
    sel = (
        (mamajek_df['V-Ks'] != ".....")
        &
        (mamajek_df['V-Ks'] != "....")
        &
        (mamajek_df['V-Ks'] != "...")
        &
        (mamajek_df["Teff"] > 3300)
        &
        (mamajek_df["Teff"] < 7400)
    )
    mamajek_df = mamajek_df[sel]
    mamajek_df = mamajek_df.reset_index(drop=True)

    poly_model = np.polyfit(
        mamajek_df['V-Ks'].astype(float), mamajek_df['Teff'].astype(float), 7
    )
    testmodel = np.polyval(poly_model, mamajek_df['V-Ks'].astype(float))

    Teff = np.polyval(poly_model, VmKs)

    return Teff



