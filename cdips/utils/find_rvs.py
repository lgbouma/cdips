import os
from glob import glob
import numpy as np, pandas as pd
from numpy import array as nparr
from cdips.utils import today_YYYYMMDD

from astrobase import imageutils as iu

from astroquery.vizier import Vizier
from astroquery.eso import Eso

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

def get_rv_xmatch(ra, dec, G_mag=None, dr2_sourceid=None):
    """
    Given a single RA, dec, and optionally Gaia G mag, and DR2 sourceid, xmatch
    against spectroscopic catalogs to try and mine out an RV and error.

    Returns tuple of:
        (rv, rv_err, provenance)
        Presumably all RVs returned are heliocentric.

    The following spectroscopic surveys did not derive RVs:

        PASTEL
        TESS-HERMES (DR-1)

    The following surveys had non-uniform APIs to access their RVs:

        Gaia-ESO -- needs to be accessed thru ESO's query interface.

        GALAH -- needs to be accessed via
            https://github.com/svenbuder/GALAH_DR2 (or similar)

        APOGEE-2 (DR-15, now) -- also a fits file that gets queried.
            https://www.sdss.org/dr15/irspec/spectro_data/#catalogs.

    The following surveys are queried (fortunately, they're the biggest ones).
    In order of precedence:

        LAMOST DR-4: V/153. Note only the "A, F, G and K type stars catalog
        (4537436 rows)" includes RVs.

        RAVE DR-5: III/279

        Gaia DR2 RVs.
    """
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    print('begin vizier search for {}'.format(repr(coord)))

    # the following pattern enables return of all columns desired. e.g.,
    # "Source" for DR2 sourceid.
    v = Vizier(columns=['_RAJ2000', '_DEJ2000', 'HRV', 'e_HRV', 'RV', 'e_RV',
                        'Source'])
    result = v.query_region(coord, radius=10*u.arcsec,
                            catalog=["LAMOST", "RAVE", "GAIA"])

    # LAMOST, RAVE, GaiaDR2
    catkeys = ['V/153/stellar4', 'III/279/rave_dr5', 'I/345/gaia2']
    rv_given_cat = {
        'V/153/stellar4': ['HRV','e_HRV'],
        'III/279/rave_dr5': ['HRV','e_HRV'],
        'I/345/gaia2': ['RV','e_RV']
    }
    name_given_cat = {
        'V/153/stellar4': 'LAMOST-DR4',
        'III/279/rave_dr5': 'RAVE-DR5',
        'I/345/gaia2': 'GAIA-DR2'
    }

    for catkey in catkeys:
        if catkey in result.keys():
            res = result[catkey]
            if len(res) > 1:
                if catkey=='I/345/gaia2':
                    # just use dr2 source ID directly
                    res = res[res['Source'] == dr2_sourceid]
                else:
                    raise NotImplementedError(
                        'what do with multimatch? {}'.format(catkey)
                    )

            rv = float(res[rv_given_cat[catkey][0]])
            rv_err = float(res[rv_given_cat[catkey][1]])
            if not pd.isnull(rv):
                return rv, rv_err, name_given_cat[catkey]

    return np.nan, np.nan, np.nan




def wrangle_eso_for_rv_availability(ra, dec):
    """
    Checks via ESO query for available RVs on:
        ['HARPS', 'ESPRESSO', 'FORS2', 'UVES', 'XSHOOTER']

    Possible future expansion: actually get the RVs. (For now, just this is
    just used as a flag to let the user know the RVs might exist!)

    Returns tuple of:
        (nan, nan, provenance)
    """
    eso = Eso()
    eso.ROW_LIMIT = 9999

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    print('begin ESO search for {}'.format(repr(coord)))

    rastr = (
        str(coord.ra.to_string(u.hour)).
        replace('h',' ').replace('m',' ').replace('s',' ')
    )

    decstr = (
        str(coord.dec.to_string()).
        replace('d',' ').replace('m',' ').replace('s',' ')
    )

    # search within 10 arcsec of given position
    boxsize = '00 00 10'
    res = eso.query_main(
        column_filters={'ra':rastr, 'dec':decstr, 'box':boxsize}
    )

    if res is None:
        return np.nan, np.nan, np.nan

    # limit search to the following instruments, in order of preference
    instruments = ['HARPS', 'ESPRESSO', 'FORS2', 'UVES', 'XSHOOTER']
    sel = np.zeros((len(res))).astype(bool)
    for instrument in instruments:
        sel |= (nparr(res['Instrument']) == instrument)
    res = res[sel]

    # limit returned cateogires
    badcategories = ['CALIB']
    sel = np.zeros((len(res))).astype(bool)
    for badcategory in badcategories:
        sel |= (nparr(res['Category']) != badcategory)
    res = res[sel]

    if len(res) >= 1:

        # XSHOOTER doesn't seem to give archival RVs. would need to derive
        # from spectra yourself
        if np.all(nparr(res['Instrument']) == 'XSHOOTER'):
            return np.nan, np.nan, 'XSHOOTER'

        # Embargo lasts a year on all ESO observations.
        nt = Time.now()
        embargo_end = nt.mjd - 365
        if np.all(nparr(res['MJD-OBS']) > embargo_end):
            return np.nan, np.nan, np.unique(res['Instrument'])[0]

        # HARPS gives archival RVs. downloading them can be done... but for
        # s6+s7, only a few objects are viable.
        if np.all(nparr(res['Instrument']) == 'HARPS'):
            print('WARNING: SKIPPING AUTOMATION OF HARPS ARCHIVAL RV GETTING')
            return np.nan, np.nan, 'HARPS'

    else:
        return np.nan, np.nan, np.nan




