import os
from astroquery.vizier import Vizier
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

def get_k13_index():
    #
    # the ~3784 row table
    #
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/A+A/558/A53')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    k13_index = catalogs[1].to_pandas()
    for c in k13_index.columns:
        if c != 'N':
            k13_index[c] = k13_index[c].str.decode('utf-8')

    return k13_index


def get_k13_param_table():
    #
    # the ~3000 row table with determined parameters
    #
    cols = ['map', 'cmd', 'stars', 'Name', 'MWSC', 'Type', 'RAJ2000',
            'DEJ2000', 'r0', 'r1', 'r2', 'pmRA', 'pmDE', 'RV', 'e_RV', 'o_RV',
            'd', 'E_B-V_', 'logt', 'N1sr2', 'rc', 'rt', 'k', 'SType',
            '__Fe_H_', 'Simbad']

    v = Vizier(columns=cols)

    v.ROW_LIMIT = -1
    catalog_list = v.find_catalogs('J/A+A/558/A53')
    catalogs = v.get_catalogs(catalog_list.keys())
    k13 = catalogs[0].to_pandas()

    k13['Name'] = k13['Name'].str.decode('utf-8')
    return k13


def get_k13_member_stars(mwscid, mwscname, outpath, overwrite=1, p_0=61):
    """
    Query Vizier's Kharchenko+2013 database directly to download stars from
    that work.

    Args:
        mwscid (str), e.g., 0005
        mwscname (str), associated string name
        outpath (str) the table is written here.

    Note: if cross-matching against DR2, the k13_tmass_oids are not
    convincingly matched against 2MASS. (The tmass.nearest_neighbour table
    seems to just not work with that key, for some reason).
    """

    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite. return'.format(outpath))
        return
    if os.path.exists(outpath) and overwrite:
        os.remove(outpath)

    # if buggy, look at the links at e.g.,
    # http://cdsarc.u-strasbg.fr/viz-bin/getCatFile_Redirect/?-plus=-%2b&J/A%2bA/558/A53/stars/2m_0763_Platais_5.dat
    mwsc_urlname = mwscname.replace('_','%5F')
    url = (
        "http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/fits?-plus=-+&J/A%2BA/558/A53/stars/2m%5F{:s}%5F{:s}.dat".
        format(mwscid, mwsc_urlname)
    )

    t = Table.read(url)

    c = SkyCoord(t['RAhour'], t['DEdeg'], unit=(u.hourangle, u.degree))

    t['RAdeg'] = c.ra.value

    sel = (
        (t['Ps'] == 1)
        &
        (t['Pkin'] > p_0)
        &
        (t['PJKs'] > p_0)
        &
        (t['PJH'] > p_0)
    )

    print('Got {} stars in {} nbhd from K+13 query'.format(len(t), mwscname))
    t = t[sel]
    print('... and {} member stars'.format(len(t)))

    df = t.to_pandas()

    return df



def get_soubiran_19_rv_table():

    cols = ['ID', 'ID2', 'RA_ICRS', 'DE_ICRS', 'dmode', 'Nmemb', 'Nsele', 'RV',
            'e_RV', 's_RV', 'X', 'e_X', 'Y', 'e_Y', 'Z', 'e_Z', 'U', 'e_U',
            'V', 'e_V', 'W', 'e_W', 'Vr', 'e_Vr', 'Vphi', 'e_Vphi', 'Vz',
            'e_Vz', 'Simbad']

    v = Vizier(columns=cols)

    v.ROW_LIMIT = -1
    catalog_list = v.find_catalogs('J/A+A/619/A155')
    catalogs = v.get_catalogs(catalog_list.keys())
    df = catalogs[0].to_pandas()

    df['ID'] = df['ID'].str.decode('utf-8')

    return df
