"""
Contents:

get_cdips_catalog: Updated version of Table 1 from Bouma+2019 CDIPS-I
get_cdips_pub_catalog: ""
get_cdips_pub_catalog_entry: Single row from above.

get_toi_catalog: TOI-plus catalog from MIT TEV.
get_exofop_toi_catalog: ExoFOP-TESS TOI table.
get_exofop_ctoi_catalog: ExoFOP-TESS CTOI table.
get_exofop_ctoi_catalog_entry: Single row from above.

ticid_to_toiid: Given a TICID, get a TOI identifer
get_tic_star_information: Given TICID, query TICv8 for arbitrary columns.
"""

import pandas as pd, numpy as np
import socket, os, json
from astrobase.services.mast import tic_objectsearch

def get_cdips_catalog(ver=0.4):

    dir_d = {
        'brik':'/home/luke/local/cdips/catalogs/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
    }

    cdips_stars_dir = dir_d[socket.gethostname()]

    cdips_stars_path = os.path.join(
        cdips_stars_dir, 'OC_MG_FINAL_GaiaRp_lt_16_v{}.csv'.format(ver)
    )

    df = pd.read_csv(cdips_stars_path, sep=';')

    return df


def get_cdips_pub_catalog(ver=0.4):

    dir_d = {
        'brik':'/home/luke/local/cdips/catalogs/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
    }

    cdips_stars_dir = dir_d[socket.gethostname()]

    cdips_stars_path = os.path.join(
        cdips_stars_dir, 'OC_MG_FINAL_v{}_publishable.csv'.format(ver)
    )

    df = pd.read_csv(cdips_stars_path, sep=';')

    return df


def get_cdips_pub_catalog_entry(source_id, ver=0.4):
    """
    Given a Gaia DR2 source_id, query the CDIPS target star catalog for the
    information it contains.

    Under the hood, the query uses grep, because this is much faster than
    reading the entire catalog. It returns a single-row dataframe if it
    succeeds; else, returns None.
    """

    dir_d = {
        'brik':'/home/luke/local/cdips/catalogs/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
    }

    cdips_stars_dir = dir_d[socket.gethostname()]

    cdips_stars_path = os.path.join(
        cdips_stars_dir, 'OC_MG_FINAL_v{}_publishable.csv'.format(ver)
    )

    colnames = r = os.popen(
        'head -n1 {}'.format(cdips_stars_path)
    ).read()
    colnames = colnames.rstrip('\n').split(';')

    rowentry = r = os.popen(
        'grep {} {}'.format(source_id, cdips_stars_path)
    ).read()
    rowentry = rowentry.rstrip('\n').split(';')

    if len(rowentry) >= 1:

        df = pd.DataFrame(
            {k: v for (k, v) in zip(colnames, rowentry)},
            index=[0]
        )

        if np.all(df.columns == 'source_id'):
            return None

        return df

    else:

        return None


def get_toi_catalog(ver='2020-01-08'):
    # a misnomer: really, the TOI-plus catalog, from MIT. (exported from
    # https://tev.mit.edu/)
    # note: this catalog is a bit janky. for example, they give transit epoch
    # to 5 decimal points in precision and sometimes fewer, which introduces a
    # minimum error of 300 seconds (5 minutes) after only one year.

    dir_d = {
        'brik':'/home/luke/Dropbox/proj/cdips/data/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/'
    }

    toi_stars_dir = dir_d[socket.gethostname()]

    toi_stars_path = os.path.join(
        toi_stars_dir, 'csv-file-toi-plus-catalog-{}.csv'.format(ver)
    )

    df = pd.read_csv(toi_stars_path, sep=',', comment='#')

    return df


def get_exofop_toi_catalog(ver='2020-01-08', returnpath=False):
    # https://exofop.ipac.caltech.edu/tess/view_toi.php, with pipe

    dir_d = {
        'brik':'/home/luke/Dropbox/proj/cdips/data/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/'
    }

    toi_stars_dir = dir_d[socket.gethostname()]

    toi_stars_path = os.path.join(
        toi_stars_dir, 'toi-exofop-{}.csv'.format(ver)
    )

    df = pd.read_csv(toi_stars_path, sep='|')

    if not returnpath:
        return df
    if returnpath:
        return toi_stars_path


def get_exofop_toi_catalog_entry(tic_id):

    df = get_exofop_toi_catalog()

    out_r = df[df['TIC ID'].astype(str) == str(tic_id)]

    if len(out_r) >= 1:
        return out_r

    else:
        return None


def get_exofop_ctoi_catalog(ver='2020-01-08'):

    dir_d = {
        'brik':'/home/luke/Dropbox/proj/cdips/data/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/'
    }

    ctoi_dir = dir_d[socket.gethostname()]

    ctoipath = os.path.join(ctoi_dir, 'ctoi-exofop-{}.csv'.format(ver))
    ctoidf = pd.read_csv(ctoipath, sep='|')

    return ctoidf


def get_exofop_ctoi_catalog_entry(tic_id):

    df = get_exofop_ctoi_catalog()

    out_r = df[df['TIC ID'].astype(str) == str(tic_id)]

    if len(out_r) >= 1:
        return out_r

    else:
        return None


def ticid_to_toiid(tic_id):

    assert isinstance(tic_id, str)

    if tic_id == '772008799':
        # exofop-tess has wrong ticid for TOI 1014.
        return '1014.01'

    toidf = get_exofop_toi_catalog()

    sel = toidf['TIC ID'].astype(str) == tic_id

    try:
        toi_id = str(toidf[sel]['TOI'].iloc[0])
        if len(toi_id) > 1:
            return str(toi_id)
        else:
            return None
    except:
        return None


def get_tic_star_information(
    ticid,
    desiredcols=['ID', 'GAIA', 'Bmag', 'Vmag', 'Jmag', 'Hmag', 'Kmag', 'Tmag',
                 'Teff', 'logg', 'rad', 'mass'],
    raise_error_on_multiple_match=True):
    """
    Given sting ticid, return single-rowed dataframe with TICv8 information of
    star. If no ticid match is found, None is returned.

    desiredcols : a subset of

        ['ID', 'version', 'HIP', 'TYC', 'UCAC', 'TWOMASS', 'SDSS', 'ALLWISE', 'GAIA',
        'APASS', 'KIC', 'objType', 'typeSrc', 'ra', 'dec', 'POSflag', 'pmRA', 'e_pmRA',
        'pmDEC', 'e_pmDEC', 'PMflag', 'plx', 'e_plx', 'PARflag', 'gallong', 'gallat',
        'eclong', 'eclat', 'Bmag', 'e_Bmag', 'Vmag', 'e_Vmag', 'umag', 'e_umag',
        'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 'Jmag',
        'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag', 'TWOMflag', 'prox', 'w1mag',
        'e_w1mag', 'w2mag', 'e_w2mag', 'w3mag', 'e_w3mag', 'w4mag', 'e_w4mag',
        'GAIAmag', 'e_GAIAmag', 'Tmag', 'e_Tmag', 'TESSflag', 'SPFlag', 'Teff',
        'e_Teff', 'logg', 'e_logg', 'MH', 'e_MH', 'rad', 'e_rad', 'mass', 'e_mass',
        'rho', 'e_rho', 'lumclass', 'lum', 'e_lum', 'd', 'e_d', 'ebv', 'e_ebv',
        'numcont', 'contratio', 'disposition', 'duplicate_id', 'priority', 'eneg_EBV',
        'epos_EBV', 'EBVflag', 'eneg_Mass', 'epos_Mass', 'eneg_Rad', 'epos_Rad',
        'eneg_rho', 'epos_rho', 'eneg_logg', 'epos_logg', 'eneg_lum', 'epos_lum',
        'eneg_dist', 'epos_dist', 'distflag', 'eneg_Teff', 'epos_Teff', 'TeffFlag',
        'gaiabp', 'e_gaiabp', 'gaiarp', 'e_gaiarp', 'gaiaqflag', 'starchareFlag',
        'VmagFlag', 'BmagFlag', 'splists', 'e_RA', 'e_Dec', 'RA_orig', 'Dec_orig',
        'e_RA_orig', 'e_Dec_orig', 'raddflag', 'wdflag', 'objID']
    """

    if not isinstance(ticid, str):
        raise ValueError('ticid needs to be string')

    ticres = tic_objectsearch(ticid)

    with open(ticres['cachefname'], 'r') as json_file:
        data = json.load(json_file)

    if len(data['data']) >= 1:

        if len(data['data']) > 1:
            errmsg = (
                'Got {} hits for TICID {}'.format(len(data['data']), ticid)
            )
            raise ValueError(errmsg)

        d = data['data'][0]

        outd = {}

        for col in desiredcols:
            outd[col] = d[col]

        df = pd.DataFrame(outd, index=[0])

        return df

    else:

        return None
