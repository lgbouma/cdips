"""
Contents:

get_cdips_catalog: Updated version of Table 1 from Bouma+2019 CDIPS-I
get_cdips_pub_catalog: ""
get_cdips_pub_catalog_entry: Single row from above.

get_toi_catalog: TOI-plus catalog from MIT TEV.
get_exofop_toi_catalog: ExoFOP-TESS TOI table.
get_exofop_ctoi_catalog: ExoFOP-TESS CTOI table.
get_exofop_ctoi_catalog_entry: Single row from above.

get_nasa_exoplanet_archive_pscomppars: Pull the latest NASA exoplanet archive
    composite planet parameter table.

ticid_to_toiid: Given a TICID, get a TOI identifer
get_tic_star_information: Given TICID, query TICv8 for arbitrary columns.
"""

import pandas as pd, numpy as np
import socket, os, json, csv
from astrobase.services.mast import tic_objectsearch
from cdips.utils import today_YYYYMMDD
TODAYSTR = '-'.join([today_YYYYMMDD()[:4],
                     today_YYYYMMDD()[4:6],
                     today_YYYYMMDD()[6:]])

from cdips.paths import LOCALDIR

def get_nasa_exoplanet_archive_pscomppars(ver=TODAYSTR, N_max=int(2e4)):
    """
    If newestpossible is True, will download the latest NEA pscomppars table,
    from today. Otherwise, it'll take the most recent from those already
    downloaded.
    """

    from astroquery.utils.tap.core import TapPlus

    savedir = os.path.join(LOCALDIR, 'catalogs')

    if not os.path.exists(savedir):
        try:
            os.mkdir(savedir)
        except:
            raise NotImplementedError(f'Tried to make {savedir} and failed.')

    nea_path = os.path.join(
        savedir, f'nasaexoplanetarchive-pscomppars-{ver}.csv'
    )

    if not os.path.exists(nea_path):

        tap = TapPlus(url="https://exoplanetarchive.ipac.caltech.edu/TAP/")
        query = (
            f'select top {N_max} '+
            'pl_name, hostname, pl_letter, gaia_id, tic_id, ra, dec, '+
            'discoverymethod, disc_year, disc_facility, pl_orbper, pl_orbsmax, pl_rade, '+
            'pl_radeerr1, pl_radeerr2, '+
            'pl_radjerr1, pl_radjerr2, '+
            'pl_radj, pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_bmassj, '+
            'pl_bmassjerr1, pl_bmassjerr2, pl_orbeccen, pl_imppar, '+
            'pl_insol, pl_insolerr1, pl_insolerr2, '+
            'pl_eqt, pl_eqt, pl_eqt, '+
            'st_teff, st_rad, st_mass, st_met, st_logg, st_rotp, sy_dist, '+
            'sy_disterr1, sy_disterr2, sy_plx, sy_plxerr1, sy_plxerr2, '+
            'sy_vmag, sy_tmag, '+
            'tran_flag, rv_flag, ima_flag, '+
            'st_age, st_ageerr1, st_ageerr2 from pscomppars'
        )
        print(query)
        j = tap.launch_job(query=query)
        r = j.get_results()

        assert len(r) != N_max

        df = r.to_pandas()
        df.to_csv(nea_path, index=False)

    else:
        df = pd.read_csv(nea_path, sep=',')

    return df



def get_cdips_catalog(ver=0.6):

    if not isinstance(ver, float):
        ver = float(ver)


    dir_d = {
        'brik':'/home/luke/local/cdips/catalogs/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess3':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'PU-C02C76B8MD6T':'/Users/luke/local/cdips/catalogs/',
        'marduk.local':'/Users/luke/local/cdips/catalogs/'
    }

    cdips_stars_dir = dir_d[socket.gethostname()]

    if ver <= 0.4:
        cdips_stars_path = os.path.join(
            cdips_stars_dir, 'OC_MG_FINAL_GaiaRp_lt_16_v{}.csv'.format(ver)
        )
        df = pd.read_csv(cdips_stars_path, sep=';')
    else:
        cdips_stars_path = os.path.join(
            cdips_stars_dir,
            f'cdips_targets_v{ver}_gaiasources_Rplt16_orclose.csv'
        )
        df = pd.read_csv(cdips_stars_path, sep=',')

    return df


def get_cdips_pub_catalog(ver=0.6):

    if not isinstance(ver, float):
        ver = float(ver)

    dir_d = {
        'brik':'/home/luke/local/cdips/catalogs/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phtess3':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'PU-C02C76B8MD6T':'/Users/luke/local/cdips/catalogs/',
        'marduk.local':'/Users/luke/local/cdips/catalogs/',
        'WWS-C02C76B8MD6T':'/Users/luke/local/cdips/catalogs/'
    }

    cdips_stars_dir = dir_d[socket.gethostname()]

    if ver <= 0.4:
        cdips_stars_path = os.path.join(
            cdips_stars_dir, 'OC_MG_FINAL_v{}_publishable.csv'.format(ver)
        )
        df = pd.read_csv(cdips_stars_path, sep=';')
    else:
        cdips_stars_path = os.path.join(
            cdips_stars_dir,
            f'cdips_targets_v{ver}_gaiasources_Rplt16_orclose.csv'
        )
        df = pd.read_csv(cdips_stars_path, sep=',')


    return df


def get_cdips_pub_catalog_entry(source_id, ver=0.6):
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
        'phtess3':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/',
        'PU-C02C76B8MD6T':'/Users/luke/local/cdips/catalogs/',
        'marduk.local':'/Users/luke/local/cdips/catalogs/',
        'WWS-C02C76B8MD6T':'/Users/luke/local/cdips/catalogs/'
    }

    cdips_stars_dir = dir_d[socket.gethostname()]

    if ver <= 0.4:
        cdips_stars_path = os.path.join(
            cdips_stars_dir, 'OC_MG_FINAL_v{}_publishable.csv'.format(ver)
        )
        delim = ';'
    else:
        cdips_stars_path = os.path.join(
            cdips_stars_dir,
            f'cdips_targets_v{ver}_gaiasources_Rplt16_orclose.csv'
        )
        delim = ','

    colnames = r = os.popen(
        'head -n1 {}'.format(cdips_stars_path)
    ).read()
    colnames = colnames.rstrip('\n').split(delim)

    rowentry = r = os.popen(
        'grep {} {}'.format(source_id, cdips_stars_path)
    ).read()
    if delim == ',':
        # comma-separated ages, references, etc. are protected with the
        # quotechar.
        rowentry = list(csv.reader(
            [rowentry], delimiter=',', quotechar='"')
        )[0]
    elif delim == ';':
        rowentry = rowentry.rstrip('\n').split(delim)
    else:
        raise NotImplementedError

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


def get_toi_catalog(ver=TODAYSTR):
    """
    Download the TOI-plus catalog, from MIT. (exported from
    https://tev.mit.edu/) note: this catalog is a bit janky. for example, they
    give transit epoch to 5 decimal points in precision and sometimes fewer,
    which introduces a minimum error of 300 seconds (5 minutes) after only one
    year.
    """

    dir_d = {
        'brik':'/home/luke/Dropbox/proj/cdips/data/toi-plus_catalogs/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/toi-plus_catalogs/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/toi-plus_catalogs/',
        'phtess3':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/toi-plus_catalogs/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/toi-plus_catalogs/',
        'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/toi-plus_catalogs/',
        'PU-C02C76B8MD6T':'/Users/luke/Dropbox/proj/cdips/data/toi-plus_catalogs/',
        'marduk.local':'/Users/luke/local/cdips/catalogs/',
        'WWS-C02C76B8MD6T':'/Users/luke/Dropbox/proj/cdips/data/toi-plus_catalogs/'
    }

    toi_stars_dir = dir_d[socket.gethostname()]

    toi_stars_path = os.path.join(
        toi_stars_dir, 'toi-plus-{}.csv'.format(ver)
    )

    if not os.path.exists(toi_stars_path):
        if not os.path.exists(toi_stars_dir):
            os.mkdir(toi_stars_dir)
        df = pd.read_csv('https://tev.mit.edu/data/collection/193/csv/6/',
                         sep=',', comment='#')
        df.to_csv(toi_stars_path, index=False, sep=',')

    else:
        df = pd.read_csv(toi_stars_path, sep=',', comment='#')

    return df



def get_exofop_toi_catalog(ver=TODAYSTR, returnpath=False):
    # https://exofop.ipac.caltech.edu/tess/view_toi.php, with pipe

    dir_d = {
        'brik':'/home/luke/Dropbox/proj/cdips/data/toi-exofop_catalogs/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/toi-exofop_catalogs/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/toi-exofop_catalogs/',
        'phtess3':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/toi-exofop_catalogs/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/toi-exofop_catalogs/',
        'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/toi-exofop_catalogs/',
        'PU-C02C76B8MD6T':'/Users/luke/Dropbox/proj/cdips/data/toi-exofop_catalogs/',
        'marduk.local':'/Users/luke/local/cdips/catalogs/',
        'WWS-C02C76B8MD6T':'/Users/luke/Dropbox/proj/cdips/data/toi-exofop_catalogs/'
    }

    toi_stars_dir = dir_d[socket.gethostname()]

    toi_stars_path = os.path.join(
        toi_stars_dir, 'toi-exofop-{}.csv'.format(ver)
    )

    if not os.path.exists(toi_stars_path):
        if not os.path.exists(toi_stars_dir):
            os.mkdir(toi_stars_dir)
        df =  pd.read_csv(
            'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
        )
        df.to_csv(toi_stars_path, index=False, sep='|')

    else:
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


def get_exofop_ctoi_catalog(ver=TODAYSTR, returnpath=False):

    dir_d = {
        'brik':'/home/luke/Dropbox/proj/cdips/data/ctoi_catalogs/',
        'phtess1':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/ctoi_catalogs/',
        'phtess2':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/ctoi_catalogs/',
        'phtess3':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/ctoi_catalogs/',
        'phn12':'/nfs/phtess1/ar1/TESS/PROJ/lbouma/ctoi_catalogs/',
        'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/ctoi_catalogs/',
        'PU-C02C76B8MD6T':'/Users/luke/Dropbox/proj/cdips/data/ctoi_catalogs/',
        'marduk.local':'/Users/luke/local/cdips/catalogs/',
        'WWS-C02C76B8MD6T':'/Users/luke/Dropbox/proj/cdips/data/ctoi_catalogs/'
    }

    ctoi_dir = dir_d[socket.gethostname()]

    ctoipath = os.path.join(ctoi_dir, 'ctoi-exofop-{}.csv'.format(ver))

    if not os.path.exists(ctoipath):
        if not os.path.exists(ctoi_dir):
            os.mkdir(ctoi_dir)
        ctoidf =  pd.read_csv(
            'https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv'
        )
        ctoidf.to_csv(ctoipath, index=False, sep='|')

    if returnpath:
        return ctoipath
    else:
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


def toiid_to_ticid(toi_id):

    assert isinstance(toi_id, str)

    toidf = get_exofop_toi_catalog()

    sel = toidf['TOI'].astype(str) == toi_id

    try:
        tic_id = str(toidf[sel]['TIC ID'].iloc[0])
        if len(tic_id) > 1:
            return str(tic_id)
        else:
            return None
    except:
        return None


def get_tic_star_information(
    ticid,
    desiredcols=['ID', 'GAIA', 'ra', 'dec', 'Bmag', 'Vmag', 'Jmag', 'Hmag', 'Kmag', 'Tmag',
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
