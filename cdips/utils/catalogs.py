import pandas as pd
import socket, os

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
