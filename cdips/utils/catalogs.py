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


def get_toi_catalog(ver='2019-12-05'):
    # a misnomer: really, the TOI-plus catalog.

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



