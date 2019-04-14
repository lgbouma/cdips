"""
Collect lightcurves from phtess[1|2] for the CDIPS project.

How?

* Get lists of lightcurves that were made for each requested projid / sector /
cam / ccd.

* Crossmatch these lists against the CDIPS star catalog
(OC_MG_FINAL_GaiaRp_lt_16.csv).

* symlink them to a PROJ directory.

"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from glob import glob
import os, pickle, subprocess, itertools

from numpy import array as nparr

from astropy.table import Table
from astropy import units as u, constants as c

from astroquery.gaia import Gaia

from astropy.io.votable import from_table, writeto, parse

def scp_lightcurves(lcbasenames,
                    lcdir='/nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0002/ISP_1-2-1163',
                    outdir='../data/cluster_data/lightcurves/Blanco_1/'):

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for lcbasename in lcbasenames:

        fromstr = "lbouma@phn12:{}/{}".format(lcdir, lcbasename)
        tostr = "{}/.".format(outdir)

        p = subprocess.Popen([
            "scp",
            fromstr,
            tostr,
        ])
        sts = os.waitpid(p.pid, 0)

    return 1

def make_local_lc_directories(
    sectors=range(1,5+1,1),
    cams=range(1,4+1,1),
    ccds=range(1,4+1,1),
    cdipslcdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS'):

    for sector in sectors:
        sectordir = os.path.join(cdipslcdir,'sector-{}'.format(sector))
        if not os.path.exists(sectordir):
            os.mkdir(sectordir)
            print('made {}'.format(sectordir))
    for cam,ccd in zip(cams, ccds):
        camccddir = os.path.join(cdipslcdir,sector,'cam{}_ccd{}'.format(cam,ccd))
        if not os.path.exists(camccddir):
            os.mkdir(camccddir)
            print('made {}'.format(camccddir))


def given_sector_cam_ccd_get_projid(_sector,_cam,_ccd):

    projid = 1300

    d = {}
    for snum in [2,3,4,5,1]:
        d[sector] = {}
        for cam in range(1,5):
            d[sector][cam] = cam
            for ccd in range(1,5):
                d[sector][cam][ccd]
                projid += 1

    return d[_sector][_cam][_ccd]

def symlink_cdips_lcs(
    cdips_ids,
    sectors=range(1,5+1,1),
    cams=range(1,4+1,1),
    ccds=range(1,4+1,1),
    basedir='/nfs/phtess1/ar1/TESS/FFI/LC/FULL/',
    cdipslcdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_LCS',
    lcglob='*_llc.fits'):

    for sector in sectors:
        for cam,ccd in zip(cams, ccds):

            sstr = 's'+str(sector).zfill(4)
            projid = given_sector_cam_ccd_get_projid(sector,cam,ccd)
            dirstr = 'ISP_{}-{}-{}'.format(cam,ccd,projid)

            lcpaths = glob(os.path.join(basedir, sstr, dirstr, lcglob))

            have_ids = np.array([
                lcpath.split('_llcs.fits')[0] for lcpath in lcpaths])

            is_cdips = np.in1d(have_ids, cdips_ids)

            import IPython; IPython.embed()
            if len(lcpaths[is_cdips]) == 0:
                print('{}: did not find any CDIPS stars in these lcs!!'.
                      format(dirstr))
                continue

            print('{}: begin symlinking'.format(dirstr))
            for lcpath in lcpaths[is_cdips]:
                dst = os.path.join(cdipslcdir,
                                   'sector-{}'.format(sector),
                                   'cam{}_ccd{}'.format(cam,ccd),
                                   os.path.basename(lcpath))
                os.symlink(lcpath, dst)
                print('\tsymlink {} -> {}'.format(src,dst))

def get_cdips_sourceids():

    cdips_stars_path = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/OC_MG_FINAL_GaiaRp_lt_16.csv'
    )

    df = pd.read_csv(cdips_stars_path)

    cdips_sourceids = np.array(df['source_id']).astype(str)

    return cdips_sourceids


def main(
    sectors=range(1,5+1,1),
    cams=range(1,4+1,1),
    ccds=range(1,4+1,1)):

    make_local_lc_directories(sectors=sectors, cams=cams, ccds=ccds)

    cdips_sourceids = get_cdips_sourceids()

    symlink_cdips_lcs(cdips_sourceids, sectors=sectors, cams=cams, ccds=ccds)

if __name__ == "__main__":

    main(
        sectors=[2],
        cams=[1,2,3],
        ccds=[1,2,3,4]
    )
