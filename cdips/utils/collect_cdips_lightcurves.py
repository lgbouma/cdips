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
    sectors=None,
    cams=None,
    ccds=None,
    cdipssymlinkdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS'):

    for sector in sectors:
        sectordir = os.path.join(cdipssymlinkdir,'sector-{}'.format(sector))
        if not os.path.exists(sectordir):
            os.mkdir(sectordir)
            print('made {}'.format(sectordir))

        for cam in cams:
            for ccd in ccds:
                camccddir = os.path.join(cdipssymlinkdir,
                                         sectordir,
                                         'cam{}_ccd{}'.format(cam,ccd))
                if not os.path.exists(camccddir):
                    os.mkdir(camccddir)
                    print('made {}'.format(camccddir))


def given_sector_cam_ccd_get_projid(_sector,_cam,_ccd):

    projid = 1300

    d = {}
    for snum in [2,3,4,5,1]:
        d[snum] = {}
        for cam in range(1,5):
            d[snum][cam] = {}
            for ccd in range(1,5):
                d[snum][cam][ccd] = projid
                projid += 1

    projid = 1500

    for snum in [6,7,8,9,10,11,12,13]:
        d[snum] = {}
        for cam in range(1,5):
            d[snum][cam] = {}
            for ccd in range(1,5):
                d[snum][cam][ccd] = projid
                projid += 1

    return d[_sector][_cam][_ccd]

def symlink_cdips_lcs(
    cdips_ids,
    sectors=range(1,5+1,1),
    cams=range(1,4+1,1),
    ccds=range(1,4+1,1),
    basedir='/nfs/phtess1/ar1/TESS/FFI/LC/FULL/',
    cdipssymlinkdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS',
    lcglob='*_llc.fits'):

    for sector in sectors:
        for cam in cams:
            for ccd in ccds:

                sstr = 's'+str(sector).zfill(4)
                projid = given_sector_cam_ccd_get_projid(sector,cam,ccd)
                dirstr = 'ISP_{}-{}-{}'.format(cam,ccd,projid)

                lcpaths = np.array(
                    glob(os.path.join(basedir, sstr, dirstr, lcglob))
                )

                have_ids = np.array([
                    os.path.basename(lcpath).split('_llc.fits')[0]
                    for lcpath in lcpaths]
                )

                is_cdips = np.in1d(have_ids, cdips_ids)

                if len(lcpaths[is_cdips]) == 0:
                    print('{}: did not find any CDIPS stars in these lcs!!'.
                          format(dirstr))
                    continue

                print('{}: begin symlinking'.format(dirstr))
                for lcpath in lcpaths[is_cdips]:
                    dst = os.path.join(cdipssymlinkdir,
                                       'sector-{}'.format(sector),
                                       'cam{}_ccd{}'.format(cam,ccd),
                                       os.path.basename(lcpath))
                    if not os.path.exists(dst):
                        os.symlink(lcpath, dst)
                        print('\tsymlink {} -> {}'.format(lcpath,dst))
                    else:
                        print('\t found {}'.format(dst))

def get_cdips_sourceids(ver=0.2):

    cdips_stars_path = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/OC_MG_FINAL_GaiaRp_lt_16_v{}.csv'.format(ver)
    )

    df = pd.read_csv(cdips_stars_path, sep=';')

    cdips_sourceids = np.array(df['source_id']).astype(str)

    # we've saved names as 19-character gaia id strings. if the names in the
    # cluster list have different max character length, is bad.
    assert np.max(list(map(lambda x: len(x), cdips_sourceids))) == 19

    return cdips_sourceids

def plot_cdips_lcs(
    cdipssymlinkdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS',
    sectors=[2],
    cams=[3],
    ccds=[1]):

    for sector in sectors:
        for cam in cams:
            for ccd in ccds:
                lcpaths = glob(os.path.join(cdipssymlinkdir,
                                            'sector-{}'.format(sector),
                                            'cam{}_ccd{}'.format(cam,ccd),
                                            '*_llc.fits'))

                from lcstatistics import plot_raw_tfa_bkgd_fits
                for lcpath in lcpaths:
                    savdir = os.path.dirname(lcpath)
                    plot_raw_tfa_bkgd_fits(lcpath, savdir)

def main(
    make_symlinks=1,
    make_plots=1,
    sectors=None,
    cams=None,
    ccds=None,
    OC_MG_CAT_ver=0.2
):

    if make_symlinks:
        make_local_lc_directories(sectors=sectors, cams=cams, ccds=ccds)
        cdips_sourceids = get_cdips_sourceids(ver=OC_MG_CAT_ver)
        symlink_cdips_lcs(cdips_sourceids, sectors=sectors, cams=cams, ccds=ccds)

    if make_plots:
        plot_cdips_lcs(sectors=sectors, cams=cams)



if __name__ == "__main__":

    main(
        sectors=[6],
        cams=[1,2,3],
        ccds=[1,2,3,4],
        make_symlinks=1,
        make_plots=0,
        OC_MG_CAT_ver=0.2
    )
