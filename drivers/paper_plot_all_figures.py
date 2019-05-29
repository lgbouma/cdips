from glob import glob
import datetime, os, pickle, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from numpy import array as nparr
from astropy.io import fits
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy import units as u

OUTDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/paper_figures/'

def main():

    plot_cluster_and_field_star_scatter()

    plot_wcs_verification()

    pass


def get_positions(lcpaths, outpath=None):

    if os.path.exists(outpath):
        print('found {}, reading'.format(outpath))
        return pd.read_csv(outpath)

    xs, ys, ras, decs = [], [], [], []
    for selpath in selpaths:
        hdul = fits.open(selpath)
        xs.append(hdul[0].header['XCC'])
        ys.append(hdul[0].header['YCC'])
        ras.append(hdul[0].header['RA[deg]'])
        decs.append(hdul[0].header['Dec[deg]'])
    xs, ys, ras, decs = nparr(xs), nparr(ys), nparr(ras), nparr(decs)

    outdf = pd.DataFrame({'x':xs,'y':ys,'ra':ras,'dec':decs})
    outdf.to_csv(outpath.replace('.png', '.csv'), index=False)
    print('made {}'.format(outpath.replace('.png', '.csv')))

    return outdf



def plot_cluster_and_field_star_scatter():

    cam = 1

    for ccd in [1,2,3,4]:

        alldir = (
            '/nfs/phtess1/ar1/TESS/FFI/LC/FULL/s0006/ISP_{}-{}-15??'.
            format(cam, ccd)
        )
        # TODO FIXME: switch to phtess2 once they've been re-run
        allglob = '*llc.fits'

        lcpaths = glob(os.path.join(alldir, allglob))


        cdipsdir = (
            '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam{}_ccd{}'.
            format(cam, ccd)
        )
        cdipsglob = 'hlsp_cdips*_llc.fits'

        cdipspaths = glob(os.path.join(cdipsdir, cdipsglob))

        # get (ra,dec,x,y):
        csvpath = os.path.join(
            OUTDIR, 'cluster_and_field_star_positions_cam{}.csv'.format(cam)
        )
        df = get_positions(lcpaths, outpath=csvpath)


        # determine if is CDIPS star or not



    # todo -- eventually plot!
    f, ax = plt.subplots(figsize=(4,4))
    ax.scatter(xs, ys, c='k', alpha=0.5, s=0.5, rasterized=True, linewidths=0)
    ax.set_title(os.path.basename(lcdir))

    ax.set_xlabel('x on photref')
    ax.set_ylabel('y on photref')

    f.savefig(outpath, bbox_inches='tight', dpi=350)
    print('made {}'.format(outpath))



def plot_wcs_verification():
    pass

if __name__ == "__main__":
    main()
