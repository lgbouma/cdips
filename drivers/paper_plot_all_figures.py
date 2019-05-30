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
    #
    # fig N: positions of field and cluster stars
    #
    plot_cluster_and_field_star_scatter(overwrite=0)

    #
    # fig N: RMS vs catalog T mag
    #
    plot_rms_vs_mag(overwrite=1)

    #
    # fig N: wcs quality verification
    #
    plot_wcs_verification()

    pass


def plot_rms_vs_mag(overwrite=0):
    pass


def plot_cluster_and_field_star_scatter(overwrite=0):

    outpath = os.path.join(OUTDIR, 'cam1_cluster_field_star_positions.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    sectornum = 6
    cam = 1
    ccds = [1] #FIXME add ccds
    csvpaths = []
    N_max = 100000

    for ccd in ccds:

        # all lightcurves
        lcdir = (
            '/nfs/phtess2/ar0/TESS/FFI/LC/FULL/s0006/ISP_{}-{}-15??/'.
            format(cam, ccd)
        )
        lcglob = '*_llc.fits'
        alllcpaths = glob(os.path.join(lcdir, lcglob))

        # CDIPS LCs
        cdipslcdir = (
            '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam{}_ccd{}'.
            format(cam, ccd)
        )

        csvpath = os.path.join(
            OUTDIR,'cam{}_ccd{}_cluster_field_star_positions.csv'.
            format(cam, ccd)
        )

        csvpaths.append(csvpath)

        get_cluster_and_field_star_positions(alllcpaths, cdipslcdir, csvpath,
                                             sectornum, N_desired=N_max)


    df = pd.concat((pd.read_csv(f) for f in csvpaths))

    plot_cluster_and_field_star_positions(df, outpath, title='s0006,cam1')


def get_cluster_and_field_star_positions(lcpaths, cdipslcdir, outpath,
                                         sectornum, cdipsvnum=1,
                                         N_desired=200):

    if len(lcpaths) > N_desired:
        selpaths = np.random.choice(lcpaths, size=N_desired, replace=False)
    else:
        selpaths = lcpaths

    print('beginning get LC positions on {} LCs'.format(len(selpaths)))

    if os.path.exists(outpath):
        print('found {}, skip'.format(outpath))
        return

    gaiaids, xs, ys, ras, decs, iscdips = [], [], [], [], [], []
    for selpath in selpaths:
        hdul = fits.open(selpath)
        lcgaiaid = hdul[0].header['Gaia-ID']
        gaiaids.append(lcgaiaid)
        xs.append(hdul[0].header['XCC'])
        ys.append(hdul[0].header['YCC'])
        ras.append(hdul[0].header['RA[deg]'])
        decs.append(hdul[0].header['Dec[deg]'])

        cdipsname = (
            'hlsp_cdips_tess_ffi_'
            'gaiatwo{zsourceid}-{zsector}_'
            'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            zsourceid=str(lcgaiaid).zfill(22),
            zsector=str(sectornum).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )
        if os.path.exists(os.path.join(cdipslcdir,cdipsname)):
            iscdips.append(True)
        else:
            iscdips.append(False)

    xs, ys, ras, decs = nparr(xs), nparr(ys), nparr(ras), nparr(decs)
    gaiaids = nparr(gaiaids)
    iscdips = nparr(iscdips)

    outdf = pd.DataFrame({'x':xs,'y':ys,'ra':ras,'dec':decs,
                          'gaiaid':gaiaids, 'iscdips': iscdips})
    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def plot_cluster_and_field_star_positions(df, outpath, title='s0006,cam1'):
    """
    scatter of (ra,dec) for [subset of] stars with lightcurves.

    gray background points: field stars

    blue foreground points: cluster stars
    """

    f, ax = plt.subplots(figsize=(4,4))

    iscdips = df['iscdips']

    ax.scatter(df[~iscdips]['ra'], df[~iscdips]['dec'], c='k', alpha=0.5,
               s=0.5, rasterized=True, linewidths=0, zorder=1)
    ax.scatter(df[iscdips]['ra'], df[iscdips]['dec'], c='C0', alpha=0.8,
               s=0.5, rasterized=True, linewidths=0, zorder=2)

    ax.set_title(title)

    ax.set_xlabel('ra [deg]')
    ax.set_ylabel('dec [deg]')

    f.savefig(outpath, bbox_inches='tight', dpi=350)
    print('made {}'.format(outpath))


def plot_wcs_verification():
    pass


if __name__ == "__main__":
    main()
