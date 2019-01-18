import numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
import os
from shutil import copyfile

from astropy.io import fits

def collect_lightcurve_set(lcdir = '/home/luke/local/tess-trex/lightcurves'):
    """
    make file with x,y,Teff, and lc paths, for selection.
    """

    lcpaths = glob(os.path.join(lcdir,'*_llc.fits'))

    _paths, _xcc, _ycc, _teff = [], [], [], []
    for ix, lcpath in enumerate(lcpaths):

        print('{}/{}'.format(ix, len(lcpaths)))

        hdulist = fits.open(lcpath)

        hdr = hdulist[0].header

        _xcc.append(hdr['XCC'])
        _ycc.append(hdr['YCC'])
        _paths.append(lcpath)
        _teff.append(hdr['teff_val'])

    df = pd.DataFrame({'xcc':_xcc, 'ycc':_ycc, 'paths':_paths, 'teff':_teff})

    outpath = '../data/projid1030_cam3_ccd3_lcinfo.csv'
    df.to_csv(outpath, index=False)
    print('saved {}'.format(outpath))


def copy_lightcurves_mkdirs(lcinfopath):

    df = pd.read_csv(lcinfopath)

    sel_corner = (
        (df['xcc'] > 250) & (df['xcc'] < 350)
        &
        (df['ycc'] > 250) & (df['ycc'] < 350)
    )

    sel_center = (
        (df['xcc'] > 1650) & (df['xcc'] < 1750)
        &
        (df['ycc'] > 1650) & (df['ycc'] < 1750)
    )

    f,ax = plt.subplots()
    ax.scatter(df['xcc'],df['ycc'],rasterized=True,s=3)
    ax.set_xlabel('xcc')
    ax.set_ylabel('ycc')
    figpath = '../results/sanity_checks/lc_positions.png'
    f.savefig(figpath)
    print('saved {}'.format(figpath))

    ##########################################
    print('copying {} lcs from center'.format(len(df[sel_center])))
    print('copying {} lcs from corner'.format(len(df[sel_corner])))

    for inpath in df.loc[sel_center, 'paths']:
        indir = os.path.dirname(inpath)
        inname = os.path.basename(inpath)
        outdir = '../results/projid1030_lc_fit_check/center_lcs'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outpath = os.path.join(outdir, inname)
        copyfile(inpath, outpath)
        print('{} -> {}'.format(inpath, outpath))

    for inpath in df.loc[sel_corner, 'paths']:
        indir = os.path.dirname(inpath)
        inname = os.path.basename(inpath)
        outdir = '../results/projid1030_lc_fit_check/corner_lcs'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outpath = os.path.join(outdir, inname)
        copyfile(inpath, outpath)
        print('{} -> {}'.format(inpath, outpath))


if __name__=="__main__":

    lcinfopath = '../data/projid1030_cam3_ccd3_lcinfo.csv'
    if not os.path.exists(lcinfopath):
        collect_lightcurve_set()

    copy_lightcurves_mkdirs(lcinfopath)
