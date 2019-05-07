"""
First, run drivers/make_fficut_wget_script.py
Then, run this.
"""
from glob import glob
import datetime, os, pickle, shutil
import numpy as np, pandas as pd

from numpy import array as nparr
from astropy.io import fits
from datetime import datetime

from cdips.lcproc import mask_orbit_edges as moe

from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord

def make_wget_script(tfasrdir, xlen_px=10, ylen_px=10, tesscutvernum=0.1):
    """
    From astrocut/TESScut API, make shell script used to get FFI cutouts for
    all the TCEs.  Example curl line ::

        curl -O "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=345.3914&dec=-22.0765&y=15&x=10&units=px&sector=All"

    the corresponding driver is drivers/fficut_wget_script.py

    ############
    Args:

    tfasrdir: str
        The directory with signal-reconstructed TFA lightcurves for TCEs, e.g.,
        "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6_TFA_SR"

    """

    lcpaths = glob(os.path.join(tfasrdir,'*.fits'))

    outlines = []
    ras, decs= [], []
    for lcpath in lcpaths:

        hdul = fits.open(lcpath)
        hdr = hdul[0].header

        try:
            ra = hdr["RA[deg]"]
            dec = hdr["Dec[deg]"]
        except:
            ra = hdr["RA_OBJ"]
            dec = hdr["DEC_OBJ"]

        outtemplate = (
        '\ncurl -O "https://mast.stsci.edu/tesscut/api/'
        'v{tesscutvernum:.1f}'
        '/astrocut?'
        'ra={ra:.4f}&dec={dec:.4f}&y={ylen_px:d}&x={xlen_px:d}'
        '&units=px&sector=All"'
        )

        outline = outtemplate.format(
            tesscutvernum=tesscutvernum,
            ra=ra,
            dec=dec,
            ylen_px=ylen_px,
            xlen_px=xlen_px
        )

        outlines.append(outline)
        ras.append(ra)
        decs.append(dec)

    outlines.insert(0,'#!/usr/bin/env bash\n')

    outdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_cutouts'
    outdir = os.path.join(outdir,tfasrdir.rstrip('/').split('/')[-1])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(outdir, 'wget_the_TCE_cutouts.sh')
    with open(outpath, 'w') as f:
        f.writelines(outlines)
    print('made {}'.format(outpath))

    outpath = os.path.join(outdir, 'ra_dec_to_lcpath_dict.csv')
    df = pd.DataFrame({'lcpath':lcpaths,'ra':ras,'dec':decs})
    df.to_csv(outpath)
    print('made {}'.format(outpath))



def do_centroid_analysis(sectornum=6):

    #
    # assume fficut_wget_script has been run.
    # outdir is full of files like
    # tess-s0006-1-1_84.0898_-2.1764_10x10_astrocut.fits
    #
    cutdir = "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_cutouts/temp/"
    fficutpaths = os.path.join(cutdir, 'tess*.fits')

    dfpath = os.path.join(cutdir, 'ra_dec_to_lcpath_dict.csv')
    df = pd.read_csv(dfpath)

    szfill = str(sectornum).zfill(4)

    globs = []
    for ra,dec in zip(nparr(df['ra']),nparr(df['dec'])):

        rastr = '{:.4f}'.format(ra)
        decstr = '{:.4f}'.format(dec)

        globs.append(
                'tess-s{snum:s}-?-?_{ra:s}_{dec:s}_10x10_astrocut.fits'.
                format(snum=szfill,ra=rastr.rstrip('0'),dec=decstr.rstrip('0'))
        )

    globpaths = [os.path.join(cutdir,g) for g in globs]
    globpathexists = nparr([len(glob(g)) for g in globpaths]).astype(bool)

    if not np.all(globpathexists):
        raise AssertionError('not np.all(globpathexists)')

    df['cutpath'] = [glob(g)[0] for g in globpaths]

    #NOTE: df now contains lcpath and cutpath. So you can now match in the
    #period-finding information, and do the image averaging.

    #TODO TODO TODO TODO

    import IPython; IPython.embed() #FIXME
    assert 0

    for f in fficutpaths:
        measure_centroid(f)


def make_mean_intransit_and_oot_images_per_transit():
    pass

def make_mean_intransit_and_oot_images_avg_over_transit():
    pass

def make_oot_minus_intransit_image():
    pass

def measure_centroid(f):
    # f: fficutpath


    make_mean_intransit_and_oot_images_per_transit(f)

    make_mean_intransit_and_oot_images_avg_over_transit(f)

    make_oot_minus_intransit_image(f)
    pass
