"""
use WCS in the given calibrated images to compute how many CDIPS stars are
expected to fall on silicon. (in every sector, over all images)

environment: brik

preparation step:
    $ scp lbouma@phn12:/nfs/phtess2/ar0/TESS/FFI/RED/sector-7/cam?_ccd?/tess2019013135936*_cal_img.fits .
"""
import os, textwrap
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from cdips.utils import collect_cdips_lightcurves as ccl
from glob import glob

from astropy.io import fits
from astropy import wcs

from astropy.coordinates import SkyCoord

from numpy import array as nparr
from astropy import units as u
from datetime import datetime

def how_many_cdips_stars_on_silicon(sector=None, ver=None):

    fitsdir = (
        '/nfs/phtess2/ar0/TESS/FFI/RED/sector-{}/'.
        format(sector)
    )
    fnames = np.sort(glob(os.path.join(fitsdir,'cam*_ccd*/*cal_img.fits')))

    df = ccl.get_cdips_catalog(ver=ver)
    c = SkyCoord(nparr(df['ra']), nparr(df['dec']), unit=(u.deg,u.deg))

    N_onchip = {}
    for fname in fnames:

        print('{}...'.format(fname))
        hdul = fits.open(fname)
        # nb. this ignores the trim, but for statistics it's ok
        w = wcs.WCS(hdul[0].header)

        try:
            x, y = w.all_world2pix(c.ra, c.dec, 1)

        except wcs.wcs.NoConvergence as e:

            ind_div = e.divergent
            ind_slow = e.slow_conv
            best_soln = e.best_solution
            acc = e.accuracy

            ind_ok = np.arange(0, len(best_soln))
            in_div = np.in1d(ind_ok, ind_div)
            in_slow = np.in1d(ind_ok, ind_slow)
            ind_ok = ind_ok[~in_div & ~in_slow]

            # print("Indices of diverging points: {0}".format(ind_div))
            # print("Indices of poorly converging points: {0}".format(ind_slow))
            # print("Diverging best solution:\n{}".format(best_soln[ind_div]))
            # print("Poorly converging best solution:\n{}".format(best_soln[ind_slow]))
            # print("OK in best solution:\n{0}".format(best_soln[ind_ok]))
            # print("Achieved accuracy in OK soln:\n{0}".format(acc[ind_ok]))

            x, y = best_soln.T

            onchip = (x > 0) & (x < 2048) & (y > 0) & (y < 2048)

            N_onchip[fname] = len(best_soln[onchip])

    N_tot = np.sum([v for v in N_onchip.values()])
    dictstr = "\n\t".join("{}: {}".format(k, v) for k, v in N_onchip.items())

    outpath = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/star_catalog/'+
        'how_many_cdips_stars_on_silicon_sector{}.txt'.
        format(sector)
    )
    with open(outpath,'w') as f:
        txt = (
        """
        made by: how_many_cdips_stars_on_silicon.py
        date: {date}

        CDIPS catalog version num: {ver}

        Sector number: {sector}

        Total number of CDIPS stars expected on silicon: {N_tot}

        Details per camera:

        {N_onchip}
        """
        ).format(
            date=datetime.utcnow().isoformat(),
            ver=ver,
            sector=sector,
            N_tot=N_tot,
            N_onchip=dictstr
        )

        f.write(textwrap.dedent(txt))
        print('made {}'.format(outpath))

if __name__ == "__main__":

    sector = 7
    ver = 0.6

    how_many_cdips_stars_on_silicon(sector=sector, ver=ver)
