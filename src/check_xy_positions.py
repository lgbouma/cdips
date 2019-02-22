"""
given a directory with lightcurves from same camera... check the stars are
spread all across the x,y plane.
"""
import os
from glob import glob
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.io import fits

def make_csv_of_positions():
    lcdir = '/home/luke/temp/ISP_2-2-1190'
    lcfiles = glob(os.path.join(lcdir,'*_llc.fits'))

    x, y = [], []
    for ix, lcfile in enumerate(lcfiles):
        print('{}/{}'.format(ix, len(lcfiles)))

        hdulist = fits.open(lcfile)

        xcc, ycc = hdulist[0].header['XCC'], hdulist[0].header['YCC']
        x.append(xcc)
        y.append(ycc)

    x, y = np.array(x), np.array(y)

    df = pd.DataFrame({'x':x,'y':y})

    outpath = '../results/sanity_checks/xy_positions_projid1190.csv'
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


if __name__=="__main__":

    csvpath = '../results/sanity_checks/xy_positions_projid1190.csv'

    if not os.path.exists(csvpath):
        make_csv_of_positions()

    df = pd.read_csv(csvpath)

    f,ax = plt.subplots()
    ax.scatter(df['x'],df['y'], s=1)

    outpath = '../results/sanity_checks/xy_positions_projid1190.png'
    f.savefig(outpath, dpi=400)
    print('made {}'.format(outpath))
