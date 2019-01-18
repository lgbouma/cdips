"""
scatter plots of magnitudes vs various parameters... to see whether the EPD
linear fitting approach works.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from glob import glob
import os

from astropy.io import fits

def get_data(n_lcs=10):
    # return list of length n_lcs, where each element is a FITS_rec table
    # containing all the needed information.

    lcdir = '../data/lightcurves/'
    lcfiles = glob(os.path.join(lcdir, '*_llc.fits'))

    if len(lcfiles)>n_lcs:
        lcfiles = np.random.choice(lcfiles, size=n_lcs, replace=False)
    else:
        pass

    datalist = []
    for lcfile in lcfiles:
        hdulist = fits.open(lcfile)
        datalist.append(hdulist[1].data)

    return lcfiles, datalist


def make_scatter_plots(lcpaths, lcdatalist, magtype='IRM1'):

    xcols = ['FSV','FDV','FKV','XIC','YIC','CCDTEMP']
    xkeys = ['s','d','k','x','y','T']
    desiredx = ['s', 'd', 'k', 'x', 'y', 'T', 'decimal_x', 'decimal_y',
                'sin(2pi*x)', 'sin(2pi*y)', 'cos(2pi*x)', 'cos(2pi*y)', 's^2',
                'd^2', 'k^2', 's*d', 'd*k', 's*k']

    for lcpath, lc in zip(lcpaths, lcdatalist):

        d = {}
        for xcol, k in zip(xcols, xkeys):
            d[k] = lc[xcol]

        for xstr in desiredx:

            savdir = '../results/epd_parameter_selection/'
            savname = '{}_vs_{}_{}.png'.format(
                magtype, xstr, os.path.basename(lcpath))
            savpath = os.path.join(savdir, savname)

            if os.path.exists(savpath):
                print('found {}, continue'.format(savpath))
                continue

            plt.close('all')
            f,ax = plt.subplots(figsize=(4,3))

            yval = lc[magtype]

            if xstr in ['s','d','k','x','y','T']:
                xval = d[xstr]

            elif xstr == 'decimal_x':
                xval = d['x'] - np.floor(d['x'])

            elif xstr == 'decimal_y':
                xval = d['y'] - np.floor(d['y'])

            elif xstr == 'sin(2pi*x)':
                xval = np.sin(2*np.pi*d['x'])

            elif xstr == 'sin(2pi*y)':
                xval = np.sin(2*np.pi*d['y'])

            elif xstr == 'cos(2pi*x)':
                xval = np.cos(2*np.pi*d['x'])

            elif xstr == 'cos(2pi*y)':
                xval = np.cos(2*np.pi*d['y'])

            elif xstr == 's^2':
                xval = d['s']*d['s']

            elif xstr == 'd^2':
                xval = d['d']*d['d']

            elif xstr == 'k^2':
                xval = d['k']*d['k']

            elif xstr == 's*d':
                xval = d['s']*d['d']

            elif xstr == 'd*k':
                xval = d['d']*d['k']

            elif xstr == 's*k':
                xval = d['s']*d['k']

            ax.scatter(xval, yval, s=15, alpha=0.8, lw=0)
            ax.set_xlabel(xstr)
            ax.set_ylabel(magtype)

            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')

            ax.set_ylim(
                [np.mean(yval)-3*np.std(yval),
                 np.mean(yval)+3*np.std(yval) ]
            )
            ax.set_xlim(
                [np.mean(xval)-3*np.std(xval),
                 np.mean(xval)+3*np.std(xval) ]
            )

            f.tight_layout()
            f.savefig(savpath, bbox_inches='tight', dpi=400)

            print('made {}'.format(savpath))


if __name__=="__main__":

    np.random.seed(42)

    n_lcs = 10

    lcpaths, lcdatalist = get_data(n_lcs=n_lcs)

    make_scatter_plots(lcpaths, lcdatalist)
