"""
magnitude vs (time, T, x, y, s, d, k, bgd). for: single stars. and 4 chosen
stars (for paper).
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from glob import glob
import os

from astropy.io import fits

from bspline_fit_xyTmag_BIC_approach import homog_get_data

from astrobase import lcmath

def plot_mag_vs_EPD_parameters(lcpaths, lcdatalist, magtype='IRM1',
                               get_norbits=1, expected_norbits=2, savdir=None):

    xcols = ['TMID_BJD','CCDTEMP','XIC','YIC','FSV','FDV','FKV']
    xkeys = ['tmidbjd','T','x','y','s','d','k']
    # could also include other parameters, like decimal_x, decimal_y,
    # cross-terms...
    desiredx = xkeys

    if get_norbits not in [1]:
        raise AssertionError('you usually only want to get 1 orbit')

    for lcpath, lc in zip(lcpaths, lcdatalist):

        # collect the data for the first orbit in this sector
        times = lc['TMID_BJD']

        orbitgap = 1. # days
        found_norbits, groups = lcmath.find_lc_timegroups(times, mingap=orbitgap)

        if found_norbits != expected_norbits:
            outmsg = (
                'assuming given two orbits. {} orbits. Time {}'.
                format(norbits, repr(times))
            )
            raise AssertionError(outmsg)

        if get_norbits==1:
            group = groups[0]
        else:
            raise NotImplementedError

        # collect the data for the first orbit in this sector
        d = {}
        for xcol, k in zip(xcols, xkeys):
            d[k] = lc[xcol][group]

        if np.all(pd.isnull(lc[magtype])):
            print('mags are all NaN for {}, continue'.format(savname))
            continue

        ##########################################

        # begin plotting.
        nrows = len(xcols)

        plt.close('all')
        fig,axs = plt.subplots(nrows=nrows, ncols=1, figsize=(3,2*nrows))
        axs = axs.flatten()

        yval = lc[magtype][group]
        yoffset = np.round(np.median(yval),decimals=2)
        yval -= yoffset
        yval *= 1e3

        savname = ('{}_vs_EPD_parameters_{}_norbit{}.png'.
                   format(magtype,
                          os.path.basename(lcpath).replace('.fits',''),
                          get_norbits)
        )
        savpath = os.path.join(savdir, savname)

        if os.path.exists(savpath):
            print('found {}, continue'.format(savpath))
            continue

        for xstr, ax in zip(desiredx, axs):

            if xstr in ['s','d','k','x','y','T','tmidbjd']:
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

            if xstr == 'tmidbjd':
                xval -= np.median(xval)
            if xstr in ['x','y']:
                xoffset = int(np.median(xval))
                xval -= xoffset

            if xstr == 'tmidbjd':
                ax.scatter(xval, yval, rasterized=True, alpha=0.8, zorder=3,
                           c='k', lw=0, s=5)
            else:
                ax.scatter(xval, yval, rasterized=True, alpha=0.8, zorder=3,
                           c='k', lw=0, s=4)

            if xstr == 'tmidbjd':
                xstr = 't - t$_0$ [d]'
            if xstr in ['x','y']:
                xstr = xstr + ' - {}'.format(xoffset)
            ax.text(
                0.98, 0.02, xstr, fontsize='x-small',
                ha='right',va='bottom',
                transform=ax.transAxes
            )
            ax.set_ylabel('{} - {} [mmag]'.format(magtype, yoffset),
                          fontsize='x-small')

            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')

            ylim = ax.get_ylim()
            ax.set_ylim((max(ylim), min(ylim)))

        fig.tight_layout(h_pad=0.15)
        fig.savefig(savpath, dpi=400, bbox_inches='tight')

        print('made {}'.format(savpath))


def plot_mag_vs_EPD_parameters_four_stars(magtype='IRM1', get_norbits=1,
                                          expected_norbits=2, savdir=None):
    """
    for 4 chosen stars, do columns of their flux vs EPD parameters (x,y,T,
    etc).

    two left stars are selected to be representative of center of field.
    two right stars are selected to be representative of corner of field.
    """

    # get the lcpaths and data list for the selected stars

    # brightish, center
    # IRM1_vs_EPD_parameters_4979631232408118784_llc_norbit1.png
    # IRM1_vs_EPD_parameters_4979603607178484608_llc_norbit1.png
    # ditto, corner
    # IRM1_vs_EPD_parameters_4955130505568582528_llc_norbit1.png
    # IRM1_vs_EPD_parameters_4955138064711041280_llc_norbit1.png

    centerdir = '../results/projid1088_cam2_ccd2_lcs/center_lcs/'
    cornerdir = '../results/projid1088_cam2_ccd2_lcs/corner_lcs/'
    lcpaths = [
        os.path.join(centerdir,'4979631232408118784_llc.fits'),
        os.path.join(centerdir,'4979603607178484608_llc.fits'),
        os.path.join(cornerdir,'4955130505568582528_llc.fits'),
        os.path.join(cornerdir,'4955138064711041280_llc.fits'),
    ]

    lcdatalist = []
    for lcfile in lcpaths:
        hdulist = fits.open(lcfile)
        lcdatalist.append(hdulist[1].data)
        hdulist.close()

    xcols = ['TMID_BJD','CCDTEMP','XIC','YIC','FSV','FDV','FKV']
    xkeys = ['tmidbjd','T','x','y','s','d','k']
    desiredx = xkeys

    if get_norbits not in [1]:
        raise AssertionError('you only want to get 1 orbit')

    plt.close('all')
    nrows = len(xcols)
    ncols = len(lcpaths)
    fig,axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols,2*nrows))

    savname = ('{}_vs_EPD_parameters_fourstars_norbit{}.png'.
               format(magtype,
                      get_norbits)
    )
    savpath = os.path.join(savdir, savname)

    if os.path.exists(savpath):
        print('found {}, quitting'.format(savpath))
        return 0

    # iterate over columns
    for lcpath, lc, colaxs in zip(lcpaths, lcdatalist, axs.T):

        # collect the data for the first orbit in this sector
        times = lc['TMID_BJD']

        orbitgap = 1. # days
        found_norbits, groups = lcmath.find_lc_timegroups(times, mingap=orbitgap)

        if found_norbits != expected_norbits:
            outmsg = (
                'assuming given two orbits. {} orbits. Time {}'.
                format(norbits, repr(times))
            )
            raise AssertionError(outmsg)

        if get_norbits==1:
            group = groups[0]
        else:
            raise NotImplementedError

        # collect the data for the first orbit in this sector
        d = {}
        for xcol, k in zip(xcols, xkeys):
            d[k] = lc[xcol][group]

        if np.all(pd.isnull(lc[magtype])):
            print('mags are all NaN for {}, continue'.format(savname))
            continue

        ##########################################

        # begin plotting.
        yval = lc[magtype][group]
        yoffset = np.round(np.median(yval),decimals=2)
        yval -= yoffset
        yval *= 1e3

        for xstr, ax in zip(desiredx, colaxs):

            if xstr in ['s','d','k','x','y','T','tmidbjd']:
                xval = d[xstr]

            if xstr == 'tmidbjd':
                xval -= np.median(xval)
            if xstr in ['x','y']:
                xoffset = int(np.median(xval))
                xval -= xoffset

            if xstr == 'tmidbjd':
                ax.scatter(xval, yval, rasterized=True, alpha=0.8, zorder=3,
                           c='k', lw=0, s=5)
            else:
                ax.scatter(xval, yval, rasterized=True, alpha=0.8, zorder=3,
                           c='k', lw=0, s=4)

            if xstr == 'tmidbjd':
                xstr = 't - t$_0$ [d]'
            if xstr in ['x','y']:
                xstr = xstr + ' - {}'.format(xoffset)
            ax.text(
                0.98, 0.02, xstr, fontsize='x-small',
                ha='right',va='bottom',
                transform=ax.transAxes
            )
            ax.set_ylabel('{} - {} [mmag]'.format(magtype, yoffset),
                          fontsize='x-small')

            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')

            ylim = ax.get_ylim()
            ax.set_ylim((max(ylim), min(ylim)))

    fig.tight_layout(h_pad=0.15)
    fig.savefig(savpath, dpi=400, bbox_inches='tight')

    print('made {}'.format(savpath))


if __name__=="__main__":

    makerandomlcplots = 0
    makefourstarlcplot = 1

    np.random.seed(42)

    if makerandomlcplots:

        for lctype in ['center','corner']:

            savdir = '../results/projid1088_cam2_ccd2_lcs/{}_lcs/'.format(lctype)
            lcdir = savdir

            lcpaths, lcdatalist = homog_get_data(lctype=lctype, lcdir=lcdir)

            plot_mag_vs_EPD_parameters(lcpaths, lcdatalist, magtype='IRM1',
                                       get_norbits=1, expected_norbits=2,
                                       savdir=savdir)

    if makefourstarlcplot:
        plot_mag_vs_EPD_parameters_four_stars(
            magtype='IRM1', get_norbits=1, expected_norbits=2,
            savdir='../results/projid1088_cam2_ccd2_lcs'
        )
