"""
scatter plots of magnitudes vs various parameters... to see whether the EPD
linear fitting approach works.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from glob import glob
import os

from astropy.io import fits

from bspline_fit_xyTmag_BIC_approach import homog_get_data

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_flux_vs_EPD_param(lcpaths, lcdatalist, magtype='IRM1',
                                frac_of_lc=1.0, savdir=None):

    xcols = ['TMID_BJD','XIC','YIC','CCDTEMP','FSV','FDV','FKV','BGV']
    xkeys = ['tmidbjd','x','y','T','s','d','k','bkgd']

    for lcpath, lc in zip(lcpaths, lcdatalist):

        savname = '{}_vs_EPDparams_frac{:.1f}_{}.png'.format(
            magtype, frac_of_lc, os.path.splitext(os.path.basename(lcpath))[0])

        savpath = os.path.join(savdir, savname)

        if os.path.exists(savpath):
            print('found {}, continue'.format(savpath))
            continue

        time = lc['TMID_UTC']

        flux = lc[magtype]
        timeind = np.argsort(time)
        d = {}
        for xcol, k in zip(xcols, xkeys):
            # pandas/seaborn wants little-endian
            le_array = lc[xcol].byteswap().newbyteorder()
            # sort everything by time
            time_sorted_arr = le_array[timeind]
            # take the cut so that you deal with orbit-specific data, if
            # desired
            d[k] = time_sorted_arr[:int(frac_of_lc*len(time_sorted_arr))]

        d[magtype] = flux[timeind][:int(frac_of_lc*len(time_sorted_arr))]

        df = pd.DataFrame(d)

        if np.all(pd.isnull(df[magtype])):
            print('mags are all NaN for {}, continue'.format(savname))
            continue

        # PLOT flux vs each xcol.
        plt.close('all')

        fig,axs = plt.subplots(nrows=len(xcols), ncols=1, figsize=(6,6))
        axs = axs.flatten()

        cm = plt.cm.get_cmap('viridis')

        timeval = np.array(df['tmidbjd'])
        for ax, xkey in zip(axs, xkeys):
            xval = df[xkey]
            yval = df[magtype]

            cs = ax.scatter(xval, yval, rasterized=True, alpha=0.8, zorder=3,
                            c=timeval, lw=0, s=3, cmap=cm)

            ax.set_ylabel(magtype+' vs '+xkey, fontsize='xx-small')

            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')
            ax.xaxis.set_tick_params(labelsize='xx-small')
            ax.yaxis.set_tick_params(labelsize='xx-small')

        cbar_ax = fig.add_axes([1., 0.15, 0.03, 0.7])
        cbar = fig.colorbar(cs, cax=cbar_ax, pad=0.01, use_gridspec=True)
        cbar_ax.yaxis.set_tick_params(labelsize='xx-small')

        cbar.ax.tick_params(direction='in')
        cbar.ax.set_ylabel('Time (BJD)', fontsize='xx-small')

        fig.tight_layout(h_pad=-1, w_pad=-1)
        fig.savefig(savpath, dpi=400, bbox_inches='tight')
        print('made {}'.format(savpath))


def plot_EPD_parameters_vs_time(lcpaths, lcdatalist, magtype='IRM1',
                                frac_of_lc=1.0, savdir=None):

    xcols = ['TMID_UTC','XIC','YIC','CCDTEMP','FSV','FDV','FKV','BGV']
    xkeys = ['tmidbjd','x','y','T','s','d','k','bkgd']

    for lcpath, lc in zip(lcpaths, lcdatalist):

        savname = 'EPDparams_vs_time_frac{:.1f}_{}.png'.format(
            frac_of_lc, os.path.splitext(os.path.basename(lcpath))[0])

        savpath = os.path.join(savdir, savname)

        if os.path.exists(savpath):
            print('found {}, continue'.format(savpath))
            continue

        time = lc['TMID_UTC']

        flux = lc[magtype]
        timeind = np.argsort(time)
        d = {}
        for xcol, k in zip(xcols, xkeys):
            # pandas/seaborn wants little-endian
            le_array = lc[xcol].byteswap().newbyteorder()
            # sort everything by time
            time_sorted_arr = le_array[timeind]
            # take the cut so that you deal with orbit-specific data, if
            # desired
            d[k] = time_sorted_arr[:int(frac_of_lc*len(time_sorted_arr))]

        d[magtype] = flux[timeind][:int(frac_of_lc*len(time_sorted_arr))]

        df = pd.DataFrame(d)

        if np.all(pd.isnull(df[magtype])):
            print('mags are all NaN for {}, continue'.format(savname))
            continue

        # PLOT each xcol vs time
        plt.close('all')

        fig,axs = plt.subplots(nrows=len(xcols), ncols=1, figsize=(6,6),
                               sharex=True)
        axs = axs.flatten()

        timeval = np.array(df['tmidbjd'])
        for ax, xkey in zip(axs, xkeys):
            if xkey=='tmidbjd':
                xkey = magtype
            yval = df[xkey]
            xval = df['tmidbjd']

            ax.scatter(xval, yval, rasterized=True, alpha=0.8, zorder=3, c='k',
                       lw=0, s=3)

            ax.set_ylabel(xkey, fontsize='xx-small')

            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')
            ax.xaxis.set_tick_params(labelsize='xx-small')
            ax.yaxis.set_tick_params(labelsize='xx-small')

        ax.set_xlabel('BJD')

        fig.tight_layout(h_pad=-1, w_pad=-1)
        fig.savefig(savpath, dpi=400, bbox_inches='tight')
        print('made {}'.format(savpath))




if __name__=="__main__":

    #for lctype in ['corner','center']:
    for lctype in ['center']:

        savdir = '../results/projid1030_lc_fit_check/{}_lcs/'.format(lctype)

        lcpaths, lcdatalist =  homog_get_data(lctype=lctype)

        plot_EPD_parameters_vs_time(lcpaths, lcdatalist, magtype='IRM1',
                                    frac_of_lc=1.0, savdir=savdir)

        plot_EPD_parameters_vs_time(lcpaths, lcdatalist, magtype='IRM1',
                                    frac_of_lc=0.4, savdir=savdir)

        plot_flux_vs_EPD_param(lcpaths, lcdatalist, magtype='IRM1',
                               frac_of_lc=0.4, savdir=savdir)

        #FIXME: you want this to by X vs time, Y vs time, etc.... not the plot
        #that you have made!!!
