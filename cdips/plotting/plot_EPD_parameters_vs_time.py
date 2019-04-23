"""
(mag, T, x, y, s, d, k, bgd) vs time
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from glob import glob
import os

from astropy.io import fits

from bspline_fit_xyTmag_BIC_approach import homog_get_data

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_EPD_parameters_vs_time(lcpaths, lcdatalist, magtype='IRM1',
                                frac_of_lc=1.0, savdir=None):

    xcols = ['TMID_BJD','XIC','YIC','CCDTEMP','FSV','FDV','FKV','BGV']
    xkeys = ['tmidbjd','x','y','T','s','d','k','bkgd']

    for lcpath, lc in zip(lcpaths, lcdatalist):

        savname = 'EPDparams_vs_time_frac{:.1f}_{}.png'.format(
            frac_of_lc, os.path.splitext(os.path.basename(lcpath))[0])

        savpath = os.path.join(savdir, savname)

        if os.path.exists(savpath):
            print('found {}, continue'.format(savpath))
            continue

        time = lc['TMID_BJD']

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

        xoffset = int(np.median(df['tmidbjd']))
        xval = df['tmidbjd'] - xoffset

        for ax, xkey in zip(axs, xkeys):
            if xkey=='tmidbjd':
                xkey = magtype
            yval = df[xkey]

            if xkey in ['x','y']:
                yoffset = int(np.mean(yval))
                yval -= yoffset
                xkey += '- {:d}'.format(yoffset)
            elif xkey in [magtype]:
                yoffset = np.round(np.median(yval), decimals=1)
                yval -= yoffset
                yval *= 1e3
                xkey += '- {:.1f} [mmag]'.format(yoffset)

            ax.scatter(xval, yval, rasterized=True, alpha=0.8, zorder=3, c='k',
                       lw=0, s=3)

            ax.set_ylabel(xkey, fontsize='xx-small')

            #ax.xaxis.set_ticks_position('both')
            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')
            ax.xaxis.set_tick_params(labelsize='xx-small')
            ax.yaxis.set_tick_params(labelsize='xx-small')

            if magtype in xkey:
                ylim = ax.get_ylim()
                ax.set_ylim((max(ylim), min(ylim)))

        ax.set_xlabel('BJD$_{\mathrm{TDB}}$ - '+'{}'.format(xoffset))

        fig.tight_layout(h_pad=-1)
        fig.savefig(savpath, dpi=400, bbox_inches='tight')
        print('made {}'.format(savpath))




if __name__=="__main__":

    for lctype in ['center','corner']:

        savdir = '../results/projid1088_cam2_ccd2_lcs/{}_lcs/'.format(lctype)
        lcdir = savdir

        lcpaths, lcdatalist = homog_get_data(lctype=lctype, lcdir=lcdir)

        plot_EPD_parameters_vs_time(lcpaths, lcdatalist, magtype='IRM1',
                                    frac_of_lc=1.0, savdir=savdir)

        #plot_EPD_parameters_vs_time(lcpaths, lcdatalist, magtype='IRM1',
        #                            frac_of_lc=0.4, savdir=savdir)

