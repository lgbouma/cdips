"""
scatter plots of magnitudes vs various parameters... to see whether the EPD
linear fitting approach works.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from glob import glob
import os

from astropy.io import fits

import seaborn as sns

from scipy.interpolate import splprep, splev

from numpy import array as nparr


def homog_get_data(lctype='corner', lcdir=None):
    # lctype = 'corner' or 'center'

    lcfiles = glob(os.path.join(lcdir, '*_llc.fits'))

    datalist = []
    for lcfile in lcfiles:
        hdulist = fits.open(lcfile)
        datalist.append(hdulist[1].data)
        hdulist.close()

    return lcfiles, datalist



def make_pairplots(lcpaths, lcdatalist, magtype='IRM1', frac_of_lc=0.4,
                   savdir='../results/bspline_fit_xyTmag/'):
    # make pairplots of X,Y,T,MAG,S,D,K, and just X,Y,T,MAG
    # frac_of_lc: for fitting purposes, we want orbit-specific data.

    xcols = ['FSV','FDV','FKV','XIC','YIC','CCDTEMP',magtype]
    xkeys = ['s','d','k','x','y','T',magtype]

    for lcpath, lc in zip(lcpaths, lcdatalist):

        savname = '{}_frac{:.1f}_pairplot_{}.png'.format(
            magtype, frac_of_lc, os.path.splitext(os.path.basename(lcpath))[0])

        savpath = os.path.join(savdir, savname)

        if os.path.exists(savpath):
            print('found {}, continue'.format(savpath))
            continue

        time = lc['TMID_UTC']
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

        df = pd.DataFrame(d)

        if np.all(pd.isnull(df[magtype])):
            print('mags are all NaN for {}, continue'.format(savname))
            continue

        # PLOT X,Y,T,MAG,S,D,K
        plt.close('all')

        g = sns.PairGrid(df)
        g = g.map_diag(plt.hist)
        g = g.map_offdiag(plt.scatter, rasterized=True, s=10, alpha=0.8)

        plt.savefig(savpath, bbox_inches='tight', dpi=400)
        print('made {}'.format(savpath))

        # PLOT SUBSET: ONLY X,Y,T,MAG
        savname = '{}_frac{:.1f}_pairplot_xyTmag_{}.png'.format(
            magtype, frac_of_lc, os.path.splitext(os.path.basename(lcpath))[0])
        savpath = os.path.join(savdir, savname)
        if os.path.exists(savpath):
            print('found {}, continue'.format(savpath))
            continue

        plt.close('all')

        g = sns.PairGrid(df, vars=['x','y','T',magtype])
        g = g.map_diag(plt.hist)
        g = g.map_offdiag(plt.scatter, rasterized=True, s=10, alpha=0.8)

        plt.savefig(savpath, bbox_inches='tight', dpi=400)
        print('made {}'.format(savpath))


def do_bspline_fit_xyTmag(lcpaths, lcdatalist, magtype='IRM1', frac_of_lc=0.4,
                          isffi=True, savdir=None):

    if not isffi:
        raise AssertionError('windowsize for polling std assumes is ffi')

    # spline parameters
    s_initial = 1.0 # smoothness parameter
    korder = 3 # spline order
    nest = -1 # estimate of number of knots needed (-1 = maximal)

    # Here "s" is a hyperparameter. We want to tune it, by chi^2 minimization.
    # (Attempted cross-validation; getting splprep to work in the needed mode
    # of "given some new u values, give me the appropriate spline coefficients"
    # was not working).

    magerrtype = magtype.replace('M','E')
    xcols = ['XIC','YIC','CCDTEMP',magtype,magerrtype]
    xkeys = ['x','y','T',magtype,magerrtype]

    for lcpath, lc in zip(lcpaths, lcdatalist):

        time = lc['TMID_UTC']
        timeind = np.argsort(time)
        d = {}
        for xcol, k in zip(xcols, xkeys):
            # pandas/seaborn wants little-endian
            le_array = lc[xcol].byteswap().newbyteorder()
            # sort everything by time
            time_sorted_arr = le_array[timeind]
            # take the cut so that you fit orbit-specific data
            d[k] = time_sorted_arr[:int(frac_of_lc*len(time_sorted_arr))]

        df = pd.DataFrame(d)

        if np.all(pd.isnull(df[magtype])):
            print('mags are all NaN for {}, continue'.format(savname))
            continue

        savname = '{}_frac{:.1f}_scatterfit_xyTmag_{}.png'.format(
            magtype, frac_of_lc, os.path.splitext(os.path.basename(lcpath))[0])
        savpath = os.path.join(savdir, savname)

        #FIXME put in
        # if os.path.exists(savpath):
        #     print('found {}, continue'.format(savpath))
        #     continue

        # find the knot points
        x = [nparr(df['x']), nparr(df['y']),
             nparr(df['T']), nparr(df[magtype])]

        ndim = len(x)

        # to get the weight vector used in chi^2 estimation, estimate
        # uncertainty in x,y,T,mag as 1-sigma standard deviation from 6-hr
        # timescale window.
        sigma_vec = []
        for _x in x:
            windowsize = 12 # 6 hour timescale = 12 time points for FFIs.
            _sigma = pd.rolling_std(_x, windowsize)
            sigma_vec.append(np.ones_like(_x)*np.nanmean(_sigma))

        # Find B-spline representation of N-dimensional curve. Assumption is
        # that the list of time-series vectors represent a curve in
        # N-dimensional space parametrized by u, for u in [0,1]. We are trying
        # to find a smooth approximating spline curve g(u). This function wraps
        # the PARCUR routine from FITPACK, a FORTRAN library. Written by Paul
        # Dierckx.  PARCUR is for fitting of parametric open curves. (e.g.,
        # Dierckx, "Curve and surface fitting with splines", (1993, pg 111, sec
        # 6.3.1).  The method, and statement of the optimization problem, are
        # equations 6.46 and 6.47 of the above reference. 
        # The smoothing parameter s must be positive.

        s_grid = np.logspace(-3,-1,num=50)

        spld = {}
        BICs = []
        # For each value of the smoothing parameter, refit for the optimal
        # number of knots. Calculate chi^2 and BIC. Set the maximum number of
        # knots for a 13.7 day time-series (one orbit) to be 13, to prevent
        # fitting out <1 day frequencies unless they involve sharp features.
        # In a sense, this imposes a high-pass filter.
        for s in s_grid:
            n_knot_max = 13
            (tckp, u), metric, ier, msg = splprep(x, s=s, k=korder,
                                                  nest=n_knot_max, task=0,
                                                  full_output=1)

            # _t: knots of the spline curve; i.e. position of the knots in "u"
            # _c: coefficients in the b-spline representation of the spline
            # curve
            # _k: order
            _t,_c,_k = tckp
            coeffs, knots, order = tckp
            import IPython; IPython.embed() #FIXME

            n_dim = 4
            n_data = len(x[0])
            n_knots = np.atleast_1d(knots).shape[1]
            k_freeparam = n_dim*( n_knots + korder + 1 )

            # how do we compute chisq? answer: evaluate the spline. and then
            # compute it for what we're actually fitting: the flux, as a
            # function of temp, x position, and y position.

            x_pred, y_pred, T_pred, mag_pred = splev(u, tckp)
            pred_vec = x_pred, y_pred, T_pred, mag_pred

            mag = nparr(df[magtype])
            sigma_mag = nparr(df[magerrtype])
            chisq = 0
            for _x, _x_pred, _sigma_x in zip(x, pred_vec, sigma_vec):
                chisq += np.sum( ( (_x - _x_pred)/_sigma_x)**2 )

            BIC = chisq + k_freeparam * np.log(n_data)

            spld[s] = {}
            spld[s]['tckp'] = tckp
            spld[s]['n_knots'] = n_knots
            spld[s]['u'] = u
            spld[s]['metric'] = metric
            spld[s]['chisq'] = chisq
            spld[s]['n_data'] = n_data
            spld[s]['k_freeparam'] = k_freeparam
            spld[s]['BIC'] = BIC

            BICs.append(BIC)

        BICs = nparr(BICs)

        s_chosen = s_grid[np.argmin(BICs)]

        print('s\t\chisq\tn_data\tn_knots\tk_freeparam\tBIC')
        print(42*'-')
        for s in s_grid:
                print(s, spld[s]['chisq'], spld[s]['n_data'],
                      spld[s]['n_knots'], spld[s]['k_freeparam'],
                      spld[s]['BIC'])
        print(42*'-')

        spld_chosen = spld[s_chosen]
        tckp_best = spld_chosen['tckp']

        # evaluate spline, including interpolated points
        xnew, ynew, Tnew, magnew = splev(np.linspace(0,1,400), tckp_best)
        newd = {}
        newd['x'] = xnew
        newd['y'] = ynew
        newd['T'] = Tnew
        newd[magtype] = magnew

        newd_grid = {}
        for s in s_grid:
            newd_grid[s] = {}

            tckp = spld[s]['tckp']

            xnew, ynew, Tnew, magnew = splev(np.linspace(0,1,400), tckp)

            newd_grid[s]['x'] = xnew
            newd_grid[s]['y'] = ynew
            newd_grid[s]['T'] = Tnew
            newd_grid[s][magtype] = magnew

        # 3 by 3 triangle plot
        plt.close('all')

        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(6,6))
        axs = axs.flatten()

        noplotinds = nparr([2,3,6])-1
        plotinds = nparr([1,4,5,7,8,9])-1
        xvals=['x','x','y','x','y','T']
        yvals=['y','T','T',magtype,magtype,magtype]

        noxlabelinds = nparr([1,4,5])-1
        noylabelinds = nparr([5,8,9])-1

        for ind, xval, yval in zip(plotinds, xvals, yvals):
            ax = axs[ind]

            ax.scatter(df[xval], df[yval], rasterized=True, label='data',
                       alpha=0.8, zorder=5, c='k', lw=0, s=3)

            for s, sstr in zip([np.min(s_grid), np.max(s_grid), s_chosen],
                               ['min', 'max', 'best']):
                alpha = 0.3 if sstr != 'best' else 0.8
                ax.plot(
                    newd_grid[s][xval], newd_grid[s][yval],
                    label='{:s} s={:.1e}; got {:d} knots; $\chi^2$ = {:.1e}; {:d} points; BIC = {:.1e}'.
                    format(sstr, s, spld[s]['n_knots'], spld[s]['chisq'],
                           spld[s]['n_data'], spld[s]['BIC']),
                    lw=0.7, markersize=0, zorder=6, alpha=alpha
                )

            if ind==0:
                ax.legend(bbox_to_anchor=(0.95,0.95), loc='upper right',
                          bbox_transform=fig.transFigure, fontsize='xx-small',
                          framealpha=1)

            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')

            if ind not in noxlabelinds:
                ax.set_xlabel(xval, fontsize='xx-small')
            if ind not in noylabelinds:
                ax.set_ylabel(yval, fontsize='xx-small')
            if ind in noxlabelinds:
                ax.set_xticklabels([])
            if ind in noylabelinds:
                ax.set_yticklabels([])

        for ind in noplotinds:
            ax = axs[ind]
            ax.axis('off')

        fig.tight_layout(h_pad=-2, w_pad=-2)
        fig.savefig(savpath, dpi=400, bbox_inches='tight')
        print('made {}'.format(savpath))



if __name__=="__main__":

    np.random.seed(42)

    for lctype in ['corner','center']:

        savdir = '../results/projid1030_lc_fit_check/{}_lcs/'.format(lctype)

        lcpaths, lcdatalist =  homog_get_data(lctype=lctype)

        if len(glob(os.path.join(savdir, "*pairplot*.png"))) < 10:
            make_pairplots(lcpaths, lcdatalist, magtype='IRM1', frac_of_lc=1.0,
                           savdir=savdir)
            make_pairplots(lcpaths, lcdatalist, magtype='IRM1', frac_of_lc=0.4,
                           savdir=savdir)

        do_bspline_fit_xyTmag(lcpaths, lcdatalist, savdir=savdir)
