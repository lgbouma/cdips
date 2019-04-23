"""
scatter plots of magnitudes vs various parameters... to see whether the EPD
linear fitting approach works.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from glob import glob
import os

from astropy.io import fits

from plot_mag_vs_EPD_parameters import get_data

import seaborn as sns

from scipy.interpolate import splprep, splev

from numpy import array as nparr

def make_pairplots(lcpaths, lcdatalist, magtype='IRM1', frac_of_lc=0.4):
    # make pairplots of X,Y,T,MAG,S,D,K, and just X,Y,T,MAG
    # frac_of_lc: for fitting purposes, we want orbit-specific data.

    xcols = ['FSV','FDV','FKV','XIC','YIC','CCDTEMP',magtype]
    xkeys = ['s','d','k','x','y','T',magtype]

    for lcpath, lc in zip(lcpaths, lcdatalist):

        savdir = '../results/bspline_fit_xyTmag/'
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
                          isffi=True):

    if not isffi:
        raise AssertionError('windowsize for polling std assumes is ffi')

    # spline parameters
    s_initial = 1.0 # smoothness parameter
    korder = 3 # spline order
    nest = -1 # estimate of number of knots needed (-1 = maximal)

    # NOTE: here "s" is a hyperparameter. We want to tune it, by
    # cross-validation. (Not BIC/chi-squared, which doesn't seem to be
    # well-defined here.)

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

        savdir = '../results/bspline_fit_xyTmag/'
        savname = '{}_frac{:.1f}_scatterfit_xval_xyTmag_{}.png'.format(
            magtype, frac_of_lc, os.path.splitext(os.path.basename(lcpath))[0])
        savpath = os.path.join(savdir, savname)

        if os.path.exists(savpath):
            print('found {}, continue'.format(savpath))
            continue

        # find the knot points
        vec_x_list = [nparr(df['x']), nparr(df['y']), nparr(df['T']),
                 nparr(df[magtype])]

        ndim = len(vec_x_list)

        # NOTE omitting any weight vector (seems to be OK).
        # # to get the weight vector, estimate uncertainty in x,y,T,mag as
        # # 1-sigma standard deviation from 6-hr timescale window. Then weights =
        # # 1/standard deviation.
        # w = []
        # for _x in x:
        #     windowsize = 12 # 6 hour timescale = 12 time points for FFIs.
        #     _w = pd.rolling_std(_x, windowsize)
        #     w.append(1/(np.ones_like(_x)*np.nanmean(_w)))

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

        s_grid = np.logspace(-3,-1,num=5)
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score, explained_variance_score

        spld = {}
        r_sq_means, expl_var_means, tckp_d = [], [], {}
        for s in s_grid[::-1]: #FIXME

            # split data into k subsets
            n_splits = 4 # = k

            X = np.atleast_2d(vec_x_list).T

            n_data = X.shape[0]

            kf = KFold(n_splits=n_splits)

            r_sq_list, expl_var_list = [], []
            for train_index, test_index in kf.split(X):

                X_train, X_test = X[train_index], X[test_index]

                # supposedly necessary transformation for splprep to run
                X_train_list = [ X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3] ]
                X_test_list = [ X_test[:,0],X_test[:,1],X_test[:,2],X_test[:,3] ]

                mag_true = X_test[:,3]

                (tckp_train, u_train), metric, ier, msg = splprep(X_train_list,
                                                                  s=s,
                                                                  k=korder,
                                                                  nest=-1,
                                                                  task=0,
                                                                  full_output=1)

                # tckp[0]: knots as a function of u
                # tckp[1]: knots as a function of (x,y,T,mag)
                # tckp[2]: order
                train_knots = tckp_train[0]

                # get coefficients for the TEST data, using the set of knots
                # determined for the TRAINING data.
                import IPython; IPython.embed() #FIXME: does this work?
                (tckp_test, u_test), metric_test, ier_test, msg_test = (
                    splprep(X_test_list, s=s, k=korder, t=train_knots, task=-1,
                            full_output=1)
                )

                # tckp contains information about the knots. answers: what are the
                # spline fit coefficients? _where_ are the knots? (doesn't need to
                # be x,y,T,mag). what order is the spline?
                x_pred, y_pred, T_pred, mag_pred = splev(u_test, tckp_test)

                r_sq_list.append(r2_score(mag_true, mag_pred))
                expl_var_list.append(explained_variance_score(mag_true, mag_pred))

                import IPython; IPython.embed() #FIXME: does this work?

            r_sq_means.append( np.mean(r_sq_list) )
            expl_var_means.append( np.mean(expl_var_list) )

            # SEPARATELY, get the spline fit coefficient for the entire
            # dataset. (using whatever knots are found to be best. this is the
            # same number, since it's the same s as above).
            (tckp_full, u_full), _, _, _ = (
                splprep(vec_x_list, s=s, k=korder, nest=-1, task=0,
                        full_output=1)
            )
            tckp_d[s] = {}
            tckp_d[s]['tckp_full'] = tckp_full
            tckp_d[s]['u_full'] = u_full

            tckp_d[s]['xval_rsq'] = np.mean(r_sq_list)
            tckp_d[s]['xval_explvar'] = np.mean(expl_var_list)

        r_sq_means = nparr(r_sq_means)
        expl_var_means = nparr(expl_var_means)

        best_s_from_r_sq = s_grid[ np.argmax(r_sq_means) ]
        best_s_from_explvar = s_grid[ np.argmax(expl_var_means) ]

        newd_grid = {}
        for s in s_grid:
            newd_grid[s] = {}

            tckp = spld[s]['tckp_full']

            # evaluate b-spline along the full interval u=[0,1]. use the knots
            # and coefficients from the b-spline fits.
            xnew, ynew, Tnew, magnew = splev(np.linspace(0,1,400), tckp)

            newd_grid[s]['x'] = xnew
            newd_grid[s]['y'] = ynew
            newd_grid[s]['T'] = Tnew
            newd_grid[s][magtype] = magnew

            newd_grid[s]['n_knots'] = len(tckp[1])
            newd_grid[s]['n_data'] = len(nparr(df['x']))

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

            for s in s_grid:

                labelstr = ('s={:.1e}; got {:d} knots; {:d} points; '
                           'xval $R^2$ {:.2f}; xval explvar {:.2f}' )
                label = labelstr.format(s, spld[s]['n_knots'],
                                        spld[s]['n_data'],
                                        tckp_d[s]['xval_rsq'],
                                        tckp_d[s]['xval_explvar'])

                ax.plot(newd_grid[s][xval], newd_grid[s][yval], label=label,
                        lw=1, markersize=0, zorder=6, alpha=0.6)

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

    n_lcs = 10

    lcpaths, lcdatalist = get_data(n_lcs=n_lcs)

    make_pairplots(lcpaths, lcdatalist, magtype='IRM1', frac_of_lc=1.0)
    make_pairplots(lcpaths, lcdatalist, magtype='IRM1', frac_of_lc=0.4)

    do_bspline_fit_xyTmag(lcpaths, lcdatalist)
