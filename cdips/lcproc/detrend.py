from wotan import flatten
import os, shutil
from glob import glob

import matplotlib
matplotlib.use("AGG")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits

from numpy import array as nparr, all as npall, isfinite as npisfinite

import imageutils as iu

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import cross_val_score

def detrend_flux(time, flux):

    # matched detrending to do_initial_period_finding

    break_tolerance = 0.5
    flat_flux, trend_flux = flatten(time, flux,
                                    method='pspline',
                                    return_trend=True,
                                    break_tolerance=break_tolerance)

    return flat_flux, trend_flux


def _get_mag(fitspath, ap):
    with fits.open(fitspath) as hdulist:
        mag = hdulist[1].data[ap]
    return mag

def compute_scores(X, n_components):
    pca = PCA()
    fa = FactorAnalysis()
    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
        fa.n_components = n
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=5)))

    return pca_scores, fa_scores


def prepare_pca(cam, ccd, sector, projid, N_to_make=20):
    """
    This function:

        * calculates light curve principal components using TFA template
        stars given in "trendlist_tfa_ap[1-3].txt"

        * writes each set to /statsdir/pca_data/principal_component_ap[1-3].txt

        * calculates the optimal number of principal components to use based on
        a factor analysis cross-validation.

        * writes the optimal number, for each aperture, to
        /statsdir/pca_data/optimal_n_components.csv

        * makes plots showing the effect of using different numbers of trend
        stars for N_to_make random stars from the cam/ccd/sector/projid
        combination.

    Args:

        cam,ccd,sector,projid: ints

        N_to_make: integer number of plots showing the effects of adding more
        principal components to the fit.

    Returns:

        (eigveclist, optimal_n_comp_df):

            eigveclist = [eigenvecs_ap1, eigenvecs_ap2, eigenvecs_ap3] where is
            a np.ndarray.

            optimal_n_comp_df: dataframe.

    Using them, do linear least squares (or a variant thereof) to get
    components matched to each LC.
    """

    lcdir = ('/nfs/phtess2/ar0/TESS/FFI/LC/FULL/s{}/ISP_{}-{}-{}/'.
             format(str(sector).zfill(4), cam, ccd, projid))

    statsdir = os.path.join(lcdir, 'stats_files')

    pcadir = os.path.join(statsdir, 'pca_data')
    if not os.path.exists(pcadir):
        os.mkdir(pcadir)

    csvpath = os.path.join(pcadir, 'optimal_n_components.csv')
    if os.path.exists(csvpath):

        comppaths = [os.path.join(pcadir,
                                  'principal_component_ap{}.txt'.format(ap))
                     for ap in range(1,4)]

        eigveclist = [np.genfromtxt(f) for f in comppaths]

        n_comp_df = pd.read_csv(csvpath)

        return eigveclist, n_comp_df

    #
    # path, x, y. space-separated. for all ~30k light curves to do TFA on.
    #
    tfalclist_path = os.path.join(statsdir,'lc_list_tfa.txt')
    #
    # key, time
    #
    datestfa_path = os.path.join(statsdir,'dates_tfa.txt')

    eigveclist, optimal_n_comp = [], {}
    for ap in [1,2,3]:

        #
        # path, x ,y. space-separated. for 200 TFA template stars.
        #
        trendname = 'trendlist_tfa_ap{}.txt'.format(ap)
        trendlisttfa = os.path.join(statsdir,trendname)

        df_template_stars = pd.read_csv(trendlisttfa, sep=' ', header=None,
                                        names=['path','x','y'])

        df_dates = pd.read_csv(datestfa_path, sep=' ', header=None,
                               names=['rstfc','btjd'])

        lcpaths = glob(os.path.join(lcdir, '*_llc.fits'))

        #
        # prepare data as a (N_template_stars x N_times) matrix. We have N=200
        # template light curves, with K=N_times measurements in each. Think of
        # these as N vectors in a K-dimensional space. I.e. the flux at each
        # time point is a "measured feature". So we have N samples (light
        # curves), and K features (points per light curve).
        #

        mags = nparr(
            list(
                map(_get_mag,
                    nparr(df_template_stars['path']),
                    np.repeat('IRM{}'.format(ap), len(df_template_stars))
                   )
            )
        )

        #
        # subtract mean, as is standard in PCA.
        #
        mean_mags = np.nanmean(mags, axis=1)
        X = mags - mean_mags[:, None]

        pca = PCA()
        pca.fit(X)

        #
        # (200 x N_times) eigenvector matrix. these are basis vectors for the
        # original data. can use factor analysis components instead, without
        # much differnece.
        #

        eigenvecs = pca.components_

        comppath = os.path.join(pcadir,
                                'principal_component_ap{}.txt'.format(ap))
        np.savetxt(comppath, eigenvecs)
        print('saved {}'.format(comppath))

        eigveclist.append(eigenvecs)

        #
        # plot a sequence of reconstructions, for a set of random light curves
        #
        for i in range(N_to_make):

            if ap != 2:
                continue
            savpath = os.path.join(pcadir,
                                   'test_reconstruction_{}_ap{}.png'.
                                   format(i, ap))
            if os.path.exists(savpath):
                continue

            np.random.seed(i)
            mag = _get_mag(np.random.choice(lcpaths, size=1)[0],
                           'IRM{}'.format(ap))
            if np.all(pd.isnull(mag)):
                while np.all(pd.isnull(mag)):
                    mag = _get_mag(np.random.choice(lcpaths, size=1)[0],
                                   'IRM{}'.format(ap))
            mean_mag = np.nanmean(mag)
            mag = mag - mean_mag

            component_list = [1, 2, 4, 8, 12, 16, 20]

            plt.close('all')
            f, axs = plt.subplots(nrows=len(component_list), ncols=2,
                                  sharex=True, figsize=(8,9))

            for n_components, ax, ax_r in zip(component_list, axs[:,0], axs[:,1]):
                #
                # eigenvecs shape: 200 x N_times
                #
                # model: 
                # y(w, x) = w_0 + w_1 x_1 + ... + w_p x_p.
                #
                # X is matrix of (n_samples, n_features).
                #

                # either linear regression or bayesian ridge regression seems fine
                reg = LinearRegression(fit_intercept=True)
                #reg = BayesianRidge(fit_intercept=True)

                y = mag
                _X = eigenvecs[:n_components, :]

                reg.fit(_X.T, y)

                model_mag = reg.intercept_ + (reg.coef_ @ _X)

                time = nparr(df_dates['btjd'])

                ax.scatter(time, mag + mean_mag, c='k', alpha=0.9,
                           zorder=2, s=1, rasterized=True, linewidths=0)
                ax.plot(time, model_mag + mean_mag, c='C0', zorder=1,
                        rasterized=True, lw=0.5, alpha=0.7 )

                txt = '{} components'.format(n_components)
                ax.text(0.02, 0.02, txt, ha='left', va='bottom',
                        fontsize='medium', transform=ax.transAxes)

                ax_r.scatter(time, mag-model_mag, c='k', alpha=0.9, zorder=2,
                             s=1, rasterized=True, linewidths=0)
                ax_r.plot(time, model_mag-model_mag, c='C0', zorder=1,
                          rasterized=True, lw=0.5, alpha=0.7)

            for a in axs[:,0]:
                a.set_ylabel('raw mag')
            for a in axs[:,1]:
                a.set_ylabel('resid')
            for a in axs.flatten():
                a.get_yaxis().set_tick_params(which='both', direction='in')
                a.get_xaxis().set_tick_params(which='both', direction='in')

            f.text(0.5,-0.01, 'BJDTDB [days]', ha='center')

            f.tight_layout(h_pad=0, w_pad=0.5)
            f.savefig(savpath, dpi=350, bbox_inches='tight')
            print('made {}'.format(savpath))

        #
        # make plot to find optimal number of components. write output to...
        # /statsdir/pca_data/optimal_n_components.csv
        # see
        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
        #
        n_components = np.arange(0,50,1)

        pca_scores, fa_scores = compute_scores(X, n_components)

        plt.close('all')
        f,ax = plt.subplots(figsize=(4,3))

        ax.plot(n_components, pca_scores, label='PCA CV score')
        ax.plot(n_components, fa_scores, label='FA CV score')

        ax.legend(loc='best')
        ax.set_xlabel('N components')
        ax.set_ylabel('Cross-validation score')

        f.tight_layout(pad=0.2)
        savpath = os.path.join(pcadir, 'test_optimal_n_components.png')
        f.savefig(savpath, dpi=350, bbox_inches='tight')
        print('made {}'.format(savpath))

        #
        # write the factor analysis maximum to a dataframe
        #
        n_components_pca_cv = n_components[np.argmax(pca_scores)]
        n_components_fa_cv = n_components[np.argmax(fa_scores)]

        print('n_components_pca_cv: {}'.format(n_components_pca_cv))
        print('n_components_fa_cv: {}'.format(n_components_fa_cv))

        optimal_n_comp['pca_cv_ap{}'.format(ap)] = n_components_pca_cv
        optimal_n_comp['fa_cv_ap{}'.format(ap)] = n_components_fa_cv

    optimal_n_comp_df = pd.DataFrame(optimal_n_comp, index=[0])
    optimal_n_comp_df.to_csv(csvpath, index=False)
    print('made {}'.format(csvpath))

    return eigveclist, optimal_n_comp_df
