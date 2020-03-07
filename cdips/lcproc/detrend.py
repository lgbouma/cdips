import matplotlib
matplotlib.use("AGG")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits

from datetime import datetime
import os, shutil
from glob import glob

from numpy import array as nparr, all as npall, isfinite as npisfinite

from astrobase import imageutils as iu

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import cross_val_score

from wotan import flatten, version
wotanversion = version.WOTAN_VERSIONING
wotanversiontuple = tuple(wotanversion.split('.'))
assert int(wotanversiontuple[0]) >= 1
assert int(wotanversiontuple[1]) >= 4


def detrend_flux(time, flux, break_tolerance=0.5, method='pspline'):

    if method == 'pspline':
        # matched detrending to do_initial_period_finding
        flat_flux, trend_flux = flatten(time, flux,
                                        method='pspline',
                                        return_trend=True,
                                        break_tolerance=break_tolerance,
                                        robust=True)

    elif method == 'biweight':
        # another option:
        flat_flux, trend_flux = flatten(time, flux,
                                        method='biweight',
                                        return_trend=True,
                                        break_tolerance=0.5,
                                        window_length=0.3,
                                        cval=6)

    else:
        raise NotImplementedError

    return flat_flux, trend_flux


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


def insert_nans_given_rstfc(mag, mag_rstfc, full_rstfc):
    """
    args:

        mag: the vector of magnitudes that needs nans inserted

        mag_rstfc: RSTFC frame id vector of same length as mag

        full_rstfc: RSTFC frame id vector of larger length, that will be
        matched against.

    returns:

        full_mag: vector of magnitudes with nans inserted, of same length as
        full_rstfc
    """

    if not len(full_rstfc) >= len(mag_rstfc):

        raise AssertionError('full_rstfc needs to be the big one')

    if len(full_rstfc) == len(mag_rstfc):

        return mag

    else:
        # input LC has too few frame stamps (for whatever reason -- often
        # because NaNs are treated differently by VARTOOLS TFA or other
        # detrenders, so they are omitted). The following code finds entries in
        # the TFA lightcurve that are missing frameids, and put nans in those
        # cases. It does this by first making an array of NaNs with length
        # equal to the original RAW data. It then puts the magnitude values at
        # the appropriate indices, through the frame-id matching ("RSTFC" ids).

        inarr = np.in1d(full_rstfc, mag_rstfc)

        inds_to_put = np.argwhere(inarr).flatten()

        full_mag = (
            np.ones_like(np.arange(len(full_rstfc),
                                   dtype=np.float))*np.nan
        )

        full_mag[ inds_to_put ] = mag

        wrn_msg = (
            '{} WRN!: found missing frameids. added NaNs'.
            format(datetime.utcnow().isoformat())
        )
        print(wrn_msg)

        return full_mag


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
        # prepare data as a (N_template_stars x N_times) matrix. We have N~=200
        # template light curves, with K=N_times measurements in each. Think of
        # these as N vectors in a K-dimensional space. I.e. the flux at each
        # time point is a "measured feature". So we have N samples (light
        # curves), and K features (points per light curve).
        #

        mags = nparr(
            list(
                map(iu.get_data_keyword,
                    nparr(df_template_stars['path']), # file,
                    np.repeat('IRM{}'.format(ap), len(df_template_stars)), # keyword
                    np.repeat(1, len(df_template_stars)) # extension
                   )
            )
        )
        mag_rstfc = nparr(
            list(
                map(iu.get_data_keyword,
                    nparr(df_template_stars['path']), # file,
                    np.repeat('RSTFC', len(df_template_stars)), # keyword
                    np.repeat(1, len(df_template_stars)) # extension
                   )
            )
        )

        #
        # for the fit, require that for each tempalte light curve is only made
        # of finite values. this might drop a row or two.
        #
        fmags = mags[~np.isnan(mags).any(axis=1)]

        #
        # subtract mean, as is standard in PCA.
        #
        mean_mags = np.nanmean(fmags, axis=1)
        X = fmags - mean_mags[:, None]

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
            this_lcpath = np.random.choice(lcpaths, size=1)[0]
            mag = iu.get_data_keyword(this_lcpath, 'IRM{}'.format(ap))
            mag_rstfc = iu.get_data_keyword(this_lcpath, 'RSTFC')

            if np.all(pd.isnull(mag)):
                while np.all(pd.isnull(mag)):
                    next_lcpath = np.random.choice(lcpaths, size=1)[0]
                    mag = iu.get_data_keyword(next_lcpath, 'IRM{}'.format(ap))
                    mag_rstfc = iu.get_data_keyword(next_lcpath, 'RSTFC')

            mean_mag = np.nanmean(mag)
            mag = mag - mean_mag
            mag = mag[~pd.isnull(mag)]

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

                #FIXME FIXME. probably have a shape related bug?!?!?!?!?!!?!?!?!?
                # either linear regression or bayesian ridge regression seems fine
                reg = LinearRegression(fit_intercept=True)
                #reg = BayesianRidge(fit_intercept=True)

                #FIXME : you need to simultaneously cast the eigenvectors and
                # magnitudes to the same shape of nans?!

                y = mag
                _X = eigenvecs[:n_components, :]

                try:
                    reg.fit(_X.T, y)
                except Exception as e:
                    print(e)
                    print(n_components)
                    continue
                    #import IPython; IPython.embed() #FIXME error is here.

                model_mag = reg.intercept_ + (reg.coef_ @ _X)

                # given "true" (full) RSTFC list, and the actual list
                # ("rstfc" above), need a function that gives model mags
                # (or data mags) with nans in correct place

                time = nparr(df_dates['btjd'])
                full_rstfc = nparr(df_dates['rstfc'])

                full_mag = insert_nans_given_rstfc(mag, mag_rstfc, full_rstfc)

                full_model_mag = insert_nans_given_rstfc(
                    model_mag, mag_rstfc, full_rstfc
                )

                ax.scatter(time, full_mag + mean_mag, c='k', alpha=0.9,
                           zorder=2, s=1, rasterized=True, linewidths=0)
                ax.plot(time, full_model_mag + mean_mag, c='C0', zorder=1,
                        rasterized=True, lw=0.5, alpha=0.7 )

                txt = '{} components'.format(n_components)
                ax.text(0.02, 0.02, txt, ha='left', va='bottom',
                        fontsize='medium', transform=ax.transAxes)

                ax_r.scatter(time, full_mag-full_model_mag, c='k', alpha=0.9, zorder=2,
                             s=1, rasterized=True, linewidths=0)
                ax_r.plot(time, full_model_mag-full_model_mag, c='C0', zorder=1,
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
