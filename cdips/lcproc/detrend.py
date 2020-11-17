"""
detrend.py - Luke Bouma (bouma.luke@gmail) - Jul 2019, Nov 2020

Contents:

Very useful:

    detrend_flux: a wrapper to wotan pspline or biweight detrending (given
        vectors of time and flux).

PCA / "shared systematic trend" detrending:

    detrend_systematics: removes systematic trends, by wrapping prepare_pca,
        get_dtrvecs, and calculate_linear_model_mag.

    prepare_pca: given TFA template stars, calculates PCA eigenvectors, and the
        "optimal" number to use according to a particular heuristic.

    get_dtrvecs: given PCA eigenvectors and a lightcurve, construct the vectors
        that will actually be used in decorrelation.

    calculate_linear_model_mag:  given a set of basis vectors in a linear model
        for a target light curve (y), calculate the coefficients and apply the
        linear model prediction.

Helper functions for the above:

    eigvec_smooth_fn: a wrapper to wotan biweight detrending, with hard tuning
        for eigenvector smoothing in PCA detrending.

    insert_nans_given_rstfc: NaN insertion for PCA prep.

    compute_scores: factor analysis and cross-validation PCA score helper.
"""
import matplotlib
matplotlib.use("AGG")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits

from datetime import datetime
import os, shutil
from glob import glob

from numpy import array as nparr, all as npall, isfinite as npisfinite

from astrobase import imageutils as iu
from astrobase.lcmath import find_lc_timegroups

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from wotan import flatten, version
wotanversion = version.WOTAN_VERSIONING
wotanversiontuple = tuple(wotanversion.split('.'))
assert int(wotanversiontuple[0]) >= 1
assert int(wotanversiontuple[1]) >= 4

def detrend_flux(time, flux, break_tolerance=0.5, method='pspline', cval=None,
                 window_length=None):

    # Initial pre-processing: verify that under break_tolerance, time and flux
    # do not have any sections with <=6 points. Spline detrending routines do
    # not like fitting lines.
    N_groups, group_inds = find_lc_timegroups(time, mingap=break_tolerance)
    SECTION_CUTOFF = 6
    for g in group_inds:
        if len(time[g]) <= SECTION_CUTOFF:
            time[g], flux[g] = np.nan, np.nan

    try:
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
                                            break_tolerance=break_tolerance,
                                            window_length=window_length,
                                            cval=cval)
        else:
            raise NotImplementedError

    except ValueError as e:
        msg = (
            'WRN! {}. Probably have a short segment. Trying to nan it out.'
            .format(repr(e))
        )
        print(msg)

        SECTION_CUTOFF = min([len(time[g]) for g in group_inds])
        for g in group_inds:
            if len(time[g]) <= SECTION_CUTOFF:
                time[g], flux[g] = np.nan, np.nan

        # NOTE: code duplication here
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
                                            break_tolerance=break_tolerance,
                                            window_length=window_length,
                                            cval=cval)
        else:
            raise NotImplementedError

    return flat_flux, trend_flux


def detrend_systematics(lcpath, max_n_comp=5):
    """
    Wraps functions in lcproc.detrend for all-in-one systematics removal, using
    a tuned variant of PCA.

    See doc/20201109_injectionrecovery_completeness_goldenvariability.txt for a
    verbose explanation of the options that were explored, and the assumptions
    that were ultimately made for this "tuned variant".

    Returns:
        Tuple of primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs
    """

    from cdips.utils import given_lcpath_get_infodict
    infodict = given_lcpath_get_infodict(lcpath)

    eigveclist, n_comp_df = prepare_pca(
        infodict['CAMERA'], infodict['CCD'],
        infodict['SECTOR'], infodict['PROJID']
    )

    sysvecnames = ['BGV']
    dtrvecs, sysvecs, ap, primaryhdr, data, eigenvecs, smooth_eigenvecs = (
        get_dtrvecs(lcpath, eigveclist, sysvecnames=sysvecnames)
    )
    time, y = data['TMID_BJD'], data[f'IRM{ap}']
    source_id = primaryhdr['OBJECT']

    n_components = min([int(n_comp_df[f'fa_cv_ap{ap}']), max_n_comp])
    n_components += len(sysvecnames)

    model_mag, n_comp = calculate_linear_model_mag(
        y, dtrvecs, n_components, method='LinearRegression'
    )

    data[f'PCA{ap}'] = model_mag
    primaryhdr[f'PCA{ap}NCMP'] = (
        n_comp,
        f'N principal components PCA{ap}'
    )

    return primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs


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

            eigveclist = [eigenvecs_ap1, eigenvecs_ap2, eigenvecs_ap3] where
            each element is a np.ndarray.

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


def get_dtrvecs(lcpath, eigveclist, sysvecnames=['BGV'],
                use_smootheigvecs=True):
    """
    Given a CDIPS light curve file, and the PCA eigenvectors for this
    sector/cam/ccd, construct the vectors to "detrend" or "decorrelate" against
    (presumably via a linear model).

    Args:
        lcpath: CDIPS light curve file

        eigveclist: list of np.ndarray PCA eigenvectors, length 3, calculated
        by a call to cdips.lcutils.detrend.prepare_pca.

        sysvecnames: list of vector names to also be decorrelated against.
        E.g., ['BGV', 'CCDTEMP', 'XIC', 'YIC']. Default is just ['BGV'].

        use_smootheigvecs: whether or not to smooth the PCA eigenvectors, using
        a windowed biweight filter.

    Returns:
        tuple containing:
        dtrvecs (np.ndarray), sysvecs (np.ndarray), ap (the optimal aperture),
        data (the entire lcpath data table), eigenvecs (np.ndarray chosen from
        eigveclist given the optimal aperture), smooth_eigenvecs (np.ndarray).
    """

    from cdips.utils.lcutils import get_best_ap_number_given_lcpath

    hdul = fits.open(lcpath)
    primaryhdr, hdr, data = (
        hdul[0].header, hdul[1].header, hdul[1].data
    )
    hdul.close()
    ap = min([get_best_ap_number_given_lcpath(lcpath), 2])

    ##########################################
    # begin PCA.
    #
    # eigenvecs shape: N_templates x N_times
    #
    # model: 
    # y(w, x) = w_0 + w_1 x_1 + ... + w_p x_p.
    #
    # X is matrix of (n_samples, n_features), so each template is a "sample",
    # and each time is a "feature". Analogous to e.g., doing PCA to reconstruct
    # a spectrum, and you want each wavelength bin to be a feature.
    ##########################################
    eigenvecs = eigveclist[ap-1]

    if np.any(pd.isnull(eigenvecs)):
        raise ValueError('got nans in eigvecs. bad!')

    if use_smootheigvecs:
        smooth_eigenvecs = []
        for e in eigenvecs:
            smooth_eigenvec = eigvec_smooth_fn(data['TMID_BJD'], e)
            smooth_eigenvecs.append(smooth_eigenvec-1)
        smooth_eigenvecs = np.array(smooth_eigenvecs)
        assert not np.any(pd.isnull(smooth_eigenvecs))
    else:
        smooth_eigenvecs = None

    use_sysvecs = True if isinstance(sysvecnames, list) else False

    if use_sysvecs:

        # Use the (0-1 scaled) systematic vectors directly. Don't smooth.
        sysvecs = np.vstack(
            [
                MinMaxScaler().fit_transform(
                    data[s][:,None].astype(np.float64)
                ).flatten()
                -
                np.nanmedian(MinMaxScaler().fit_transform(
                    data[s][:,None].astype(np.float64)
                ).flatten())
                for s in sysvecnames
            ]
        )

        if use_smootheigvecs:
            dtrvecs = np.vstack([sysvecs, smooth_eigenvecs])
        else:
            dtrvecs = np.vstack([sysvecs, eigenvecs])

    else:

        sysvecs = None

        if use_smootheigvecs:
            dtrvecs = smooth_eigenvecs
        else:
            dtrvecs = eigenvecs

    return dtrvecs, sysvecs, ap, primaryhdr, data, eigenvecs, smooth_eigenvecs


def eigvec_smooth_fn(time, eigenvec):

    _, smoothed = flatten(
        time-np.nanmin(time), 1+eigenvec,
        break_tolerance=0.5,
        method='biweight', window_length=1, cval=6, edge_cutoff=0,
        return_trend=True
    )

    return smoothed


def calculate_linear_model_mag(y, basisvecs, n_components,
                               method='LinearRegression', verbose=False):
    """
    Given a set of basis vectors in a linear model for a target light curve
    (y), calculate the coefficients and apply the linear model prediction.

    Allow methods:

        'LinearRegression': ordinary least squares.

        'RidgeCV': ridge regression, which is ordinary least squares, plus an
        L2 norm with a regularization coefficient. The regularization is solved
        for via a cross-validation grid search.

    Returns:
        out_mag, n_comp: the model light curve (supposedly with instrumental
        systematic removed), and the number of PCA components used during the
        regression.
    """

    if method == 'LinearRegression':
        reg = LinearRegression(fit_intercept=True)
    elif method == 'RidgeCV':
        reg = RidgeCV(alphas=np.logspace(-10, 10, 210), fit_intercept=True)

    if np.all(pd.isnull(y)):
        # if all nan, job is easy
        out_mag = y
        n_comp = 'nan'

    elif np.any(pd.isnull(y)):

        #
        # if some nan in target light curve, tricky. strategy: impose the
        # same nans onto the eigenvectors. then fit without nans. then put
        # nans back in.
        #
        mean_mag = np.nanmean(y)
        std_mag = np.nanstd(y)
        norm_mag = (y[~pd.isnull(y)] - mean_mag)/std_mag

        _X = basisvecs[:n_components, :]

        _X = _X[:, ~pd.isnull(y)]

        reg.fit(_X.T, norm_mag)

        # see note below in the else statement about how the prediction step
        # works.
        model_mag = mean_mag + std_mag*reg.predict(_X.T)

        #
        # now insert nans where they were in original target light curve.
        # typically nans occur in groups. insert by groups. this is a
        # tricky procedure, so test after to ensure nan indice in the model
        # and target light curves are the same.
        #
        naninds = np.argwhere(pd.isnull(y)).flatten()

        ngroups, groups = find_lc_timegroups(naninds, mingap=1)

        for group in groups:

            thisgroupnaninds = naninds[group]

            model_mag = np.insert(model_mag,
                                  np.min(thisgroupnaninds),
                                  [np.nan]*len(thisgroupnaninds))

        np.testing.assert_(
            model_mag.shape == y.shape
        )
        np.testing.assert_(
            (len((model_mag-y)[pd.isnull(model_mag-y)])
             ==
             len(y[pd.isnull(y)])
            ),
            'got different nan indices in model mag than target mag'
        )

        #
        # save result as IRM mag - PCA model mag + mean IRM mag
        #
        out_mag = y - model_mag + mean_mag
        n_comp = n_components

    else:
        # otherwise, just directly fit the principal components
        mean_mag = np.nanmean(y)
        std_mag = np.nanstd(y)
        norm_mag = (y - mean_mag)/std_mag

        _X = basisvecs[:n_components, :]

        reg.fit(_X.T, norm_mag)

        # NOTE: in ordinary least squares, the line below is the same as
        # `model_mag = reg.intercept_ + (reg.coef_ @ _X)`. But the sklearn
        # interface is sick, and using reg.predict is easier than doing the
        # extra terms for other regressors.
        model_mag = mean_mag + std_mag * reg.predict(_X.T)

        out_mag = y - model_mag + mean_mag
        n_comp = n_components

    if verbose:
        if method == 'RidgeCV':
            try:
                print(f'RidgeCV alpha: {reg.alpha_:.2e}')
            except AttributeError:
                pass

    return out_mag, n_comp
