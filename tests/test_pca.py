"""
test_pca.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Exploration used to check different PCA variants (ridge regression vs ordinary
least squares; to smooth or not smooth the eigenvectors; what supplementary
"systematic" vectors to include).  The outcomes of these tests are documented
at /doc/20201109_injectionrecovery_completeness_goldenvariability.txt

Wrapper level:

    explore_pca(sector=9, cam=2, ccd=3)

    explore_ic2602()

Plot level:

    run_explore_pca(lcpath, eigveclist, n_comp_df):
"""

import os
from glob import glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import MinMaxScaler

from wotan import flatten, version

from cdips.lcproc import (
    reformat_lcs_for_mast as rlm,
    mask_orbit_edges as moe,
    detrend as dtr
)
from cdips.utils.lcutils import (
    find_cdips_lc_paths, get_lc_data, get_best_ap_number_given_lcpath
)

import astrobase.imageutils as iu
from astrobase.lcmath import find_lc_timegroups

def rescale_ipr(vec, target_ipr):
    current_ipr = np.nanpercentile(vec, 95) - np.nanpercentile(vec, 5)
    factor = target_ipr / current_ipr
    return vec*factor

def scale_to_unitvariance(vec):
    return vec/np.nanstd(vec)

def center_and_unitvariance(vec):
    return (vec-np.nanmean(vec))/(np.nanstd(vec))


def explore_pca(
    sector=9,
    cam=2,
    ccd=3,
    symlinkdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS/',
    outdir='/home/lbouma/proj/cdips/tests/test_pca_plots/',
    OC_MG_CAT_ver=0.4,
    cdipsvnum=1,
    max_n_test_lcs=100
):

    lcpaths = glob(os.path.join(
        symlinkdir, f'sector-{sector}', f'cam{cam}_ccd{ccd}', '*_llc.fits')
    )
    np.random.seed(42)
    lcpaths = np.random.choice(lcpaths, max_n_test_lcs, replace=False)

    projid = iu.get_header_keyword(lcpaths[0], 'PROJID')

    eigveclist, n_comp_df = dtr.prepare_pca(cam, ccd, sector, projid)

    outdir = os.path.join(outdir, f's{sector}_c{cam}_c{ccd}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    global OUTDIR
    OUTDIR = outdir

    for lcpath in lcpaths:
        run_explore_pca(lcpath, eigveclist, n_comp_df, use_sysvecs=True)


def calculate_pca_model_mag(y, eigenvecs, n_components,
                            method='LinearRegression', verbose=False):
    """
    Given the PCA eigenvectors (eigenvecs) as basis vectors in a linear model
    for the target light curve (y), calculate the coefficients and apply the
    linear model prediction.

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

        _X = eigenvecs[:n_components, :]

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

        _X = eigenvecs[:n_components, :]

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


def smooth_fn(time, eigenvec):
    _, smoothed = flatten(
        time-np.nanmin(time), 1+eigenvec,
        break_tolerance=0.5,
        method='biweight', window_length=1, cval=6, edge_cutoff=0,
        return_trend=True
    )
    return smoothed


def get_dtrvecs(lcpath, eigveclist, sysvecnames=['BGV'],
                use_smootheigvecs=True):
    """
    Given a CDIPS light curve file, and the PCA eigenvectors for this
    sector/cam/ccd, get the vectors to detrend against.

    Args:
        lcpath: CDIPS light curve file

        eigveclist: list of np.ndarray PCA eigenvectors, length 3, calculated
        by a call to cdips.lcutils.detrend.prepare_pca.

        sysvecnames: list of vector names to also be decorrelated against.
        E.g., ['BGV', 'CCDTEMP', 'XIC', 'YIC']. Default is just ['BGV'].

        use_smootheigvecs: whether or not to smooth the PCA eigenvectors.

    Returns:
        dtrvecs
    """

    hdul = fits.open(lcpath)
    primaryhdr, hdr, data = (
        hdul[0].header, hdul[1].header, hdul[1].data
    )
    hdul.close()
    source_id = primaryhdr['OBJECT']
    ap = min([get_best_ap_number_given_lcpath(lcpath), 2])

    ##########################################
    # begin PCA.
    #
    # first, get ensemble pca magnitude vectors. these will be appended.
    # (in an annoying order. so be it).
    #
    # eigenvecs shape: N_templates x N_times
    #
    # model: 
    # y(w, x) = w_0 + w_1 x_1 + ... + w_p x_p.
    #
    # X is matrix of (n_samples, n_features), so each template is a "sample",
    # and each time is a "feature". Analogous to e.g., doing PCA to reconstruct
    # a spectrum, and you want each wavelength bin to be a feature.
    #
    #
    pca_mags = {}
    scales = {}

    eigenvecs = eigveclist[ap-1]

    if use_smootheigvecs:
        smooth_eigenvecs = []
        for e in eigenvecs:
            smooth_eigenvec = smooth_fn(data['TMID_BJD'], e)
            smooth_eigenvecs.append(smooth_eigenvec-1)
        smooth_eigenvecs = np.array(smooth_eigenvecs)
        assert not np.any(pd.isnull(smooth_eigenvecs))

    if np.any(pd.isnull(eigenvecs)):
        raise ValueError('got nans in eigvecs. bad!')

    y = data[f'IRM{ap}']

    use_sysvecs = True if isinstance(sysvecnames, list) else False

    if use_sysvecs:

        # Use the (0-1 scaled) systematic vectors directly. Don't smooth.
        sysvecs = np.vstack(
            [
                MinMaxScaler().fit_transform(
                    data[s][:,None].astype(np.float64)
                ).flatten()
                for s in sysvecnames
            ]
        )

        if use_smootheigvecs:
            dtrvecs = np.vstack([sysvecs, smooth_eigenvecs])
        else:
            dtrvecs = np.vstack([sysvecs, eigenvecs])

    else:

        if use_smootheigvecs:
            dtrvecs = smooth_eigenvecs
        else:
            dtrvecs = eigenvecs

    return dtrvecs, y, ap, data['TMID_BJD'], data, eigenvecs, smooth_eigenvecs



def run_explore_pca(lcpath, eigveclist, n_comp_df, use_sysvecs=False,
                    use_smootheigvecs=True):
    """
    use_sysvecs:
        whether or not to include extra vectors in the regression.
    """

    sysvecnames = ['BGV']
    dtrvecs, y, ap, time, data, eigenvecs, smooth_eigenvecs = (
        get_dtrvecs(lcpath, eigveclist, sysvecnames=sysvecnames)
    )

    ##########################################
    # NOTE: below will be ported
    n_components = min([int(n_comp_df[f'fa_cv_ap{ap}']), 5])
    if use_sysvecs:
        n_components += len(sysvecnames)
    model_mag, n_comp = calculate_pca_model_mag(
        y, dtrvecs, n_components, method='LinearRegression'
    )

    pca_mags[f'PCA{ap}'] = model_mag
    primaryhdr[f'PCA{ap}NCMP'] = (
        n_comp,
        f'N principal components PCA{ap}'
    )
    # TODO: once the above is tested, port it to
    #lcproc.reformat_lcs_for_mast, because it's much cleaner
    ##########################################

    model_mag_ridge, _ = calculate_pca_model_mag(
        y, dtrvecs, n_components, method='RidgeCV',
        verbose=True
    )

    model_mag_sys, _ = calculate_pca_model_mag(
        y, sysvecs, len(sysvecs), method='RidgeCV',
        verbose=True
    )

    pca_mags[f'SYS{ap}'] = model_mag_sys
    pca_mags[f'OLS{ap}'] = model_mag
    pca_mags[f'RR{ap}'] = model_mag_ridge
    pca_mags[f'IRM{ap}'] = y

    vecnames = [f'IRM{ap}', f'SYS{ap}', f'OLS{ap}', f'RR{ap}',
                f'BGV', 'CCDTEMP']

    #
    # make the plot
    #

    plt.close('all')
    fig, axs = plt.subplots(nrows=len(vecnames), ncols=2,
                            figsize=(10,len(vecnames)*1.1), sharex=True)

    for ix, vecname in enumerate(vecnames):

        if ix <= 3:
            axs[ix,0].scatter(time, pca_mags[vecname], c='black',
                              alpha=0.9, zorder=2, s=3, rasterized=True,
                              linewidths=0)
            axs[ix,0].set_ylim(axs[ix,0].get_ylim()[::-1])

        else:
            axs[ix,0].scatter(
                time,
                MinMaxScaler().fit_transform(
                    data[vecname][:,None].astype(np.float64)
                ).flatten(),
                c='black', alpha=0.9, zorder=2, s=3, rasterized=True,
                linewidths=0)

            if use_smootheigvecs and vecname == 'BGV':
                # display smooth BGV just to see it
                axs[ix,0].scatter(
                    time,
                    smooth_fn(
                        time,
                        MinMaxScaler().fit_transform(
                            data[vecname][:,None].astype(np.float64)
                        ).flatten()
                    )-1,
                    c='C0', alpha=0.9, zorder=3, s=2, rasterized=True,
                    linewidths=0)

        axs[ix,0].set_ylabel(vecname)

    for ix in range(len(vecnames)):
        axs[ix,1].scatter(
            time, eigenvecs[ix, :], c='black', alpha=0.9, zorder=2, s=3,
            rasterized=True, linewidths=0
        )
        if use_smootheigvecs:
            axs[ix,1].scatter(
                time, smooth_eigenvecs[ix, :], c='C0', alpha=0.9, zorder=3,
                s=2, rasterized=True, linewidths=0
            )

    axs[-1, 0].set_xlabel('time')
    axs[-1, 1].set_xlabel('time')
    fig.suptitle(
        f"Rp={primaryhdr['phot_rp_mean_mag']:.1f}, GDR2 {source_id}"
    )

    outpath = os.path.join(
        OUTDIR,
        os.path.basename(lcpath).replace('.fits', f'_ap{ap}_pca-compare.png')
    )
    f.savefig(outpath, dpi=200, tight_layout=True)
    print(f'made {outpath}')


def explore_ic2602():
    """
    Run some (manual, plot-level) checks on IC 2602.
    """

    testname = 'ic2602_examples'

    source_path = (
        f'/home/lbouma/proj/cdips/tests/data/test_pca_{testname}.csv'
    )
    df = pd.read_csv(source_path, comment='#', names=['source_id'])

    outdir = os.path.join(
        '/home/lbouma/proj/cdips/tests/test_pca_plots', f'{testname}'
    )
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    global OUTDIR
    OUTDIR = outdir

    for source_id in list(df.source_id):

        lcpaths = find_cdips_lc_paths(source_id)

        for lcpath in lcpaths:

            hdrlist = ['CAMERA', 'CCD', 'SECTOR', 'PROJID']
            _d = iu.get_header_keyword_list(lcpath, hdrlist)
            for k,v in _d.items():
                _d[k] = int(v)

            eigveclist, n_comp_df = dtr.prepare_pca(
                _d['CAMERA'], _d['CCD'], _d['SECTOR'], _d['PROJID']
            )

            run_explore_pca(lcpath, eigveclist, n_comp_df, use_sysvecs=True)


if __name__ == "__main__":

    run_explore_ccd = 0
    run_explore_ic2602 = 1

    if run_explore_ic2602:
        explore_ic2602()

    if run_explore_ccd:
        explore_pca(sector=9, cam=2, ccd=3)
