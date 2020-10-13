"""
Test a few PCA ensemble detrending variants.
"""

import os
from glob import glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn import preprocessing

from cdips.lcproc import reformat_lcs_for_mast as rlm
from cdips.lcproc import mask_orbit_edges as moe
from cdips.lcproc import detrend as dtr

import imageutils as iu
from astrobase.lcmath import find_lc_timegroups

def test_pca(
    sector=9,
    cam=2,
    ccd=3,
    symlinkdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS/',
    outdir='/home/lbouma/proj/cdips/tests/test_pca/',
    OC_MG_CAT_ver=0.4,
    cdipsvnum=1
):

    lcpaths = glob(os.path.join(
        symlinkdir, f'sector-{sector}', f'cam{cam}_ccd{ccd}', '*_llc.fits')
    )

    projid = iu.get_header_keyword(lcpaths[0], 'PROJID')

    eigveclist, n_comp_df = dtr.prepare_pca(cam, ccd, sector, projid)

    outdir = os.path.join(outdir, f's{sector}_c{cam}_c{ccd}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    global OUTDIR
    OUTDIR = outdir

    for lcpath in lcpaths:
        run_test_pca(lcpath, eigveclist, n_comp_df, use_sysvecs=True)


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


def rescale(vec, target_iqr):

    current_iqr = np.nanpercentile(vec, 75) - np.nanpercentile(vec, 25)

    vec -= np.nanmean(vec)

    factor = target_iqr / current_iqr

    return vec*factor




def run_test_pca(lcpath, eigveclist, n_comp_df, use_sysvecs=False):
    """
    use_sysvecs:
        whether or not to include extra vectors in the regression.
    """

    # eigveclist: length 3, each with a np.ndarray of eigenvectors given by
    # dtr.prepare_pca

    hdul = fits.open(lcpath)
    primaryhdr, hdr, data = (
        hdul[0].header, hdul[1].header, hdul[1].data
    )
    hdul.close()

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

    for ix, eigenvecs in enumerate(eigveclist):

        if np.any(pd.isnull(eigenvecs)):
            raise ValueError('got nans in eigvecs. bad!')

        ap = ix+1

        y = data[f'IRM{ap}']
        target_iqr = np.nanpercentile(y, 75) - np.nanpercentile(y, 25)
        scales[ap] = target_iqr

        if use_sysvecs:
            # nb. found XIC and YIC probably introduce extra noise.
            # sysvecnames = ['BGV', 'CCDTEMP', 'XIC', 'YIC']
            #sysvecnames = ['BGV', 'CCDTEMP']
            sysvecnames = ['CCDTEMP']

            sysvecs = np.vstack(
                [rescale(data[s], target_iqr) for s in sysvecnames]
            )

            sysvecs = np.vstack(
                [np.zeros(len(data[sysvecnames[0]])), sysvecs]
            )

            dtrvecs = np.vstack([sysvecs, eigenvecs])

        else:
            dtrvecs = eigenvecs

        ##########################################
        # NOTE: below will be ported
        n_components = max([int(n_comp_df[f'fa_cv_ap{ap}']), 5])
        if use_sysvecs:
            n_components += ( len(sysvecnames) + 1 )
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
        pca_mags[f'OLS-RR{ap}'] = model_mag - model_mag_ridge
        pca_mags[f'IRM{ap}'] = y

    time = data['TMID_BJD']

    for ap in [1,2]:

        vecnames = [f'IRM{ap}', f'SYS{ap}', f'OLS{ap}', f'RR{ap}',
                    f'OLS-RR{ap}', 'CCDTEMP']

        plt.close('all')
        f, axs = plt.subplots(nrows=len(vecnames), ncols=2,
                              figsize=(10,len(vecnames)*1.1), sharex=True)

        for ix, vecname in enumerate(vecnames):
            if ix <= 4:
                axs[ix,0].scatter(time, pca_mags[vecname], c='black', alpha=0.9,
                                zorder=2, s=3, rasterized=True, linewidths=0)
            else:
                axs[ix,0].scatter(time, rescale(data[vecname], scales[ap]),
                                c='black', alpha=0.9, zorder=2, s=3,
                                rasterized=True, linewidths=0)

            axs[ix,0].set_ylabel(vecname)

        for ix in range(len(vecnames)):
            axs[ix,1].scatter(time, eigenvecs[ix, :], c='black', alpha=0.9,
                              zorder=2, s=3, rasterized=True, linewidths=0)


        axs[-1, 0].set_xlabel('time')
        axs[0, 0].set_title(f"Rp={primaryhdr['phot_rp_mean_mag']:.1f}")

        outpath = os.path.join(
            OUTDIR,
            os.path.basename(lcpath).replace('.fits', f'_ap{ap}_pca-compare.png')
        )
        f.savefig(outpath, dpi=200, tight_layout=True)
        print(f'made {outpath}')


if __name__ == "__main__":
    test_pca()
