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

from sklearn.preprocessing import MinMaxScaler
import astrobase.imageutils as iu

from cdips.lcproc import (
    reformat_lcs_for_mast as rlm,
    mask_orbit_edges as moe,
    detrend as dtr
)
from cdips.utils.lcutils import (
    find_cdips_lc_paths
)

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


def explore_pca(
    sector=9,
    cam=2,
    ccd=3,
    symlinkdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS/',
    outdir='/home/lbouma/proj/cdips/tests/test_pca_plots/',
    OC_MG_CAT_ver=0.6,
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


def run_explore_pca(lcpath, eigveclist, n_comp_df, use_sysvecs=False,
                    use_smootheigvecs=True):
    """
    use_sysvecs:
        whether or not to include extra vectors in the regression.
    """

    sysvecnames = ['BGV']
    dtrvecs, sysvecs, ap, primaryhdr, data, eigenvecs, smooth_eigenvecs = (
        dtr.get_dtrvecs(lcpath, eigveclist, sysvecnames=sysvecnames)
    )
    time, y = data['TMID_BJD'], data[f'IRM{ap}']
    source_id = primaryhdr['OBJECT']

    ##########################################
    # NOTE: below will be ported
    n_components = min([int(n_comp_df[f'fa_cv_ap{ap}']), 5])
    if use_sysvecs:
        n_components += len(sysvecnames)
    model_mag, n_comp = dtr.calculate_linear_model_mag(
        y, dtrvecs, n_components, method='LinearRegression'
    )

    pca_mags = {}
    pca_mags[f'PCA{ap}'] = model_mag
    primaryhdr[f'PCA{ap}NCMP'] = (
        n_comp,
        f'N principal components PCA{ap}'
    )
    # TODO: once the above is tested, port it to
    #lcproc.reformat_lcs_for_mast, because it's much cleaner
    ##########################################

    model_mag_ridge, _ = dtr.calculate_linear_model_mag(
        y, dtrvecs, n_components, method='RidgeCV', verbose=True
    )

    model_mag_sys, _ = dtr.calculate_linear_model_mag(
        y, sysvecs, len(sysvecs), method='RidgeCV', verbose=True
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
                ).flatten()
                -
                np.nanmedian(MinMaxScaler().fit_transform(
                    data[vecname][:,None].astype(np.float64)
                ).flatten())
                ,
                c='black', alpha=0.9, zorder=2, s=3, rasterized=True,
                linewidths=0)

            if use_smootheigvecs and vecname == 'BGV':
                # display smooth BGV just to see it
                axs[ix,0].scatter(
                    time,
                    dtr.eigvec_smooth_fn(
                        time,
                        MinMaxScaler().fit_transform(
                            data[vecname][:,None].astype(np.float64)
                        ).flatten()
                        -
                        np.nanmedian(MinMaxScaler().fit_transform(
                            data[vecname][:,None].astype(np.float64)
                        ).flatten())
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
    fig.savefig(outpath, dpi=200, tight_layout=True)
    print(f'made {outpath}')


if __name__ == "__main__":

    run_explore_ccd = 0
    run_explore_ic2602 = 1

    if run_explore_ic2602:
        explore_ic2602()

    if run_explore_ccd:
        explore_pca(sector=9, cam=2, ccd=3)
