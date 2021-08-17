import os
import pandas as pd, numpy as np
from astropy.io import fits
from astrobase import imageutils as iu

def test_single_lc():

    lcdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam1_ccd1'
    lcname = 'hlsp_cdips_tess_ffi_gaiatwo0003318240884775406080-0006_tess_v01_llc.fits'
    lcpath = os.path.join(lcdir, lcname)

    with fits.open(lcpath) as hdulist:
        verify_lightcurve(hdulist)


def test_length_of_data(hdulist):
    """
    all flux vector should have same length as time vector.
    """

    keys = ['TMID_BJD', 'IRM1', 'IRM2', 'IRM3', 'PCA1', 'PCA2', 'PCA3',
            'TFA1', 'TFA2', 'TFA3']

    N_entries = [len(hdulist[1].data[k]) for k in keys]

    if np.any(np.diff(N_entries)):
        errmsg = 'a flux vector has diff length from time vector'
        raise AssertionError(errmsg)
    else:
        print('verified flux vectors have same length as time vector')


def test_nan_placement(hdulist):
    """
    there should not be finite values in detrended lightcurve indices if there
    were nans in the raw light curves.
    """

    for ap in range(1,4):

        irm_key = 'IRM{ap}'.format(ap=ap)
        pca_key = 'PCA{ap}'.format(ap=ap)
        tfa_key = 'TFA{ap}'.format(ap=ap)

        #
        # if any nan values in the IRM light curve are NOT nan in the PCA light
        # curve, raise an error. note we know the lengths are the same.
        #
        if np.any(
            np.in1d(
                np.argwhere(np.isnan(hdulist[1].data[irm_key])),
                np.argwhere(~np.isnan(hdulist[1].data[pca_key]))
            )
        ):
            errmsg = (
                'got nan values in {irm_key} LC that are NOT nan in {pca_key} LC'.
                format(irm_key=irm_key, pca_key=pca_key)
            )
            raise AssertionError(errmsg)
        else:
            print('verified that no nans in {irm_key} are NOT nan in {pca_key}'.
                  format(irm_key=irm_key, pca_key=pca_key))

        #
        # do the same for IRM to TFA
        #
        if np.any(
            np.in1d(
                np.argwhere(np.isnan(hdulist[1].data[irm_key])),
                np.argwhere(~np.isnan(hdulist[1].data[tfa_key]))
            )
        ):
            errmsg = (
                'got nan values in {irm_key} LC that are NOT nan in {tfa_key} LC'.
                format(irm_key=irm_key, tfa_key=tfa_key)
            )
            print(errmsg)
            import IPython; IPython.embed()
            raise AssertionError(errmsg)
        else:
            print('verified that no nans in {irm_key} are NOT nan in {tfa_key}'.
                  format(irm_key=irm_key, tfa_key=tfa_key))


def verify_lightcurve(hdulist):
    """
    all flux vector should have same length as time vector.

    there should not be finite values in detrended lightcurve indices if there
    were nans in the raw light curves.
    """

    test_length_of_data(hdulist)

    test_nan_placement(hdulist)



if __name__ == "__main__":
    test_single_lc()
