"""
Contents:
    _given_mag_get_flux
    find_cdips_lc_paths: given a source_id, return the paths
    get_lc_data: given a path, return selected vectors.
"""
from glob import glob
import os
import numpy as np
from astropy.io import fits

def find_cdips_lc_paths(
    source_id,
    LCDIR='/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS'
    try_mast=False,
):
    """
    Given a Gaia source ID, return all available CDIPS light curves (i.e.,
    their paths) for that star.

    kwargs:

        LCDIR (str): local directory, containing light curves of interest (in
        arbitrarily many subdirectories), and a metadata file with their paths
        (in the lc_list_YYYYMMDD.txt format).

        try_mast (bool): default False. If True, will run an astroquery search
        through the MAST portal, and will download any available CDIPS light
        curves.  Recommended to keep this as False; the MAST portal seems to
        time out a lot, and downloading from it is slow. Better to just have
        the paths already on disk.
    """

    if try_mast:
        errmsg = (
            'Could wrap the CDIPS light curve getter from astrobase...'
            'but currently no need for this'
        )
        raise NotImplementedError(errmsg)

    # default approach: use the latest "lc_list" metadata.
    if not os.path.exists(LCDIR):
        errmsg = (
            f'Expected to find {LCDIR}'
        )
        raise ValueError(errmsg)

    METADATAPATHS = glob(os.path.join(LCDIR, 'lc_list_*.txt'))

    import IPython; IPython.embed()

    assert 0

    return lcpaths



def get_lc_data(lcpath, mag_aperture='TFA2', tfa_aperture='TFA2'):
    """
    Given a CDIPS LC path, return some key vectors.
    """

    hdul = fits.open(lcpath)

    tfa_time = hdul[1].data['TMID_BJD']
    ap_mag = hdul[1].data[mag_aperture]
    tfa_mag = hdul[1].data[tfa_aperture]

    xcc, ycc = hdul[0].header['XCC'], hdul[0].header['YCC']
    ra, dec = hdul[0].header['RA_OBJ'], hdul[0].header['DEC_OBJ']

    tmag = hdul[0].header['TESSMAG']

    hdul.close()

    # e.g.,
    # */cam2_ccd1/hlsp_cdips_tess_ffi_gaiatwo0002916360554371119104-0006_tess_v01_llc.fits
    source_id = lcpath.split('gaiatwo')[1].split('-')[0].lstrip('0')

    return source_id, tfa_time, ap_mag, xcc, ycc, ra, dec, tmag, tfa_mag


def _given_mag_get_flux(mag, err_mag=None):
    """
    Given a time-series of magnitudes, convert it to relative fluxes.
    """

    mag_0, f_0 = 12, 1e4
    flux = f_0 * 10**( -0.4 * (mag - mag_0) )
    fluxmedian = np.nanmedian(flux)
    flux /= fluxmedian

    if err_mag is None:
        return flux

    else:

        #
        # sigma_flux = dg/d(mag) * sigma_mag, for g=f0 * 10**(-0.4*(mag-mag0)).
        #
        err_flux = np.abs(
            -0.4 * np.log(10) * f_0 * 10**(-0.4*(mag-mag_0)) * err_mag
        )
        err_flux /= fluxmedian

        return flux, err_flux
