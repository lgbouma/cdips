"""
Contents:

    _given_mag_get_flux: self-descriptive.

    find_cdips_lc_paths: given a source_id, return the paths

    get_lc_data: given a path, return selected vectors.

    get_best_ap_number_given_lcpath: self-descriptive.

    stitch_light_curves: stitch lists of light curves across sectors.
"""

from glob import glob
import os
import numpy as np

from astropy.io import fits
from astropy.time import Time
from cdips.utils import bash_grep

from astrobase.imageutils import get_header_keyword
from cdips.utils.tess_noise_model import N_pixels_in_aperture_Sullivan

def find_cdips_lc_paths(
    source_id,
    LCDIR='/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS',
    raise_error=True,
    try_mast=False
):
    """
    Given a Gaia source ID, return all available CDIPS light curves (i.e.,
    their paths) for that star.

    kwargs:

        LCDIR (str): local directory, containing light curves of interest (in
        arbitrarily many subdirectories), and a metadata file with their paths
        (in the lc_list_YYYYMMDD.txt format).

        raise_error (bool): will raise an error if no light curves are found.

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

    #
    # get and use the latest "lc_list_{YYYYMMDD}" metadata.
    #
    if not os.path.exists(LCDIR):
        errmsg = (
            f'Expected to find {LCDIR}'
        )
        raise ValueError(errmsg)

    METADATAPATHS = glob(os.path.join(LCDIR, 'lc_list_*.txt'))
    assert len(METADATAPATHS) >= 1

    def _compose(f1, f2):
        return lambda x: f1(f2(x))

    get_yyyymmdd = lambda x: os.path.basename(x).split('_')[-1].split('.')[0]
    get_timejd = lambda x: Time(x[:4] + '-' + x[4:6] + '-' + x[6:]).jd
    get_jds = _compose(get_timejd, get_yyyymmdd)

    timejds = list(map(get_jds, METADATAPATHS))

    latest_lclist_path = METADATAPATHS[np.argmax(timejds)]

    #
    # now read it to get the light curve paths
    #

    pattern = f'{source_id}'
    grep_output = bash_grep(pattern, latest_lclist_path)

    if grep_output is None:

        errmsg = (
            f'Expected to find light curve for {source_id}, '
            'and did not!'
        )
        if raise_error:
            raise ValueError(errmsg)
        else:
            print('WRN!' + errmsg)

        return None

    else:

        lcpaths = [os.path.join(LCDIR, g) for g in grep_output]

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


def get_best_ap_number_given_lcpath(lcpath):
    """
    Given a CDIPS LC, figure out which aperture is "optimal" given the
    brightness of the star, assuming Sullivan+2015's defintion of "optimal".
    """

    tess_mag = get_header_keyword(lcpath, 'TESSMAG')

    optimal_N_pixels = N_pixels_in_aperture_Sullivan(float(tess_mag))

    available_N_pixels = 3.14*np.array([1, 1.5, 2.25])**2

    best_idx = np.argmin(np.abs(available_N_pixels - optimal_N_pixels))

    ap_number = best_idx + 1

    return ap_number


def stitch_light_curves(
    timelist,
    maglist,
    magerrlist,
    extravecdict=None
):
    """
    Given lists of times, magnitudes, and mag errors (where each index is
    presumably a TESS sector), returning stitched times, fluxes, and flux
    errors.

    Kwargs:

        extraveclists: list of lists of supplemental vectors. For instance,
        with two sectors, if you wanted to stitch BGV vectors as well, would
        be:
            {'BGV':[bgvlistsec0, bgvlistsec1],
             'XCC':[xcclistsec0, xcclistsec1]}

    """
    for l in [timelist, maglist, magerrlist]:
        assert isinstance(l, list)

    # get median-normalized fluxes across each sector
    fluxlist, fluxerrlist = [], []
    for t, m, e in zip(timelist, maglist, magerrlist):
        f, f_e = _given_mag_get_flux(m, e)
        fluxlist.append(f)
        fluxerrlist.append(f_e)

    starttimes = [t[0] for t in timelist]
    if not np.all(np.diff(starttimes) > 0):
        raise ValueError('expected timelist to already be sorted')

    time = np.hstack(timelist)
    flux = np.hstack(fluxlist)
    fluxerr = np.hstack(fluxerrlist)

    if extravecdict is None:
        return time, flux, fluxerr

    else:

        extravecs = {}
        for k,v in extravecdict.items():
            extravecs[k] = np.hstack(v)

        return time, flux, fluxerr, extravecs
