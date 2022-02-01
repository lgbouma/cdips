"""
Contents:

Light curve retrieval:

    find_cdips_lc_paths: given a source_id, return the paths

    find_calibration_lc_paths: given a source_id, return paths to the
    CALIBRATION light curves ({source_id}_llc.fits)

    make_calibration_list: make a metadata file consisting of the G_Rp<13
    calibration light curve paths, PLUS the G_Rp<16 cluster light curve paths.

    make_lc_list: make a metadata file consisting of the G_Rp<16 cluster light
    curves paths.

Light curve miscellanea:

    _given_mag_get_flux: self-descriptive.

    get_lc_data: given a CDIPS lcpath, return some key default vectors.

    get_best_ap_number_given_lcpath: self-descriptive.

    stitch_light_curves: stitch lists of light curves across sectors.
"""

from glob import glob
import os
import numpy as np
from copy import deepcopy

from astropy.io import fits
from astropy.time import Time
from cdips.utils import bash_grep

from astrobase.imageutils import get_header_keyword
from cdips.utils.tess_noise_model import N_pixels_in_aperture_Sullivan

def find_cdips_lc_paths(
    source_id,
    LCDIR='/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS',
    raise_error=True,
    use_calib=False,
    try_mast=False
):
    """
    Given a Gaia source ID, return list of all available CDIPS light curves
    (i.e., their paths) for that star. For this to be useful, the
    paths need to be on disk. However this function works by searching
    a metadata file.  If no matches are found, returns None.

    kwargs:

        LCDIR (str): local directory, containing light curves of interest (in
        arbitrarily many subdirectories), and a metadata file with their paths
        (in the lc_list_YYYYMMDD.txt format).

        raise_error (bool): will raise an error if no light curves are found.

        use_calib (bool): if this is True, searches for "calibration"
        light curves instead (the Rp<13 sample, PLUS the Rp<16 cluster
        sample).

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

    if not use_calib:
        METADATAPATHS = glob(os.path.join(LCDIR, 'lc_list_*.txt'))
    else:
        METADATAPATHS = glob(os.path.join(LCDIR, 'calibration_list_*.txt'))
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
            print('WRN! ' + errmsg)

        return None

    else:

        if not use_calib:
            lcpaths = [os.path.join(LCDIR, g) for g in grep_output]
        else:
            lcpaths = grep_output

        return lcpaths



def get_lc_data(lcpath, mag_aperture='PCA2', tfa_aperture='TFA2'):
    """
    Given a CDIPS LC path, return some key vectors.
    """

    hdul = fits.open(lcpath)

    time = hdul[1].data['TMID_BJD']
    ap_mag = hdul[1].data[mag_aperture]
    tfa_mag = hdul[1].data[tfa_aperture]

    xcc, ycc = hdul[0].header['XCC'], hdul[0].header['YCC']
    ra, dec = hdul[0].header['RA_OBJ'], hdul[0].header['DEC_OBJ']

    tmag = hdul[0].header['TESSMAG']

    hdul.close()

    # e.g.,
    # */cam2_ccd1/hlsp_cdips_tess_ffi_gaiatwo0002916360554371119104-0006_tess_v01_llc.fits
    source_id = lcpath.split('gaiatwo')[1].split('-')[0].lstrip('0')

    return source_id, time, ap_mag, xcc, ycc, ra, dec, tmag, tfa_mag


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
    timelist:list,
    maglist:list,
    magerrlist:list,
    extravecdict:dict = None,
    magsarefluxes:bool = False,
    normstitch:bool = False
):
    """
    Given lists of , returning stitched times, fluxes, and flux
    errors.

    Args:

        timelist, maglist, magerrlist: lists of np.ndarrays of times,
        magnitudes, and mag errors (where each list entry is a TESS sector,
        Kepler quarter, etc.)

        extravecdict: dict of lists of supplemental vectors. For instance,
        with two sectors, if you wanted to stitch BGV vectors as well, would
        be:
            {'BGV':[bgvlistsec0, bgvlistsec1],
             'XCC':[xcclistsec0, xcclistsec1]}

        magsarefluxes: if True, does not convert mag to flux.

        normstitch: normalize relative flux across sectors/quarters to keep 5th
        to 95th percentile amplitude constant.
    """
    for l in [timelist, maglist, magerrlist]:
        assert isinstance(l, list)

    # get median-normalized fluxes across each sector
    fluxlist, fluxerrlist = [], []
    for t, m, e in zip(timelist, maglist, magerrlist):
        if magsarefluxes:
            f, f_e = m, e
        else:
            f, f_e = _given_mag_get_flux(m, e)
        fluxlist.append(f)
        fluxerrlist.append(f_e)

    starttimes = [t[0] for t in timelist]
    if not np.all(np.diff(starttimes) > 0):
        print('expected timelist to already be sorted -- will sort')
        # re-sort timelist entries here by the starting time!
        sortind = np.argsort(starttimes)
        timelist = list(np.array(timelist, dtype=object)[sortind])
        fluxlist = list(np.array(fluxlist, dtype=object)[sortind])
        fluxerrlist = list(np.array(fluxerrlist, dtype=object)[sortind])

    if normstitch:
        # require 5th to 95th percentile to be constant in flux across
        # sectors/quarters.
        A_5_95 = [
            np.nanpercentile(f,95) - np.nanpercentile(f,5) for f in fluxlist
        ]
        mean_A_5_95 = np.mean(A_5_95)
        div_factor = A_5_95 / mean_A_5_95
        normfluxlist = [f/div_f for f, div_f in zip(fluxlist, div_factor)]
        norm_A_5_95 = np.array([
            np.nanpercentile(f,95) - np.nanpercentile(f,5) for f in normfluxlist
        ])
        assert np.all(np.diff(norm_A_5_95) < 1e-14)

        # renormalize around median of 1
        offsets = [np.nanmedian(f)-1 for f in normfluxlist]
        normfluxlist = [f-o for f,o in zip(normfluxlist, offsets)]

        # assign to the actual flux
        fluxlist = deepcopy(normfluxlist)

        # propagate to stitched uncertainties
        normfluxerrlist = [e/div_f for e, div_f in zip(fluxerrlist, div_factor)]
        fluxerrlist = deepcopy(normfluxerrlist)

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


def make_calibration_list(
    OUTDIR='/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS'
    ):
    """
    Make a metadata file consisting of [all of!] the G_Rp<13 calibration light
    curve paths, plus the G_Rp<16 target light curve paths.  In other words,
    paths to all the light curves that have been made. Writes it to
    OUTDIR/calibration_list_YYYYMMDD.txt.  Also writes count logs to
    OUTDIR/counts/calibration_sector_{ix}.count
    """

    from cdips.utils import today_YYYYMMDD
    todaystr = today_YYYYMMDD()

    outpath = os.path.join(OUTDIR, f'calibration_list_{todaystr}.txt')

    countdir = os.path.join(OUTDIR, 'counts')
    if not os.path.exists(countdir):
        os.mkdir(countdir)

    CALIBDIR = '/nfs/phtess2/ar0/TESS/FFI/LC/FULL'

    for i in range(1, 14):

        # e.g., /nfs/phtess2/ar0/TESS/FFI/LC/FULL/s0012/*/*.fits
        calibglob = os.path.join(CALIBDIR, f's{str(i).zfill(4)}', '*', '*_llc.fits')

        countpath = os.path.join(countdir, f'calibration_sector_{i}.count')

        print(f'Sector {i}...')

        countcmd = (
            f'echo {calibglob} | xargs ls | wc -l > {countpath}'
        )
        returncode = os.system(countcmd)
        print(f'ran {countcmd}')
        if returncode != 0:
            raise AssertionError('count cmd failed!!')

        appendcmd = (
            f'echo {calibglob} | xargs ls >> {outpath}'
        )
        returncode = os.system(appendcmd)
        print(f'ran {appendcmd}')
        if returncode != 0:
            raise AssertionError('append cmd failed!!')


def make_lc_list(
    listpath='/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/lc_list_20220131.txt',
    sector_interval=None
):
    """
    Make a TXT file consisting of the CDIPS cluster light curves paths. This
    will almost always be run on phtess[N] machines, unless you collect the
    entire CDIPS reductions on some other system.  This TXT metadata file is
    useful for quickly retrieving files given a Gaia DR2 source_id.

    Args:
        listpath (str): where the list of light curve paths will be written.

        sector_interval (list): E.g., [1,19] to span Sectors 1 through 19 in
        the list that is created.
    """
    assert isinstance(sector_interval, list)
    if os.path.exists(listpath):
        raise ValueError(f'Found {listpath}. Escaping to not overwrite.')

    sector_start = int(sector_interval[0])
    sector_end = int(sector_interval[1])

    assert sector_end > sector_start

    BASEDIR = "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS"
    LCGLOB = "hlsp_cdips_*_llc.fits"

    for sector in range(sector_start, sector_end+1):
        print(42*'-')
        print(f'Beginning LC retrieval for sector {sector}...')

        lcpaths = glob(os.path.join(
            BASEDIR, f"sector-{sector}", "cam*_ccd*", LCGLOB
        ))

        N_lcs = len(lcpaths)
        print(f'Sector {sector} has {N_lcs} CDIPS light curves (G_RP<16).')

        with open(listpath, "a") as fbuf:
            fbuf.writelines(
                "\n".join(lcpaths)+"\n"
            )
        print(f'... appended them to {listpath}')
