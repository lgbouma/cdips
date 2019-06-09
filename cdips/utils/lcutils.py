from astropy.io import fits

def get_lc_data(lcpath, tfa_aperture='TFA2'):

    hdul = fits.open(lcpath)

    tfa_time = hdul[1].data['TMID_BJD']
    tfa_mag = hdul[1].data[tfa_aperture]

    xcc, ycc = hdul[0].header['XCC'], hdul[0].header['YCC']
    ra, dec = hdul[0].header['RA_OBJ'], hdul[0].header['DEC_OBJ']

    tmag = hdul[0].header['TESSMAG']

    hdul.close()

    # e.g.,
    # */cam2_ccd1/hlsp_cdips_tess_ffi_gaiatwo0002916360554371119104-0006_tess_v01_llc.fits
    source_id = lcpath.split('gaiatwo')[1].split('-')[0].lstrip('0')

    return source_id, tfa_time, tfa_mag, xcc, ycc, ra, dec, tmag
