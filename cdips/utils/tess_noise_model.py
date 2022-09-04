'''
Parametrized noise model without optimal apertures (AKA with *selected*
apertures).

Given source T mag, and a coordinate, this function gives predicted TESS RMS
for the source.

It does so by using an analytic N_pix(T_mag) given to the TSWG by Jon Jenkins,
and, using an updated PSF.

The relevant function is `noise_model(...)`.

If run on the command line,

    >>> $(bash) python noise_model.py

then this script produces a csv file with the tabulated values.

NOTE:
This code is derivative of both Zach Berta-Thompson's SNR calculator and Josh
Winn's IDL TESS SNR calculator.

The former is at https://github.com/zkbt/spyffi/blob/master/Noise.py.
The latter is saved in this directory (`JNW_calc_noise.pro`).

Author: Luke Bouma.
Date: Thu 18 Jan 2018 05:47:35 PM EST
'''
from __future__ import division, print_function

import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

import cdips as cd
datadir = os.path.join(os.path.dirname(cd.__path__[0]), 'data')
resultsdir = os.path.join(os.path.dirname(cd.__path__[0]), 'results')

###############################################################################
# Fixed TESS properties are kept as globals.

global subexptime, e_pix_ro, effective_area, sys_limit, pix_scale

subexptime = 2.0      # subexposure time [seconds] (n_exp = exptime/subexptime)
e_pix_ro = 10.0       # rms in no. photons/pixel from readout noise
# https://archive.stsci.edu/files/live/sites/mast/files/home/missions-and-data/active-missions/tess/_documents/TESS_Instrument_Handbook_v0.1.pdf
effective_area = 86.6 # geometric collecting area. it's in the handbook. v0.1, figure 3.4
sys_limit = 60.0      # minimum uncertainty in 1 hr of data, in ppm
pix_scale = 21.1      # arcsec per pixel

###############################################################################

def N_pix_in_aperture_Jenkins(T):
    '''
    Analytic number of pixels in aperture. Provided to TSWG by Jon Jenkins,
    based on lab testing. An extra "ring" of pixels will actually be collected
    by the TESS spacecraft.
    '''
    c_3 = -0.2592
    c_2 = 7.741
    c_1 = -77.792
    c_0 = 274.2989
    return c_3*T**3 + c_2*T**2 + c_1*T + c_0

def N_pixels_in_aperture_Sullivan(T):
    '''
    optimal n_pixels in aperture according to S+15. Less wonky structure at
    faint end compared to Jenkins.
    '''
    from scipy.interpolate import interp1d
    df = pd.read_csv(
            os.path.join(datadir, 'LITERATURE_DATA', 'Sullivan_2015_optimalnumberofpixels.txt'),
            comment='#', delimiter=','
            )
    tmag = np.array(df['tmag']) # it's really I_C
    npix = np.array(df['npix'])
    func = interp1d(tmag, npix)
    return func(T)


def photon_flux_from_source(T_mag):
    '''
    in:
        T_mag (np.ndarray): of the source(s)

    out:
        photon flux from the source in the TESS band [units: ph/s/cm^2].
    '''

    # Zero point stated in Sullivan et al 2015:
    # A T=0 star gives a photon flux of 1.514e6 ph/s/cm^2.

    F_T0 = 1.514e6

    F = 10**(-0.4 * ( T_mag )) * F_T0

    return F


def get_sky_bkgnd(coords, exptime):
    '''
    in:
        input coordinate (astropy SkyCoord instance)

        exposure time (seconds)

    out:
        sky background from zodiacal light at coords [units: e/px]

        (NB. background stars are accounted in post-processing by the TSWG's
        synthetic image procedure)
    '''

    elat = coords.barycentrictrueecliptic.lat.value
    elon = coords.barycentrictrueecliptic.lon.value
    glat = coords.galactic.b.value
    glon = coords.galactic.l.value
    glon -= 180 # Winn's eq 7 (below) requires longitude from -180 to 180
    assert np.all(glon > -180) and np.all(glon < 180)

    # Solid area of a pixel (arcsec^2).
    omega_pix = pix_scale ** 2.

    # Photoelectrons/pixel from zodiacal light.
    dlat = (np.abs(elat) - 90.) / 90.
    vmag_zodi = 23.345 - 1.148 * dlat ** 2.

    # Eqn (3) from Josh Winn's memo on sky backgrounds. This comes from
    # integrating a model ZL spectrum over the TESS bandpass.
    e_pix_zodi = (
      10.0 ** (-0.4 * (vmag_zodi - 22.8)) * 2.39e-3
      * effective_area * omega_pix * exptime
    )

    # You also need the diffuse background from unresolved stars. Eq (7) from
    # the same memo
    a0, a1, a2, a3 = 18.9733, 8.833, 4.007, 0.805
    I_surface_brightness = (
      a0 + a1 * np.abs(glat)/40 + a2 * (np.abs(glat)/180)**a3
    )
    e_pix_faintstars = (
      10.0 ** (-0.4 * I_surface_brightness) * 1.7e6
      * effective_area * omega_pix * exptime
    )

    # for CDIPS, zodi should be small, photon flux due to unresolved stars
    # should dominate

    return e_pix_faintstars


def noise_model(
        T_mags,
        coords,
        exptime=120):
    '''
    ----------
    Mandatory inputs:

    either all floats, or else all 1d numpy arrays of length N_sources.

        T_mags:
            TESS magnitude of the source(s)

        coords:
            target coordinates, a (N_sources * 2) numpy array of (ra, dec),
            specified in degrees.

    ----------
    Optional inputs:


        exptime (float):
            total exposure time in seconds. Must be a multiple of 2 seconds.

    ----------
    Returns:

        [N_sources x 6] array of:
            number of pixels in selected apertures,
            noise for selected number of pixels,
            each of the noise components (star, sky, readout, systematic).

    '''

    # Check inputs. Convert coordinates to astropy SkyCoord instance.
    if not isinstance(T_mags, np.ndarray):
        T_mags = np.array([T_mags])
    assert isinstance(coords, np.ndarray)
    if len(coords.shape)==1:
        coords = coords.reshape((1,2))
    assert coords.shape[1] == 2

    coords = SkyCoord(
                 ra=coords[:,0]*units.degree,
                 dec=coords[:,1]*units.degree,
                 frame='icrs'
                 )

    assert exptime % subexptime == 0, \
            'Exposure time must be multiple of 2 seconds.'
    assert T_mags.shape[0] == coords.shape[0]

    # Basic quantities.
    N_sources = len(T_mags)
    N_exposures = exptime/subexptime

    # Photon flux from source in ph/s/cm^2.
    f_ph_source = np.array(photon_flux_from_source(T_mags))

    # Compute number of photons from source, per exposure.
    ph_source = f_ph_source * effective_area * exptime

    # Load in average PRF produced by `ctd_avg_field_angle_avg.py`.
    prf_file = os.path.join(datadir, 'LITERATURE_DATA', 'average_PRF.fits')
    hdu = fits.open(prf_file)
    avg_PRF = hdu[0].data

    # Compute cumulative flux fraction, sort s.t. the brightest pixel is first.
    CFF = np.cumsum(np.sort(avg_PRF)[::-1])

    # For each source, compute the number of photons collected (in each
    # exposure) as a function of aperture size. Save as array of [N_sources *
    # N_pixels_in_aperture].
    ph_source_all_ap = ph_source[:, None] * CFF[None, :]

    # Convert to number of electrons collected as a function of aperture size.
    # These are the same, since Josh Winn's photon flux formula already
    # accounts for the quantum efficiency.
    e_star_all_ap = ph_source_all_ap

    e_sky = get_sky_bkgnd(coords, exptime)

    # Array of possible aperture sizes: [1,2,...,max_N_ap]
    N_pix_aper = np.array(range(1,len(CFF)+1))

    e_sky_all_ap = e_sky[:, None] * N_pix_aper[None, :]

    ##########################################################################
    # Using the analytic N_pix(T_mag) given to the TSWG by Jon Jenkins, find #
    # the resulting standard deviation in the counts in the aperture.        #
    ##########################################################################

    # N_pix_sel = N_pix_in_aperture_Jenkins(T_mags)
    # use sullivan's instead
    N_pix_sel = N_pixels_in_aperture_Sullivan(T_mags)
    N_pix_sel[(N_pix_sel < 3) | (N_pix_sel > 50)] = 3
    # leave N_pix_sel as float, for smooth display at the end
    N_pix_sel = np.round(
            np.maximum(3*np.ones_like(N_pix_sel),N_pix_sel)).astype(int)

    assert np.max(N_pix_sel) < np.max(N_pix_aper), \
            'maximum aperture size is 17px squared'

    # Indices in the dimension over all possible aperture sizes that correspond to
    # the desired number of pixels in the aperture.
    sel_inds = np.round(N_pix_sel).astype(int) - 1

    # Report the noise and number of pixels for the selected aperture size.
    e_star_sel_ap = []
    e_sky_sel_ap = []
    for ix, sel_ind in enumerate(sel_inds):
        if sel_ind > 50 or sel_ind < 2:
            sel_ind = 2
        e_star_sel_ap.append(e_star_all_ap[ix,sel_ind])
        e_sky_sel_ap.append(e_sky_all_ap[ix,sel_ind])
    e_star_sel_ap = np.array(e_star_sel_ap)
    e_sky_sel_ap = np.array(e_sky_sel_ap)

    noise_star_sel_ap = np.sqrt(e_star_sel_ap) / e_star_sel_ap

    noise_sky_sel_ap = np.sqrt(N_pix_sel * e_sky_sel_ap) / e_star_sel_ap

    noise_ro_sel_ap = np.sqrt(N_pix_sel * N_exposures) * e_pix_ro / e_star_sel_ap

    noise_sys_sel_ap = np.zeros_like(e_star_sel_ap) \
                       + sys_limit / 1e6 / np.sqrt(exptime / 3600.)

    noise_sel_ap = np.sqrt(noise_star_sel_ap ** 2. +
                           noise_sky_sel_ap ** 2. +
                           noise_ro_sel_ap ** 2. +
                           noise_sys_sel_ap ** 2.)

    return np.array(
            [N_pix_sel,
             noise_sel_ap,
             noise_star_sel_ap,
             noise_sky_sel_ap,
             noise_ro_sel_ap,
             noise_sys_sel_ap]
            )



if __name__ == '__main__':

    # Produce a csv file with tabulated values of the noise model.

    T_mags = np.arange(4,16+0.05,0.05)

    # RA, dec. (90, -66) is southern ecliptic pole
    good_coords = np.array([90*np.ones_like(T_mags), -66*np.ones_like(T_mags)]).T

    # Towards galactic center.
    bad_coords = np.array([266.25*np.ones_like(T_mags), -28.94*np.ones_like(T_mags)]).T

    for name, coords in zip(['good', 'bad'], [good_coords, bad_coords]):

        out = noise_model(T_mags, coords=coords, exptime=3600)

        df = pd.DataFrame({
                'N_pix':out[0,:],
                'T_mag':T_mags,
                'noise':out[1,:],
                'noise_star':out[2,:],
                'noise_sky':out[3,:],
                'noise_ro':out[4,:],
                'noise_sys':out[5,:]
                })

        outdir = os.path.join(resultsdir, 'tess_noise_model')
        if not os.path.exists(outdir): os.mkdir(outdir)
        outpath = os.path.join(outdir, f'selected_noise_model_{name:s}_coords.csv')
        df.to_csv(outpath, index=False, float_format='%.4g')

        plt.close('all')
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(df.T_mag, 1e6*df.noise, c='k', label='total')
        ax.plot(df.T_mag, 1e6*df.noise_star, c='C0', label='shot noise', lw=0.5)
        ax.plot(df.T_mag, 1e6*df.noise_sky, c='C1', label='sky', lw=0.5)
        ax.plot(df.T_mag, 1e6*df.noise_ro, c='C2', label='readout', lw=0.5)
        ax.plot(df.T_mag, 1e6*df.noise_sys, c='C3', label='sys', lw=0.5)
        ax.legend(loc='best', fontsize='xx-small')
        ax.grid(ls='--', lw=0.5)
        ax.update({
            'xlabel': 'T [mag]',
            'ylabel': 'σ [ppm hr$^{1/2}$]',
            'yscale': 'log',
            'ylim': [1,1e4]
        })
        outpath = os.path.join(outdir, f'smex_noise_model_{name:s}.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
