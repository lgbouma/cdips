"""
called from drivers/make_vetting_multipg_pdf.py
"""
from glob import glob
import datetime, os, pickle, shutil, requests
import numpy as np, pandas as pd
import time as pytime

from numpy import array as nparr
from astropy.io import fits
from astropy import wcs
from datetime import datetime

from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord

from transitleastsquares import transit_mask
from astrobase import lcmath

from astroquery.mast import Tesscut

DEBUG = False

def _get_desired_oot_times(time, thistra_time):

    # need a correspnding set of oot points. (ideally) half before, half
    # after the transit.
    N_pts_needed = len(thistra_time)
    N_pts_before = int(np.floor(N_pts_needed/2))
    N_pts_after = int(np.ceil(N_pts_needed/2))

    thistra_ind_start = int(np.argwhere( time == np.min(thistra_time) ))
    thistra_ind_end = int(np.argwhere( time == np.max(thistra_time) ))

    # start say at least 6 cadences (3hr) away from transit, in case of bad
    # duration estimate
    desired_before_transit_start = thistra_ind_start - 6 - N_pts_before
    desired_before_transit_end = thistra_ind_start - 6

    desired_after_transit_start = thistra_ind_end + 6
    desired_after_transit_end = thistra_ind_end + 6 + N_pts_after

    desired_before_oot_time = time[
        desired_before_transit_start:desired_before_transit_end]
    desired_after_oot_time = time[
        desired_after_transit_start:desired_after_transit_end]

    # if there are Tdur/2 points available both before and after, we're
    # golden.
    if len(desired_before_oot_time)>0 and len(desired_after_oot_time)>0:
        pass
    elif len(desired_before_oot_time)==0:
        # just do all points after instead
        desired_after_transit_end = thistra_ind_end + 6 + 2*N_pts_after
        desired_after_oot_time = time[
            desired_after_transit_start:desired_after_transit_end]
    elif len(desired_after_oot_time)==0:
        # just do all points before instead
        desired_before_transit_start = thistra_ind_start - 6 - 2*N_pts_before
        desired_before_oot_time = time[
            desired_before_transit_start:desired_before_transit_end]

    desired_oot_times = np.concatenate((desired_before_oot_time,
                                        desired_after_oot_time))
    return desired_oot_times


def compute_centroid_from_first_moment(data):

    total = np.sum(data)

    # indices[0] is a range(0,N+spatial) column vector
    # indices[1] is a range(0,N+spatial) row vector
    indices = np.ogrid[[slice(0, i) for i in data.shape]]

    # note the output array is reversed to give (x, y) order
    ctds =  np.array([np.sum(indices[axis] * data) / total
                     for axis in range(data.ndim)])[::-1]

    return ctds


def measure_centroid(t0,per,dur,sector,sourceid,c_obj,outdir):
    """
    Quoting Kostov+2019, who I followed:

    '''
    1) create the mean in-transit and out-of-transit images for each transit
    (ignoring cadences with non-zero quality flags), where the latter are based
    on the same number of exposure cadences as the former, split evenly before
    and after the transit;

    2) calculate the overall mean in-transit and out-of-transit images by
    averaging over all transits;

    3) subtract the overall mean out-of-transit image from the overall
    in-transit image to produce the overall mean difference image; and

    4) measure the center-of-light for each difference
    and out-of-transit image by calculating the corresponding x- and y-moments
    of the image. The measured photocenters for the three planet candidates are
    shown in Figures 9, 10, and 11, and listed in Table 3. We detect no
    significant photocenter shifts between the respective difference images and
    out-of-transit images for any of the planet candidates, which confirms that
    the targets star is the source of the transits.
    '''

    args:

        t0,per,dur : float, days. Epoch, period, duration.  Used to isolate
        transit windows.

        sector (int)

        sourceid (int)

        c_obj (SkyCoord): location of target star

    returns:

        outdict = {
            'm_oot_flux':m_oot_flux, # mean OOT image
            'm_oot_flux_err':m_oot_flux_err, # mean OOT image uncert
            'm_intra_flux':m_intra_flux, # mean in transit image
            'm_intra_flux_err':m_intra_flux_err,  # mean in transit image uncert
            'm_oot_minus_intra_flux':m_oot_minus_intra_flux, # mean OOT - mean intra
            'm_oot_minus_intra_flux_err':m_oot_minus_intra_flux_err,
            'm_oot_minus_intra_snr':m_oot_minus_intra_snr,
            'ctds_intra':ctds_intra, # centroids of all transits
            'ctds_oot':ctds_oot, # centroids of all ootransits
            'ctds_oot_minus_intra':ctds_oot_minus_intra, # centroids of diff
            'm_ctd_intra':m_ctd_intra, # centroid of mean intransit image
            'm_ctd_oot':m_ctd_oot,
            'intra_imgs_flux':intra_imgs_flux,
            'oot_imgs_flux':oot_imgs_flux,
            'intra_imgs_flux_err':intra_imgs_flux_err,
            'oot_imgs_flux_err':oot_imgs_flux_err
        }
    """

    print('beginning tesscut for {}'.format(repr(c_obj)))
    try:
        cuthdul = Tesscut.get_cutouts(c_obj, size=10, sector=sector)
    except (requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError) as e:
        print('got {}, try again'.format(repr(e)))
        pytime.sleep(30)
        try:
            cuthdul = Tesscut.get_cutouts(c_obj, size=10, sector=sector)
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError) as e:

            print('ERR! sourceid {} FAILED TO GET TESSCUTOUT'.format(sourceid))
            return None

    if len(cuthdul) != 1:
        print('ERR! sourceid {} GOT {} CUTOUTS'.format(sourceid, len(cuthdul)))
        return None
    else:
        cuthdul = cuthdul[0]
    data, data_hdr = cuthdul[1].data, cuthdul[1].header
    cutout_wcs = wcs.WCS(cuthdul[2].header)

    # flux and flux_err are image cubes of (time x spatial x spatial)
    quality = data['QUALITY']

    flux = data['FLUX']
    flux_err = data['FLUX_ERR']
    time = data['TIME'] # in TJD
    time += 2457000 # now in BJD

    time = time[quality == 0]
    flux = flux[quality == 0]
    flux_err = flux_err[quality == 0]

    intra = transit_mask(time, per, dur, t0)

    mingap = per/2
    ngroups, groups = lcmath.find_lc_timegroups(time[intra], mingap=mingap)

    oot_times = time[~intra]

    # make mean in-transit and out-of-transit images & uncertainty maps for
    # each transit
    intra_imgs_flux, oot_imgs_flux = [], []
    intra_imgs_flux_err, oot_imgs_flux_err = [], []
    for group in groups:

        thistra_intra_time = time[intra][group]
        thistra_intra_flux = flux[intra][group]
        thistra_intra_flux_err = flux_err[intra][group]

        thistra_oot_time = _get_desired_oot_times(time, thistra_intra_time)
        thistra_oot_flux = flux[np.in1d(time, thistra_oot_time)]
        thistra_oot_flux_err = flux_err[np.in1d(time, thistra_oot_time)]

        intra_imgs_flux.append(
            np.mean(thistra_intra_flux, axis=0)
        )
        intra_imgs_flux_err.append(
            np.mean(thistra_intra_flux_err, axis=0)
        )
        oot_imgs_flux.append(
            np.mean(thistra_oot_flux, axis=0)
        )
        oot_imgs_flux_err.append(
            np.mean(thistra_oot_flux_err, axis=0)
        )

    # shapes: (n_transits, spatial, spatial)
    intra_imgs_flux = nparr(intra_imgs_flux)
    intra_imgs_flux_err = nparr(intra_imgs_flux_err)
    oot_imgs_flux = nparr(oot_imgs_flux)
    oot_imgs_flux_err =  nparr(oot_imgs_flux_err)

    # average over transits, to get mean in-transit and out-of-transit images.
    m_intra_flux = np.mean(intra_imgs_flux,axis=0)
    m_intra_flux_err = np.mean(intra_imgs_flux_err,axis=0)
    m_oot_flux = np.mean(oot_imgs_flux,axis=0)
    m_oot_flux_err = np.mean(oot_imgs_flux_err,axis=0)

    # compute x and y centroid values for mean image
    m_ctd_intra = compute_centroid_from_first_moment(m_intra_flux)
    m_ctd_oot = compute_centroid_from_first_moment(m_oot_flux)
    ctd_m_oot_minus_m_intra = compute_centroid_from_first_moment(
        m_oot_flux - m_intra_flux)

    # compute x and y centroid values for each transit
    ctds_intra = nparr(
        [compute_centroid_from_first_moment(intra_imgs_flux[ix,:,:])
         for ix in range(intra_imgs_flux.shape[0]) ]
    )
    ctds_oot = nparr(
        [compute_centroid_from_first_moment(oot_imgs_flux[ix,:,:])
         for ix in range(oot_imgs_flux.shape[0]) ]
    )
    ctds_oot_minus_intra = nparr(
        [ compute_centroid_from_first_moment(
            oot_imgs_flux[ix,:,:]-intra_imgs_flux[ix,:,:])
          for ix in range(oot_imgs_flux.shape[0])
        ]
    )

    # make OOT - intra image. NOTE: the uncertainty map might be a bit wrong,
    # b/c you should get a sqrt(N) on the flux in the mean image. I think.
    m_oot_minus_intra_flux = m_oot_flux - m_intra_flux
    m_oot_minus_intra_flux_err = np.sqrt(
        m_oot_flux_err**2 + m_intra_flux**2
    )

    m_oot_minus_intra_snr = m_oot_minus_intra_flux / m_oot_minus_intra_flux_err

    # calculate the centroid shift amplitude, deltaC = C_oot - C_intra
    delta_ctd_px = np.sqrt(
        (m_ctd_oot[0] - m_ctd_intra[0])**2
        +
        (m_ctd_oot[1] - m_ctd_intra[1])**2
    )
    delta_ctd_arcsec = delta_ctd_px * 21 # 21arcsec/px

    # error in centroid shift amplitude: assume it is roughly the scatter in
    # OOT centroids
    delta_ctd_err_px = np.sqrt(
        np.std(ctds_oot, axis=0)[0]**2 +
        np.std(ctds_oot, axis=0)[1]**2
    )
    delta_ctd_err_arcsec = delta_ctd_err_px*21

    outdict = {
        'cutout_wcs':cutout_wcs,
        'm_oot_flux':m_oot_flux, # mean OOT image
        'm_oot_flux_err':m_oot_flux_err, # mean OOT image uncert
        'm_intra_flux':m_intra_flux, # mean in transit image
        'm_intra_flux_err':m_intra_flux_err,  # mean in transit image uncert
        'm_oot_minus_intra_flux':m_oot_minus_intra_flux, # mean OOT - mean intra
        'm_oot_minus_intra_flux_err':m_oot_minus_intra_flux_err,
        'm_oot_minus_intra_snr':m_oot_minus_intra_snr,
        'ctds_intra':ctds_intra, # centroids of all transits
        'ctds_oot':ctds_oot, # centroids of all ootransits
        'ctds_oot_minus_intra':ctds_oot_minus_intra,
        'm_ctd_intra':m_ctd_intra, # centroid of mean intransit image
        'm_ctd_oot':m_ctd_oot,
        'ctd_m_oot_minus_m_intra':ctd_m_oot_minus_m_intra,
        'intra_imgs_flux':intra_imgs_flux,
        'oot_imgs_flux':oot_imgs_flux,
        'intra_imgs_flux_err':intra_imgs_flux_err,
        'oot_imgs_flux_err':oot_imgs_flux_err,
        'delta_ctd_arcsec':delta_ctd_arcsec,
        'delta_ctd_err_arcsec':delta_ctd_err_arcsec,
        'delta_ctd_sigma':delta_ctd_arcsec/delta_ctd_err_arcsec
    }

    outpath = os.path.join(
        outdir,
        '{}_ctds.pkl'.format(str(sourceid))
    )
    with open(outpath,'wb') as f:
        pickle.dump(outdict, f)
    print('made {}'.format(outpath))

    return outdict

