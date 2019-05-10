"""
First, run drivers/make_fficut_wget_script.py
Then, run this.
"""
from glob import glob
import datetime, os, pickle, shutil
import numpy as np, pandas as pd

from numpy import array as nparr
from astropy.io import fits
from astropy import wcs
from datetime import datetime

from cdips.lcproc import mask_orbit_edges as moe

from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord

from transitleastsquares import transit_mask
from astrobase import lcmath

DEBUG = False

def make_wget_script(tfasrdir, xlen_px=10, ylen_px=10, tesscutvernum=0.1):
    """
    From astrocut/TESScut API, make shell script used to get FFI cutouts for
    all the TCEs.  Example curl line ::

        curl -O "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=345.3914&dec=-22.0765&y=15&x=10&units=px&sector=All"

    the corresponding driver is drivers/fficut_wget_script.py

    ############
    Args:

    tfasrdir: str
        The directory with signal-reconstructed TFA lightcurves for TCEs, e.g.,
        "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6_TFA_SR"

    """

    lcpaths = glob(os.path.join(tfasrdir,'*.fits'))

    outlines = []
    ras, decs= [], []
    for lcpath in lcpaths:

        hdul = fits.open(lcpath)
        hdr = hdul[0].header

        try:
            ra = hdr["RA[deg]"]
            dec = hdr["Dec[deg]"]
        except:
            ra = hdr["RA_OBJ"]
            dec = hdr["DEC_OBJ"]

        outtemplate = (
        '\ncurl -O "https://mast.stsci.edu/tesscut/api/'
        'v{tesscutvernum:.1f}'
        '/astrocut?'
        'ra={ra:.4f}&dec={dec:.4f}&y={ylen_px:d}&x={xlen_px:d}'
        '&units=px&sector=All"'
        )

        outline = outtemplate.format(
            tesscutvernum=tesscutvernum,
            ra=ra,
            dec=dec,
            ylen_px=ylen_px,
            xlen_px=xlen_px
        )

        outlines.append(outline)
        ras.append(ra)
        decs.append(dec)

    outlines.insert(0,'#!/usr/bin/env bash\n')

    outdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_cutouts'
    outdir = os.path.join(outdir,tfasrdir.rstrip('/').split('/')[-1])
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(outdir, 'wget_the_TCE_cutouts.sh')
    with open(outpath, 'w') as f:
        f.writelines(outlines)
    print('made {}'.format(outpath))

    outpath = os.path.join(outdir, 'ra_dec_to_lcpath_dict.csv')
    df = pd.DataFrame({'lcpath':lcpaths,'ra':ras,'dec':decs})
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def _match_lcs_and_cutouts(df, sectornum, cutdir):

    szfill = str(sectornum).zfill(4)

    globs = []
    for ra,dec in zip(nparr(df['ra']),nparr(df['dec'])):

        rastr = '{:.4f}'.format(ra)
        decstr = '{:.4f}'.format(dec)

        globs.append(
                'tess-s{snum:s}-?-?_{ra:s}_{dec:s}_10x10_astrocut.fits'.
                format(snum=szfill,ra=rastr.rstrip('0'),dec=decstr.rstrip('0'))
        )

    globpaths = [os.path.join(cutdir,g) for g in globs]
    globpathexists = nparr([len(glob(g)) for g in globpaths]).astype(bool)

    if not np.all(globpathexists):
        raise AssertionError('not np.all(globpathexists)')

    df['cutpath'] = [glob(g)[0] for g in globpaths]

    return df


def initialize_centroid_analysis(
    sectornum,
    cutdir = "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_cutouts/temp/"
):
    """
    drivers/make_fficut_wget_script.py needs to have been run manually before
    this, in order to download the TESSCut FFI cutouts for each TCE.

    Args:

        sectornum (int) : sector number

        cutdir (str) : path to directory full of files like

            tess-s0006-1-1_84.0898_-2.1764_10x10_astrocut.fits

        which are extracted from the wget script referenced above.

    Returns:

        mdf (pd.DataFrame) : dataframe with lcpath (to lightcurve), cutpath (to
        FFI cutout), and the TLS ephemeris used to do centroid analyses.
    """

    fficutpaths = os.path.join(cutdir, 'tess*.fits')

    dfpath = os.path.join(cutdir, 'ra_dec_to_lcpath_dict.csv')
    df = pd.read_csv(dfpath)

    # df now contains lcpath,ra,dec,cutpath
    df = _match_lcs_and_cutouts(df, sectornum, cutdir)
    # e.g.,
    # /nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6_TFA_SR/3217072174202699264_llc.fits
    df['source_id'] = nparr(list(map(
        lambda x: x.split('/')[-1].split('_llc')[0],
        nparr(df['lcpath'])
    ))).astype(np.int64)

    # will match in the period-finding information, and do the image averaging.
    # pfdf has: source_id,ls_period,ls_fap,tls_period,tls_sde,
    #           tls_t0,tls_depth,tls_duration,xcc,ycc,ra,dec
    periodfinddir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'cdips_lc_periodfinding/sector-{}'.format(sectornum)
    )
    pfpath = os.path.join(periodfinddir,'initial_period_finding_results.csv')
    pfdf = pd.read_csv(pfpath, sep=',')
    pfdf['source_id'] = pfdf['source_id'].astype(np.int64)

    mdf = df.merge(pfdf, how='left', on='source_id')
    mdfpath = os.path.join(cutdir, 'merged_cutpaths_and_periodogram_info.csv')
    mdf.to_csv(mdfpath,index=False)
    print('saved {}'.format(mdfpath))

    if np.any(pd.isnull(mdf['tls_sde'])):
        if DEBUG:
            pass
        else:
            errmsg = 'TLS SDE is null, and it should not be'
            raise AssertionError(errmsg)

    if DEBUG:
        mdf = mdf[~pd.isnull(mdf['tls_sde'])]

    return mdf


def do_centroid_analysis(sectornum=6):

    mdf = initialize_centroid_analysis(sectornum)

    for t0,per,dur,lcpath,cutpath,sourceid in zip(
        nparr(mdf['tls_t0']),
        nparr(mdf['tls_period']),
        nparr(mdf['tls_duration']),
        nparr(mdf['lcpath']),
        nparr(mdf['cutpath']),
        nparr(mdf['source_id'])
    ):

        outd = measure_centroid(t0,per,dur,lcpath,cutpath,sourceid)


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


def measure_centroid(t0,per,dur,lcpath,cutpath,sourceid):
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

        lcpath, cutpath: Path to lightcurve, path to FFI cutout.

        sourceid (int)

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

    cuthdul = fits.open(cutpath)
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

        try:
            assert len(thistra_intra_time) == len(thistra_oot_time)
        except AssertionError:
            print('fix assertion error...')
            import IPython; IPython.embed()

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
        'intra_imgs_flux':intra_imgs_flux,
        'oot_imgs_flux':oot_imgs_flux,
        'intra_imgs_flux_err':intra_imgs_flux_err,
        'oot_imgs_flux_err':oot_imgs_flux_err,
        'delta_ctd_arcsec':delta_ctd_arcsec,
        'delta_ctd_err_arcsec':delta_ctd_err_arcsec,
        'delta_ctd_sigma':delta_ctd_arcsec/delta_ctd_err_arcsec
    }

    outpath = os.path.join(
        os.path.dirname(cutpath),
        os.path.basename(cutpath).replace(
            '.fits','{}_ctds.pkl'.format(str(sourceid)))
    )
    with open(outpath,'wb') as f:
        pickle.dump(outdict, f)
    print('made {}'.format(outpath))

    return outdict

