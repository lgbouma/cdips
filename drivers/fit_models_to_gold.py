"""
usage:
python -u fit_models_to_gold.py &> logs/YYYYMMDD_fit_s{sector}.log &

contents:
    main
    _get_sector_metadata
    _define_and_make_directories
    _get_lcpaths
    _fit_transit_model_single_sector
    get_teff_rstar_logg
    fit_results_to_ctoi_csv
"""
#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, pickle, requests, socket
from glob import glob
import time as pytime

import matplotlib as mpl
mpl.use('Agg')

from numpy import array as nparr

from astropy.coordinates import SkyCoord
from astropy import units as u, constants as const
from astroquery.mast import Catalogs

from astrobase import imageutils as iu
from astrobase.lcfit.transits import fivetransitparam_fit_magseries
from astrobase.periodbase import htls

from cdips.utils import today_YYYYMMDD, str2bool
from cdips.utils.pipelineutils import save_status, load_status
from cdips.utils.catalogs import (
    get_cdips_catalog, get_toi_catalog, get_exofop_ctoi_catalog
)
from cdips.utils.mamajek import (
    get_interp_mass_from_rstar, get_interp_rstar_from_teff
)
from cdips.lcproc.find_planets import run_periodograms_and_detrend
import cdips.vetting.make_all_vetting_reports as mavr

##########
# CONFIG #
##########

DATADIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/data/mcmc_fitting_identifiers'

long_run_df = pd.read_csv(
    os.path.join(DATADIR, 'LONG_RUN_IDENTIFIERS.csv')
)
mcmc_fails_df = pd.read_csv(
    os.path.join(DATADIR, 'KNOWN_MCMC_FAILS.csv'), sep=';'
)
extra_detrend_df = pd.read_csv(
    os.path.join(DATADIR, 'KNOWN_EXTRA_DETREND.csv'), sep=';'
)
skip_convergence_df = pd.read_csv(
    os.path.join(DATADIR, 'SKIP_CONVERGENCE_IDENTIFIERS.csv')
)

KNOWN_EXTRA_DETREND = list(nparr(extra_detrend_df.source_id))
LONG_RUN_IDENTIFIERS = list(nparr(long_run_df.source_id))
KNOWN_MCMC_FAILS = list(nparr(mcmc_fails_df.source_id))
SKIP_CONVERGENCE_IDENTIFIERS = list(nparr(skip_convergence_df.source_id))

host = socket.gethostname()
if 'phtess' in host:
    CLASSIFXNDIR = "/home/lbouma/proj/cdips/results/vetting_classifications"
    resultsbase = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
    database = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/data/'
elif 'brik' in host:
    CLASSIFXNDIR = "/home/luke/Dropbox/proj/cdips/results/vetting_classifications"
    resultsbase = '/home/luke/Dropbox/proj/cdips/results/'
    database = '/home/luke/Dropbox/proj/cdips/data/'

###############
# main driver #
###############

def main(overwrite=0, sector=None, nworkers=40, cdipsvnum=1, cdips_cat_vnum=0.6,
         is_not_cdips_still_good=False):
    """
    ------------------------------------------
    Description:

    Fit Mandel-Agol transits to planet candidates. The fit parameters are then
    used for CTOIs.

    The goal of these models is to provide a good ephemeris (so: reliable
    epoch, period, and duration + uncertainties).

    For publication-quality parameters, joint modelling of stellar rotation
    signals along with planet signals may be preferred. The approach here is to
    just whiten the light curve in order to get a good ephemeris (which may
    distort the depth, for example).

    ------------------------------------------
    Explanation of design:

        Directory structure is made like:

            /results/fit_gold/sector-?/fitresults/hlsp_*/
            /results/fit_gold/sector-?/samples/hlsp_*/

        where each `fitresults` sub-directory contains images to diagnose fit
        quality, pickle files with saved parameters from the fits, etc.
        The `samples` sub-directories have the .h5 files used when sampling.

        First, all these directories are made.

        Then, pdf files under either
        /results/vetting_classifications/sector-?_CLEAR_THRESHOLD or
        /results/vetting_classifications/sector-?_NOT_CDIPS_STILL_GOOD
        are parsed to collect the light curves and any necessary metadata.

        Then there's a for loop for over each planet candidate, in which the
        fit is performed.

    ------------------------------------------
    Args:

        overwrite: if False, and the pickle file with saved parameters exists
        (i.e. you already fit the PC), no sampling is done.

        sector: the sector number.

        nworkers: for threading

        cdipsvnum: version number of CDIPS LCs in their name

        cdips_cat_vnum: target star catalog version identifier.

        is_not_cdips_still_good: if true, parses planet candidates from
        `/results/vetting_classifications/sector-?_NOT_CDIPS_STILL_GOOD`;
        otherwise does the CLEAR_THRESHOLD directory.

    """

    lcbasedir, resultsdir = _define_and_make_directories(
        sector, is_not_cdips_still_good=is_not_cdips_still_good)

    df, cdips_df, pfdf, supplementstatsdf, toidf, ctoidf = (
        _get_sector_metadata(sector, cdips_cat_vnum=cdips_cat_vnum)
    )

    lcpaths = _get_lcpaths(df, lcbasedir)

    for lcpath in lcpaths:

        _hdr = iu.get_header_keyword_list(lcpath, ['CAMERA','CCD'], ext=0)
        cam, ccd = _hdr['CAMERA'], _hdr['CCD']

        #
        # make fitresults and samples directories
        #
        outdirs = [
            os.path.join(resultsdir,'fitresults',lcname.replace('.fits','')),
            os.path.join(resultsdir,'samples',lcname.replace('.fits',''))
        ]
        for outdir in outdirs:
            if not os.path.exists(outdir):
                LOGINFO(f'Made {outdir}')
                os.mkdir(outdir)

        #
        # collect metadata for this target star
        #
        supprow = mavr._get_supprow(source_id, supplementstatsdf)
        suppfulldf = supplementstatsdf

        pfrow = pfdf.loc[pfdf['source_id']==source_id]
        if len(pfrow) != 1:
            errmsg = 'expected exactly 1 source match in period find df'
            raise AssertionError(errmsg)

        outpath = os.path.join(resultsdir,'fitresults',
                               lcname.replace('.fits',''),
                               lcname.replace('.fits','_fitparameters.csv'))

        #
        # if you haven't already made the output parameter file (which requires
        # convergence) then start fitting.
        #
        if not os.path.exists(outpath):

            _fit_transit_model_single_sector(lcpath, outpath, mdf,
                                             source_id, supprow, suppfulldf,
                                             pfdf, pfrow, toidf, ctoidf,
                                             sector, nworkers,
                                             cdipsvnum=cdipsvnum,
                                             overwrite=overwrite)

        else:

            status_file = os.path.join(
                os.path.dirname(outpath), 'run_status.stat'
            )
            status = load_status(status_file)

            fittype = 'fivetransitparam_fit'
            if str2bool(status[fittype]['is_converged']):
                LOGINFO('{} converged and already wrote wrote ctoi csv.'.
                        format(source_id))

            elif (
                not str2bool(status[fittype]['is_converged'])
                and int(source_id) in SKIP_CONVERGENCE_IDENTIFIERS
            ):
                LOGWARNING(
                    'WRN! {} not converged, but wrote wrote ctoi csv b/c in '
                    'SKIP_CONVERGENCE_IDENTIFIERS.'.format(source_id)
                )

            else:
                raise ValueError(
                    'got parameter file existing, but not converged.'
                    'should never happen. for DR2 {}'.format(source_id)
                )


def _get_sector_metadata(sector, cdips_cat_vnum=None, is_not_cdips_still_good=0):

    if is_not_cdips_still_good:
        classifxn_csv = os.path.join(
            CLASSIFXNDIR,
            "sector-{}_PCs_NOT_CDIPS_STILL_GOOD.csv".format(sector)
        )
    else:
        classifxn_csv = os.path.join(
            CLASSIFXNDIR,
            "sector-{}_PCs_CLEAR_THRESHOLD.csv".format(sector)
        )

    df = pd.read_csv(classifxn_csv, sep=';')

    cdips_df = get_cdips_catalog(ver=cdips_cat_vnum)

    pfpath = os.path.join(resultsbase,
              'cdips_lc_periodfinding/'
              'sector-{}/'.format(sector)+
              'initial_period_finding_results_supplemented.csv')
    pfdf = pd.read_csv(pfpath, sep=';')

    supppath = os.path.join(resultsbase,
                'cdips_lc_stats/sector-{}/'.format(sector)+
                'supplemented_cdips_lc_statistics.txt')
    supplementstatsdf = pd.read_csv(supppath, sep=';')

    toidf = get_toi_catalog()
    ctoidf = get_exofop_ctoi_catalog()

    return df, cdips_df, pfdf, supplementstatsdf, toidf, ctoidf


def _define_and_make_directories(sector, is_not_cdips_still_good=0):

    host = socket.gethostname()

    if 'phtess' in host:
        lcbase = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
    elif 'brik' in host:
        lcbase = '/home/luke/Dropbox/proj/cdips/data/'

    if is_not_cdips_still_good:
        resultsdir = os.path.join(
            resultsbase, f'fit_gold/sector-{sector}_NOT_CDIPS_STILL_GOOD'
        )
    else:
        resultsdir = os.path.join(
            resultsbase, f'fit_gold/sector-{sector}_CLEAR_THRESHOLD'
        )
    dirs = [resultsdir,
            os.path.join(resultsdir,'fitresults'),
            os.path.join(resultsdir,'samples')
           ]
    for _d in dirs:
        if not os.path.exists(_d):
            LOGINFO(f'Making {_d}...')
            os.mkdir(_d)

    lcbasedir = os.path.join(lcbase, f'CDIPS_LCS/sector-{sector}/')

    return lcbasedir, resultsdir


def _get_lcpaths(df, lcbasedir):

    # vet_hlsp_cdips_tess_ffi_gaiatwo0003125263468681400320-0006-cam1-ccd1_tess_v01_llc.pdf
    lcnames = list(map(
        lambda x: x.replace('vet_','').replace('.pdf','.fits'),
        nparr(df['Name'])
    ))

    lcglobs = [
        os.path.join(lcbasedir, 'cam?_ccd?', lcname) for lcname in lcnames
    ]

    lcpaths = []
    for lcg in lcglobs:
        if len(glob(lcg)) == 1:
            lcpaths.append(glob(lcg)[0])
        else:
            raise ValueError('got more than 1 lc matching glob {}'.format(lcg))

    return lcpaths


def _fit_transit_model_single_sector(lcpath, outpath, mdf,
                                     source_id, supprow, suppfulldf, pfdf,
                                     pfrow, toidf, ctoidf, sector, nworkers,
                                     cdipsvnum=1, overwrite=1):
    try_mcmc = True
    identifier = source_id

    #
    # get time & flux for detrended light curve. follows procedure from
    # do_initial_period_finding.
    #
    APNAME = 'PCA1'
    source_id, time, mag, xcc, ycc, ra, dec, _, _ = (
        lcu.get_lc_data(lcpath, mag_aperture=APNAME)
    )

    dtr_method, break_tolerance, window_length = 'best', 0.5, 0.5
    dtr_dict = {'method':dtr_method,
                'break_tolerance':break_tolerance,
                'window_length':window_length}

    r, time, flux, dtr_stages_dict = run_periodograms_and_detrend(
        source_id, time, mag, dtr_dict, return_extras=True
    )

    #
    # define the paths. get the stellar parameters, and do the fit!
    #
    fit_savdir = os.path.dirname(outpath)
    chain_savdir = os.path.dirname(outpath).replace('fitresults', 'samples')

    try:
        teff, teff_err, rstar, rstar_err, logg, logg_err = (
            get_teff_rstar_logg(hdr)
        )
    except (NotImplementedError, ValueError) as e:
        LOGERROR(e)
        LOGERROR('did not get rstar for {}. MUST MANUALLY FIX.'.
                 format(source_id))
        try_mcmc = False

    #
    # initialize status file
    #
    status_file = os.path.join(fit_savdir, 'run_status.stat')
    fittype = 'fivetransitparam_fit'
    if not os.path.exists(status_file):
        save_status(
            status_file, fittype,
            {'is_converged':False, 'n_steps_run':0}
        )
    status = load_status(status_file)[fittype]

    #
    # if not converged and no steps previously run:
    #   run 4k steps. write status file.
    # 
    # reload status file.
    # if not converged and 4k steps previously run and in long ID list:
    #   run 25k steps, write status file.
    #
    # reload status file.
    # if not converged:
    #   print a warning.
    #
    if identifier in KNOWN_MCMC_FAILS:
        LOGWARNING('WRN! identifier {} requires manual fixing.'.
              format(identifier))
        try_mcmc = False

    if (
        not str2bool(status['is_converged'])
        and int(status['n_steps_run']) == 0
        and try_mcmc
    ):

        n_mcmc_steps = 4000

        mafr, tlsr, is_converged = fivetransitparam_fit_magseries(
            time, flux, err, teff, rstar, logg, identifier, fit_savdir,
            chain_savdir, n_mcmc_steps=n_mcmc_steps,
            overwriteexistingsamples=False, n_transit_durations=5,
            make_tlsfit_plot=True, exp_time_minutes=30, bandpass='tess',
            magsarefluxes=True, nworkers=nworkers
        )

        status = {'is_converged':is_converged, 'n_steps_run':n_mcmc_steps}
        save_status(status_file, fittype, status)

    status = load_status(status_file)[fittype]
    if (
        not str2bool(status['is_converged'])
        and int(status['n_steps_run']) != 25000
        and int(identifier) in LONG_RUN_IDENTIFIERS
        and try_mcmc
    ):

        n_mcmc_steps = 25000

        # NOTE: hard-code nworkers, since we dont get multithreading
        # improvement anyway (this is some kind of bug)
        mafr, tlsr, is_converged = fivetransitparam_fit_magseries(
            time, flux, err, teff, rstar, logg, identifier, fit_savdir,
            chain_savdir, n_mcmc_steps=n_mcmc_steps,
            overwriteexistingsamples=True, n_transit_durations=5,
            make_tlsfit_plot=True, exp_time_minutes=30, bandpass='tess',
            magsarefluxes=True, nworkers=4
        )

        status = {'is_converged':is_converged, 'n_steps_run':n_mcmc_steps}
        save_status(status_file, fittype, status)

    #
    # if converged or in the list of IDs for which its fine to skip convegence
    # (because by-eye, the fits are converged), convert fit results to ctoi csv
    # format
    #
    status = load_status(status_file)[fittype]

    if (
        str2bool(status['is_converged'])
        or int(identifier) in SKIP_CONVERGENCE_IDENTIFIERS
    ):

        try:
            _ = isinstance(mafr, dict)
        except UnboundLocalError:
            #
            # get the MCMC results from the pickle file; regenerate the TLS
            # result.
            #

            fitparamdir = os.path.dirname(status_file)
            fitpklsavpath = os.path.join(
                fitparamdir,
                '{}_phased_fivetransitparam_fit_empiricalerrs.pickle'.
                format(identifier)
            )
            with open(fitpklsavpath, 'rb') as f:
                mafr = pickle.load(f)

            tlsp = htls.tls_parallel_pfind(time, flux, err, magsarefluxes=True,
                                           tls_rstar_min=0.1, tls_rstar_max=10,
                                           tls_mstar_min=0.1,
                                           tls_mstar_max=5.0, tls_oversample=8,
                                           tls_mintransits=1,
                                           tls_transit_template='default',
                                           nbestpeaks=5, sigclip=None,
                                           nworkers=nworkers)
            tlsr = tlsp['tlsresult']

        ticid = int(hdr['TICID'])
        ra, dec = hdr['RA_OBJ'], hdr['DEC_OBJ']
        LOGINFO('{} converged. writing ctoi csv.'.format(identifier))
        fit_results_to_ctoi_csv(ticid, ra, dec, mafr, tlsr, outpath, toidf,
                                ctoidf, teff, teff_err, rstar, rstar_err, logg,
                                logg_err, cdipsvnum=cdipsvnum)
    else:
        LOGWARNING('WRN! {} did not converge, after {} steps. MUST MANUALLY FIX.'.
                   format(identifier, status['n_steps_run']))


def get_teff_rstar_logg(hdr):
    """
    Given CDIPS header, acquire estimates of Teff, Rstar, logg from TICv8. If
    Teff fails, go with the Gaia DR2 Teff.  If Rstar fails, go with Gaia DR2
    Rstar.  If Rstar still fails, use Teff and Mamajek relation to
    interpolate Rstar.

    If logg fails, but you have Gaia DR2 Rstar, then go from Rstar to Mstar
    using Mamajek relation, and combine to estimate a ratty logg.
    """
    identifier = hdr['TICID']
    ra, dec = hdr['RA_OBJ'], hdr['DEC_OBJ']
    targetcoord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    radius = 10.0*u.arcsec
    try:
        stars = Catalogs.query_region(
            "{} {}".format(float(targetcoord.ra.value),
                           float(targetcoord.dec.value)),
            catalog="TIC",
            radius=radius
        )
    except requests.exceptions.ConnectionError:
        LOGERROR('ERR! TIC query failed. trying again...')
        pytime.sleep(30)
        stars = Catalogs.query_region(
            "{} {}".format(float(targetcoord.ra.value),
                           float(targetcoord.dec.value)),
            catalog="TIC",
            radius=radius
        )

    Tmag_pred = (
        hdr['phot_g_mean_mag']
        - 0.00522555 * (hdr['phot_bp_mean_mag'] - hdr['phot_rp_mean_mag'])**3
        + 0.0891337 * (hdr['phot_bp_mean_mag'] - hdr['phot_rp_mean_mag'])**2
        - 0.633923 * (hdr['phot_bp_mean_mag'] - hdr['phot_rp_mean_mag'])
        + 0.0324473
    )
    Tmag_cutoff = 1.2

    selstars = stars[np.abs(stars['Tmag'] - Tmag_pred)<Tmag_cutoff]

    if len(selstars)>=1:

        seltab = selstars[np.argmin(selstars['dstArcSec'])]

        if not int(seltab['GAIA']) == int(hdr['GAIA-ID']):
            raise ValueError(
                'TIC result doesnt match hdr target gaia-id. '+
                'Should just instead query selstars by Gaia ID you want.'
            )

        teff = seltab['Teff']
        teff_err = seltab['e_Teff']
        logg = seltab['logg']
        logg_err = seltab['e_logg']
        rstar = seltab['rad']
        rstar_err = seltab['e_rad']

        if type(teff)==np.ma.core.MaskedConstant:
            LOGWARNING('WRN! TIC teff nan. why? trying gaia value...')
            teff = hdr['teff_val']
            teff_err = 100

        if type(rstar)==np.ma.core.MaskedConstant:

            LOGWARNING('WRN! TIC rstar nan. why? trying gaia value...')
            rstar = hdr['radius_val']

            if rstar == 'NaN':
                LOGWARNING('WRN! Gaia rstar also nan. Trying to interpolate from '
                           'Teff...')

                if teff == 'NaN':
                    raise NotImplementedError(
                        'need rstar somehow. didnt get for {}'.
                        format(identifier)
                    )

                rstar = get_interp_rstar_from_teff(teff)

            if rstar != 'NaN':
                # given rstar, get mass, so that you can get logg
                rstar_err = 0.3*rstar
                mstar = get_interp_mass_from_rstar(rstar)

            if not isinstance(rstar, float):
                raise NotImplementedError('got unexpected value for rstar! '
                                          'manual debug required')

            _Mstar = mstar*u.Msun
            _Rstar = rstar*u.Rsun

            if type(logg)==np.ma.core.MaskedConstant:
                logg = np.log10((const.G * _Mstar / _Rstar**2).cgs.value)
                logg_err = 0.3*logg

    else:
        raise ValueError('bad xmatch for {}'.format(hdr['GAIA-ID']))

    return teff, teff_err, rstar, rstar_err, logg, logg_err


def fit_results_to_ctoi_csv(ticid, ra, dec, mafr, tlsr, outpath, toidf, ctoidf,
                            teff, teff_err, rstar, rstar_err, logg, logg_err,
                            cdipsvnum=1):
    """
    args:

        mafr: mandel-agol fit results dict from
        fit_phased_transit_mandelagol_and_line

        outpath: csv to save to

        toidf / ctoidf: dataframes of the TOI and cTOI lists.

    ##########################################

    FORMAT IS:

    \ Naming convention: params_planet_YYYYMMDD_nnn.txt
    \ nnn is a number (001-999) differentiating files uploaded on the same day
    \
    \ Column headers:  (*required)
    \
    \ Parameter value is required if uncertainty value is present
    \
    \
    \ *target = TOI name (must be existing) or CTOI name (can be new or existing)
    \           example TOI name = TOI125.01, TOI125.02 (must include "TOI" prefix and no spaces)
    \           example CTOI name = TIC52368076.01, TIC52368076.02 (must include "TIC" prefix and no spaces)
    \
    \ *flag = flag indicating type of upload. valid values are:
    \           newctoi - for creating a new CTOI and adding associated planet parameters
    \           newparams - for adding new planet parameters to an existing CTOI or TOI
    \
    \ *disp = disposition (CP = confirmed planet, FP = false positive, or PC = planet candidate)
    \
    \ period = orbital period (days)
    \ period_unc = orbital period uncertainty
    \ epoch = transit midpoint (BJD) - must be greater than 2000000.0
    \ epoch_unc = transit midpoint uncertainty
    \ depth = transit depth (ppm)
    \ depth_unc = transit depth uncertainty
    \ duration = transit duration (hrs)
    \ duration_unc = transit duration uncertainty
    \ inc = inclination (degrees)
    \ inc_unc = inclination uncertainty
    \ imp = impact parameter
    \ imp_unc = impact parameter uncertainty
    \ r_planet = R_planet/R_star
    \ r_planet_unc = R_planet/R_star uncertainty
    \ ar_star = a/R_star
    \ ar_star_unc = a/R_star uncertainty
    \ radius = radius (R_Earth)
    \ radius_unc = radius uncertainty
    \ mass = mass (M_Earth)
    \ mass_unc = mass uncertainty
    \ temp = equilibrium temperature (K)
    \ temp_unc = equilibrium temperature uncertainty
    \ insol = insolation flux (flux_Earth)
    \ insol_unc = insolation flux uncertainty
    \ dens = fitted stellar density (g/cm^3)
    \ dens_unc = fitted stellar density uncertainty
    \ sma = semi-major axis (AU)
    \ sma_unc = semi-major axis uncertainty
    \ ecc = eccentricity
    \ ecc_unc = eccentricity uncertainty
    \ arg_peri = argument of periastron
    \ arg_peri_unc = argument of periastron uncertainty
    \ time_peri = time of periastron (BJD) -  must be greater than 2000000.0
    \ time_peri_unc = time of periastron uncertainty
    \ vsa = velocity semi-amplitude (m/s)
    \ vsa_unc = velocity semi-amplitude uncertainty
    \ *tag = data tag number or name (e.g. YYYYMMDD_username_description_nnnnn)
    \ group = group name
    \ *prop_period = proprietary period in months (0 or 12) 
    \                Must be 0 if not associated with a group
    \                Must be 0 for new CTOIs
    \ *notes = notes about the values - maximum 120 characters
    \          (only required if new CTOI)
    """

    extranote = ''

    def _get_unc(param_errs, key):
        unc = np.mean([param_errs['std_perrs'][key],
                       param_errs['std_merrs'][key]])
        return unc

    #
    # calculate paramaters required for write out. there is some linear error
    # propgataion done here. see math in /doc/2019-06-27_transit_param_errs.pdf
    #

    # 'incl', 'period', 'rp', 'sma', 't0', 'u_linear', 'u_quad'
    params =  mafr['fitinfo']['finalparams'].keys()
    param_values =  mafr['fitinfo']['finalparams']
    param_errs =  mafr['fitinfo']['finalparamerrs']

    period = param_values['period']
    period_unc = _get_unc(param_errs,'period')
    epoch = param_values['t0']
    epoch_unc = _get_unc(param_errs,'t0')

    # calculate radius in earth radii; uncertainty from propagating errors of
    # Rp/Rstar, and assuming uncorrelated.
    rp_rs = param_values['rp']
    rp_rs_unc = _get_unc(param_errs,'rp')

    radius = ((param_values['rp'] * rstar)*u.Rsun).to(u.Rearth).value
    radius_unc = radius * np.sqrt((rp_rs_unc/rp_rs)**2 + (rstar_err/rstar)**2)
    if type(rstar_err)==np.ma.core.MaskedConstant:
        radius_unc = radius * 0.3  # assign 30% relative uncertainty if nan...
        extranote += 'TIC Rstar nan.'

    # calculate depth and uncertainty in ppm. depth is not the measured
    # quantity -- it is just (by definition) (Rp/Rstar)^2.
    depth = 1e6 * rp_rs**2
    depth_unc = depth * np.sqrt(2 * (rp_rs_unc/rp_rs)**2)

    a_rstar = param_values['sma']
    a_rstar_unc = _get_unc(param_errs,'sma')
    incl = param_values['incl']
    incl_unc = _get_unc(param_errs,'incl')

    # see https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
    cosi = np.cos(np.deg2rad(incl))
    cosi_unc = np.abs( np.sin(np.deg2rad(incl)) * np.deg2rad(incl_unc) )
    sini = np.sin(np.deg2rad(incl))
    sini_unc = np.abs( np.cos(np.deg2rad(incl)) * np.deg2rad(incl_unc) )

    b = a_rstar * cosi
    b_unc = b * np.sqrt(
        (a_rstar_unc/a_rstar)**2 + (cosi_unc/cosi)**2
    )

    # durations. see winn (2010) chapter, eq 14 for base equation.
    # 2019/06/26.[0-1] in /doc/ for derivation
    period_unc = _get_unc(param_errs,'period')

    T_tot = (period/np.pi * np.arcsin(
        (1/a_rstar) *
        np.sqrt( (1 + rp_rs )**2 - b**2 )
        / np.sin(np.deg2rad(incl))
    )*u.day).to(u.hour).value

    z = np.sqrt( (1 + rp_rs )**2 - b**2 )
    z_unc = 0.5 * np.sqrt(
        2*(rp_rs_unc/rp_rs)**2 + 2*(b_unc/b)**2
    )

    y = (
        (1/a_rstar) *
        np.sqrt( (1 + rp_rs )**2 - b**2 )
        / np.sin(np.deg2rad(incl))
    )
    y_unc = y * np.sqrt(
        (a_rstar_unc/a_rstar)**2 + (sini_unc/sini)**2 + (z_unc/z)**2
    )

    arcsiny = np.arcsin(y)
    arcsiny_unc = y_unc / np.sqrt( 1 - y**2 )

    T_tot_unc = T_tot / np.pi * np.sqrt(
        (period_unc/period)**2 +
        (arcsiny_unc/arcsiny)**2
    )

    #
    # determine target / flag / disp
    # *target = TOI name (must be existing) or CTOI name (can be new or existing)
    #           example TOI name = TOI125.01, TOI125.02 (must include "TOI" prefix and no spaces)
    #           example CTOI name = TIC52368076.01, TIC52368076.02 (must include "TIC" prefix and no spaces)
    #
    # *flag = flag indicating type of upload. valid values are:
    #           newctoi - for creating a new CTOI and adding associated planet parameters
    #           newparams - for adding new planet parameters to an existing CTOI or TOI
    #
    # *disp = disposition (CP = confirmed planet, FP = false positive, or PC = planet candidate)
    #

    # by default, assume new ctoi
    ctoiname = 'TIC{}.01'.format(ticid)
    target = ctoiname
    flag = 'newctoi'
    disp = 'PC'
    #
    # check MIT TSO TOI list for whether there are matches by TIC ID.
    # for disposition, take whatever the MIT TSO labelled as truth.
    #
    sel = nparr(toidf['TIC']==ticid)
    if len(toidf[sel]) == 1:
        tdf = toidf[sel]
        toiname = tdf['Full TOI ID'].iloc[0]
        target = toiname
        flag = 'newparams'
        disp = tdf['TOI Disposition'].iloc[0]

    elif len(toidf[sel]) > 1:
        # get match within 0.5 days
        LOGWARNING('WRN! MORE THAN 1 TOI')
        sel &= np.isclose(
            period,
            nparr(toidf[toidf['TIC']==ticid]['Orbital Period Value']),
            atol=0.5
        )
        if len(toidf[sel]) > 1:
            raise NotImplementedError('got 2 TOIs within 0.5d of obsd period')
        tdf = toidf[sel]
        toiname = tdf['Full TOI ID'].iloc[0]
        target = toiname
        flag = 'newparams'
        disp = tdf['TOI Disposition'].iloc[0]

    #
    # check EXOFOP-TESS CTOI list for matches by TIC ID
    #
    sel = nparr(ctoidf['TIC ID']==ticid)
    if len(ctoidf[sel]) == 1:
        tdf = ctoidf[sel]
        ctoiname = tdf['CTOI'].iloc[0]
        target = 'TIC'+str(ctoiname)
        flag = 'newparams'
        disp = 'PC'

    elif len(ctoidf[sel]) > 1:
        # get match within 0.5 days
        LOGWARNING('WRN! MORE THAN 1 cTOI')
        sel &= np.isclose(
            period,
            nparr(ctoidf[ctoidf['TIC ID']==ticid]['Period (days)']),
            atol=0.5
        )
        if len(toidf[sel]) > 1:
            raise NotImplementedError('got 2 TOIs within 0.5d of obsd period')
        tdf = toidf[sel]
        toiname = tdf['CTOI'].iloc[0]
        target = toiname
        flag = 'newparams'
        disp = "PC"

    ##########################################
    # check for matches by spatial position + period
    toicoords = SkyCoord(nparr(toidf['TIC Right Ascension']),
                         nparr(toidf['TIC Declination']), unit=(u.deg),
                         frame='icrs')
    ctoicoords = SkyCoord(nparr(ctoidf['RA']), nparr(ctoidf['Dec']),
                          unit=(u.deg), frame='icrs')
    c_obj = SkyCoord(ra, dec, unit=(u.deg), frame='icrs')

    toiseps = toicoords.separation(c_obj).to(u.arcsec).value
    ctoiseps = ctoicoords.separation(c_obj).to(u.arcsec).value

    spatial_cutoff = 126 # arcseconds ~= 6 pixels

    sel = nparr(toiseps < spatial_cutoff)
    if len(toiseps[sel]) == 1:

        sel &= np.isclose(
            period,
            nparr(toidf['Orbital Period Value']),
            atol=0.5
        )

        if len(toiseps[sel]) == 1:
            #
            # match is within 6 pixels of a TOI, and has same period and TIC
            # ID.  by default, assume TOI program got it right.
            #
            if len(toidf[toidf['TIC']==ticid])==1:
                tdf = toidf[sel]
                toiname = tdf['Full TOI ID'].iloc[0]
                target = toiname
                flag = 'newparams'
                disp = tdf['TOI Disposition'].iloc[0]

            #
            # match is within 6 pixels of a TOI, has same period, but different
            # TIC ID. 
            #
            elif len(toidf[toidf['TIC']==ticid])==0:
                sel = nparr(toiseps < spatial_cutoff)
                tdf = toidf[sel]
                toiname = tdf['Full TOI ID'].iloc[0]
                extranote = "<2' from TOI{}, same period; diff TICID.".format(repr(toiname))

        else:
            #
            # match is within 6 pixels of a TOI, but has different period. 
            # assume i got it right, but add warning to note.
            #
            if len(toidf[toidf['TIC']==ticid])==0:
                sel = nparr(toiseps < spatial_cutoff)
                tdf = toidf[sel]
                toiname = tdf['Full TOI ID'].iloc[0]
                extranote = "<2' from TOI{} but diff period & TICID.".format(repr(toiname))

            #
            # matched within 6 pixels of TOI, has same TICID, but different
            # period. most likely if a different planet was discovered.
            #
            else:
                raise NotImplementedError(
                    'need to manually inspect this case!'
                )
                sel = nparr(toiseps < spatial_cutoff)
                tdf = toidf[sel]
                toiname = tdf['Full TOI ID'].iloc[0]
                target = toiname
                flag = 'newparams'
                extranote = "<2' from TOI{}, same TICID; diff period.".format(repr(toiname))

    ctoisel = nparr(ctoiseps < spatial_cutoff)
    if len(ctoiseps[ctoisel]) == 1:

        ctoisel &= np.isclose(
            period,
            nparr(ctoidf['Period (days)']),
            atol=0.5
        )

        #
        # match is within 6 pixels of a CTOI, and has same period and TIC
        # ID.  by default, assume CTOI got it right, and add new params
        #
        if (len(ctoidf[ctoidf['TIC ID']==ticid])==1
            and
            len(ctoiseps[ctoisel])==1
        ):
            tdf = ctoidf[ctoisel]
            ctoiname = tdf['CTOI'].iloc[0]
            target = 'TIC'+str(ctoiname)
            flag = 'newparams'
            extranote = "Same period and TICID as CTOI{}.".format(repr(ctoiname))
            # disp = tdf['TOI Disposition'].iloc[0] #NOTE: don't update disposition

        elif (len(ctoidf[ctoidf['TIC ID']==ticid])==0
            and
            len(ctoiseps[ctoisel])==0
        ):
            tdf = ctoidf[ctoisel]
            ctoiname = tdf['CTOI'].iloc[0]
            extranote = "WRN! <2' from CTOI{} (diff period and TICID).".format(repr(ctoiname))

        elif (len(ctoidf[ctoidf['TIC ID']==ticid])==0
            and
            len(ctoiseps[ctoisel])==1
        ):
            tdf = ctoidf[ctoisel]
            ctoiname = tdf['CTOI'].iloc[0]
            extranote = "<2' from CTOI{}, matching period, but diff TICID. Blend?".format(repr(ctoiname))
            raise NotImplementedError('manually fix {}'.format(extranote))

        elif (len(ctoidf[ctoidf['TIC ID']==ticid])==1
            and
            len(ctoiseps[ctoisel])==0
        ):
            tdf = ctoidf[ctoisel]
            ctoiname = tdf['CTOI'].iloc[0]
            target = 'TIC'+str(ctoiname)
            flag = 'newparams'
            extranote = "Same TICID as CTOI{}; different period. New planet?".format(repr(ctoiname))
            raise NotImplementedError('manually fix {}'.format(extranote))
            # disp = tdf['TOI Disposition'].iloc[0] #NOTE: don't update disposition

    ##########################################

    if len(extranote) > 1:
        notes = (
            'Bouma+2019 CDIPS1. Vetting report gives details.{}'.
            format(' '+extranote)
        )
    else:
        notes = 'Bouma+2019 CDIPS1. Vetting report gives details.'

    assert len(notes) < 120

    # finally write
    d = {
        'target':target,
        'flag':flag,
        'disp':disp,
        'period':param_values['period'],
        'period_unc':_get_unc(param_errs,'period'),
        'epoch':param_values['t0'],
        'epoch_unc':_get_unc(param_errs,'t0'),
        'depth':depth,
        'depth_unc':depth_unc,
        'duration':T_tot,
        'duration_unc':T_tot_unc,
        'inc':param_values['incl'],
        'inc_unc':_get_unc(param_errs,'incl'),
        'imp':b,
        'imp_unc':b_unc,
        'r_planet':param_values['rp'],
        'r_planet_unc':_get_unc(param_errs,'rp'),
        'ar_star':param_values['sma'],
        'a_rstar_unc':_get_unc(param_errs,'sma'),
        'radius':radius, #radius (R_Earth)
        'radius_unc':radius_unc, #radius uncertainty
        'mass':np.nan, #mass (M_Earth)
        'mass_unc':np.nan, #mass uncertainty
        'temp':np.nan, #equilibrium temperature (K)
        'temp_unc':np.nan, #equilibrium temperature uncertainty
        'insol':np.nan, #insolation flux (flux_Earth)
        'insol_unc':np.nan, #insolation flux uncertainty
        'dens':np.nan, #fitted stellar density (g/cm^3)
        'dens_unc':np.nan, #fitted stellar density uncertainty
        'sma':np.nan, #semi-major axis (AU)
        'sma_unc':np.nan, #semi-major axis uncertainty
        'ecc':np.nan,
        'ecc_unc':np.nan,
        'arg_peri':np.nan,
        'arg_peri_unc':np.nan,
        'time_peri':np.nan,
        'time_peri_unc':np.nan,
        'vsa':np.nan,
        'vsa_unc':np.nan,
        'tag':'{}_bouma_cdips-v{}_00001'.format(today_YYYYMMDD(),
                                                str(cdipsvnum).zfill(2)),
        'group':'tfopwg',
        'prop_period':0,
        'notes':notes
    }

    try:
        df = pd.DataFrame(d, index=[0])
    except AttributeError:
        import IPython; IPython.embed()
        assert 0
    df.to_csv(outpath,sep="|",index=False)
    LOGINFO('made {}'.format(outpath))


if __name__=="__main__":

    sectors = range(14,27)

    for sector in sectors:

        cdips_cat_vnum = 0.6
        is_not_cdips_still_good = False

        main(sector=sector, cdips_cat_vnum=cdips_cat_vnum,
             is_not_cdips_still_good=is_not_cdips_still_good)
