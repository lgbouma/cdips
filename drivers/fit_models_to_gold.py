#
# usage:
# python -u fit_models_to_gold.py &> logs/20190815_fit_s6.log &
#
# docstring:
# see main below.
#
import os, pickle, h5py, json, shutil, requests, configparser, socket
from glob import glob

import matplotlib as mpl
mpl.use('Agg')

import numpy as np, matplotlib.pyplot as plt, pandas as pd

from numpy import array as nparr
from scipy import optimize
from scipy.interpolate import interp1d

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u, constants as const
from astroquery.mast import Catalogs

from astrobase.lcmath import sigclip_magseries
from astrobase.lcfit.transits import fivetransitparam_fit_magseries

from cdips.lcproc import detrend as dtr
from cdips.lcproc import mask_orbit_edges as moe
from cdips.plotting import vetting_pdf as vp
from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils import today_YYYYMMDD
from cdips.utils.pipelineutils import save_status, load_status

import make_vetting_multipg_pdf as mvp
from astrobase import imageutils as iu

# TODO: 3217331693306617344 fails to get rstar

LONG_RUN_IDENTIFIERS = [
    3217331693306617344, # s6 begin
    3126526081688850304,
    3107333698210242176,
    3340674976430098688,
    4827527233363019776,
    5584409013334503936,
    3050033749239975552, # s7 begin
    2919143383943171200,
    5541111035713815552,
    5599752663752776192,
    5516140233292943872,
    3064530810048196352,
    3114869682184835584
]

KNOWN_CUSTOM = [
    5290761583912487424, # s7, TLS gets 2x period, and extra detrending needed.
    5290781443841554432, # s7, systematic trends in TFASR LC, 2 transit
    5290968085934209152, # s7, systematic trends in TFASR LC, 2 transit
]

host = socket.gethostname()
if 'phtess' in host:
    CLASSIFXNDIR = "/home/lbouma/proj/cdips/results/vetting_classifications"
    resultsbase = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
    database = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/data/'
elif 'brik' in host:
    CLASSIFXNDIR = "/home/luke/Dropbox/proj/cdips/results/vetting_classifications"
    resultsbase = '/home/luke/Dropbox/proj/cdips/results/'
    database = '/home/luke/Dropbox/proj/cdips/data/'

def main(overwrite=0, sector=7, nworkers=40, cdipsvnum=1, cdips_cat_vnum=0.3):
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

        Then, pdf files under
        /results/vetting_classifications/sector-?_UNANIMOUS_GOLD are parsed to
        collect the light curves and any necessary metadata.

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
    """

    lcbasedir, tfasrdir, resultsdir = _define_and_make_directories(sector)

    df, cdips_df, pfdf, supplementstatsdf, toidf, ctoidf = _get_data(
        sector, cdips_cat_vnum=cdips_cat_vnum)

    tfa_sr_paths = _get_lcpaths(df, tfasrdir)

    import IPython; IPython.embed() #FIXME
    assert 0

    for tfa_sr_path in tfa_sr_paths:

        #
        # given the TFASR LC path, get the complete LC path
        #
        sourceid = int(tfa_sr_path.split('gaiatwo')[1].split('-')[0].lstrip('0'))
        mdf = cdips_df[cdips_df['source_id']==sourceid]
        if len(mdf) != 1:
            errmsg = 'expected exactly 1 source match in CDIPS cat'
            raise AssertionError(errmsg)

        _hdr = iu.get_header_keyword_list(tfa_sr_path, ['CAMERA','CCD'], ext=0)
        cam, ccd = _hdr['CAMERA'], _hdr['CCD']

        lcname = (
            'hlsp_cdips_tess_ffi_'
            'gaiatwo{zsourceid}-{zsector}_'
            'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            zsourceid=str(sourceid).zfill(22),
            zsector=str(sector).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )
        lcpath = os.path.join(
            lcbasedir, 'cam{}_ccd{}'.format(cam, ccd), lcname
        )

        #
        # make fitresults and samples directories
        #
        outdirs = [
            os.path.join(resultsdir,'fitresults',lcname.replace('.fits','')),
            os.path.join(resultsdir,'samples',lcname.replace('.fits',''))
        ]
        for outdir in outdirs:
            if not os.path.exists(outdir):
                os.mkdir(outdir)

        #
        # collect metadata for this target star
        #
        supprow = mvp._get_supprow(sourceid, supplementstatsdf)
        suppfulldf = supplementstatsdf

        pfrow = pfdf.loc[pfdf['source_id']==sourceid]
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

            _fit_transit_model_single_sector(tfa_sr_path, lcpath, outpath, mdf,
                                             sourceid, supprow, suppfulldf,
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
                print('{} converged and already wrote wrote ctoi csv.'.
                      format(sourceid))

            else:
                raise ValueError(
                    'got parameter file existing, but not converged.'
                    'should never happen. for DR2 {}'.format(sourceid)
                )


def _get_data(sector, cdips_cat_vnum=0.3):

    classifxn_csv = os.path.join(CLASSIFXNDIR,
                                 "sector-{}_UNANIMOUS_GOLD.csv".format(sector))
    df = pd.read_csv(classifxn_csv, sep=';')

    cdips_df = ccl.get_cdips_catalog(ver=cdips_cat_vnum)

    pfpath = os.path.join(resultsbase,
              'cdips_lc_periodfinding/'
              'sector-{}/'.format(sector)+
              'initial_period_finding_results_supplemented.csv')
    pfdf = pd.read_csv(pfpath)

    supppath = os.path.join(resultsbase,
                'cdips_lc_stats/sector-{}/'.format(sector)+
                'supplemented_cdips_lc_statistics.txt')
    supplementstatsdf = pd.read_csv(supppath, sep=';')

    toipath = os.path.join(database, 'toi-plus-2019-08-29.csv')
    toidf = pd.read_csv(toipath, sep=',')

    ctoipath = os.path.join(database, 'ctoi-exofop-2019-08-29.txt')
    ctoidf = pd.read_csv(ctoipath, sep='|')

    return df, cdips_df, pfdf, supplementstatsdf, toidf, ctoidf


def _define_and_make_directories(sector):

    host = socket.gethostname()

    if 'phtess' in host:
        lcbase = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
    elif 'brik' in host:
        lcbase = '/home/luke/Dropbox/proj/cdips/data/'

    resultsdir = os.path.join(
        resultsbase, 'fit_gold/sector-{}'.format(sector)
    )
    dirs = [resultsdir,
            os.path.join(resultsdir,'fitresults'),
            os.path.join(resultsdir,'samples')
           ]
    for _d in dirs:
        if not os.path.exists(_d):
            os.mkdir(_d)

    tfasrdir = os.path.join(lcbase,
                            'CDIPS_LCS/sector-{}_TFA_SR'.format(sector))
    lcbasedir = os.path.join(lcbase,
                             'CDIPS_LCS/sector-{}/'.format(sector))

    return lcbasedir, tfasrdir, resultsdir


def _get_lcpaths(df, tfasrdir):
    # recall all LCs (both TFA SR'd and not) are in the tfasrdir

    # vet_hlsp_cdips_tess_ffi_gaiatwo0003125263468681400320-0006_tess_v01_llc.pdf
    lcnames = list(map(
        lambda x: x.replace('vet_','').replace('.pdf','.fits'),
        nparr(df['Name'])
    ))

    lcglobs = [
        os.path.join(tfasrdir, lcname) for lcname in lcnames
    ]

    lcpaths = []
    for lcg in lcglobs:
        if len(glob(lcg)) == 1:
            lcpaths.append(glob(lcg)[0])
        else:
            raise ValueError('got more than 1 lc matching glob {}'.format(lcg))

    return lcpaths


def _fit_transit_model_single_sector(tfa_sr_path, lcpath, outpath, mdf,
                                     sourceid, supprow, suppfulldf, pfdf,
                                     pfrow, toidf, ctoidf, sector, nworkers,
                                     cdipsvnum=1, overwrite=1):
    try_mcmc = True
    identifier = sourceid
    #
    # read and re-detrend lc if needed. (recall: these planet candidates were
    # found using a penalized spline detrending in some cases).
    #
    hdul_sr = fits.open(tfa_sr_path)
    hdul = fits.open(lcpath)

    lc_sr = hdul_sr[1].data
    lc, hdr = hdul[1].data, hdul[0].header

    is_pspline_dtr = bool(pfrow['pspline_detrended'].iloc[0])

    fluxap = 'TFA2' if is_pspline_dtr else 'TFASR2'

    time, mag = lc_sr['TMID_BJD'], lc_sr[fluxap]
    try:
        time, mag = moe.mask_orbit_start_and_end(time, mag)
    except AssertionError:
        raise AssertionError(
            'moe.mask_orbit_start_and_end failed for {}'.format(tfa_sr_path)
        )

    flux = vp._given_mag_get_flux(mag)
    err = np.ones_like(flux)*1e-4

    time, flux, err = sigclip_magseries(
        time, flux, err,magsarefluxes=True, sigclip=[50,5]
    )

    if is_pspline_dtr:
        flux, _ = dtr.detrend_flux(time, flux)

    #
    # define the paths. get the stellar parameters, and do the fit!
    #
    fit_savdir = os.path.dirname(outpath)
    chain_savdir = os.path.dirname(outpath).replace('fitresults','samples')

    try:
        teff, teff_err, rstar, rstar_err, logg, logg_err = (
            get_teff_rstar_logg(hdr)
        )
    except NotImplementedError as e:
        print(e)
        print('did not get rstar for {}. MUST MANUALLY FIX.')
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
    if identifier in KNOWN_CUSTOM:
        print('WRN! identifier {} requires manual fixing.'.
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
    # if converged, convert fit results to ctoi csv format
    #
    status = load_status(status_file)[fittype]

    if str2bool(status['is_converged']):
        ticid = int(hdr['TICID'])
        ra, dec = hdr['RA_OBJ'], hdr['DEC_OBJ']
        print('{} converged. writing ctoi csv.'.format(identifier))
        fit_results_to_ctoi_csv(ticid, ra, dec, mafr, tlsr, outpath, toidf,
                                ctoidf, teff, teff_err, rstar, rstar_err, logg,
                                logg_err, cdipsvnum=cdipsvnum)
    else:
        print('WRN! {} did not converge, after {} steps. MUST MANUALLY FIX.'.
              format(identifier, status['n_steps_run']))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_teff_rstar_logg(hdr):
    #
    # Given CDIPS header, acquire estimates of Teff, Rstar, logg from TIC. If
    # Teff fails, go with the Gaia DR2 Teff.  If Rstar fails, go with Gaia DR2
    # Rstar.  If logg fails, but you have Gaia DR2 Rstar, then go from Rstar to
    # Mstar using Mamajek relation, and combine to estimate a ratty logg.
    #
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
        print('ERR! TIC query failed. trying again...')
        time.sleep(30)
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
    Tmag_cutoff = 1

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
            print('WRN! TIC teff nan. why? trying gaia value')
            teff = hdr['teff_val']
            teff_err = 100

        if type(rstar)==np.ma.core.MaskedConstant:
            print('WRN! TIC rstar nan. why? trying gaia value')
            rstar = hdr['radius_val']
            # get mass given rstar, so that you can get logg
            mamadf = pd.read_csv(
                '../data/Mamajek_Rstar_Mstar_Teff_SpT.txt',
                delim_whitespace=True
            )

            if rstar != 'NaN':

                rstar_err = 0.3*rstar

                mamarstar, mamamstar = (
                    nparr(mamadf['Rsun'])[::-1], nparr(mamadf['Msun'])[::-1]
                )
                isbad = np.insert(np.diff(mamamstar) <= 0, False, 0)
                fn_mass_to_rstar = interp1d(mamamstar[~isbad],
                                            mamarstar[~isbad],
                                            kind='quadratic',
                                            bounds_error=False,
                                            fill_value='extrapolate')
                radiusval = rstar
                fn = lambda mass: fn_mass_to_rstar(mass) - radiusval
                mass_guess = (
                    mamamstar[np.argmin(np.abs(mamarstar - radiusval))]
                )
                try:
                    mass_val = optimize.newton(fn, mass_guess)
                except RuntimeError:
                    mass_val = mass_guess
                mstar = mass_val

            else:

                raise NotImplementedError(
                    'need rstar somehow. didnt get for {}'.format(identifier)
                )

            _Mstar = mstar*u.Msun
            _Rstar = rstar*u.Rsun

            if type(logg)==np.ma.core.MaskedConstant:
                logg = np.log10((const.G * _Mstar / _Rstar**2).cgs.value)
                logg_err = 0.3*logg

    else:
        raise ValueError('bad xmatch for {}'.format(tfa_sr_path))

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
    sel = nparr(toidf['tic_id']==ticid)
    if len(toidf[sel]) == 1:
        tdf = toidf[sel]
        toiname = tdf['toi_id'].iloc[0]
        target = toiname
        flag = 'newparams'
        disp = tdf['Disposition'].iloc[0]

    elif len(toidf[sel]) > 1:
        # get match within 0.5 days
        print('WRN! MORE THAN 1 TOI')
        sel &= np.isclose(
            period,
            nparr(toidf[toidf['tic_id']==ticid]['Period']),
            atol=0.5
        )
        if len(toidf[sel]) > 1:
            raise NotImplementedError('got 2 TOIs within 0.5d of obsd period')
        tdf = toidf[sel]
        toiname = tdf['toi_id'].iloc[0]
        target = toiname
        flag = 'newparams'
        disp = tdf['Disposition'].iloc[0]

    #
    # check EXOFOP-TESS CTOI list for matches by TIC ID
    #
    sel = nparr(ctoidf['TIC ID']==ticid)
    if len(ctoidf[sel]) == 1:
        tdf = ctoidf[sel]
        ctoiname = tdf['CTOI'].iloc[0]
        target = ctoiname
        flag = 'newparams'
        disp = 'PC'

    elif len(ctoidf[sel]) > 1:
        # get match within 0.5 days
        print('WRN! MORE THAN 1 cTOI')
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
    toicoords = SkyCoord(nparr(toidf['RA']), nparr(toidf['Dec']), unit=(u.deg),
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
            nparr(toidf[toidf['tic_id']==ticid]['Period']),
            atol=0.5
        )

        if len(toiseps[sel]) == 1:
            #
            # match is within 6 pixels of a TOI, and has same period. 
            # by default, assume TOI program got it right.
            #
            tdf = toidf[sel]
            toiname = tdf['toi_id'].iloc[0]
            target = toiname
            flag = 'newparams'
            disp = tdf['Disposition'].iloc[0]

        else:
            #
            # match is within 6 pixels of a TOI, but has different period. 
            # assume i got it right, but add warning to note.
            #
            tdf = toidf[sel]
            toiname = tdf['toi_id'].iloc[0]
            extranote = "<2' from TOI{} but diff period.".format(repr(toiname))

    ctoisel = nparr(ctoiseps < spatial_cutoff)
    if len(ctoiseps[ctoisel]) == 1:

        sel &= np.isclose(
            period,
            nparr(ctoidf[ctoidf['TIC ID']==ticid]['Period (days)']),
            atol=0.5
        )

        if len(ctoiseps[sel]) == 1:
            #
            # match is within 6 pixels of a cTOI, and has same period. 
            # by default, assume I got it right. add warning to note.
            #
            tdf = ctoidf[sel]
            ctoiname = tdf['CTOI'].iloc[0]
            extranote = "WRN! <2' from CTOI{}.".format(repr(ctoiname))

        else:
            #
            # match is within 6 pixels of a TOI, but has different period. 
            # assume i got it right, but add warning to note.
            #
            tdf = ctoidf[sel]
            ctoiname = tdf['CTOI'].iloc[0]
            extranote = "<2' from CTOI{}, but diff period.".format(repr(ctoiname))

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
    print('made {}'.format(outpath))


if __name__=="__main__":
    main()
