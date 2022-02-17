"""
usage:
python -u fit_models_to_gold.py &> logs/YYYYMMDD_fit_s{sector}.log &

contents:
    main
        _fit_transit_model_single_sector
    get_teff_rstar_logg
    fit_results_to_ctoi_csv

helpers:
    _get_sector_metadata
    _define_and_make_directories
    _get_lcpaths
    _update_status
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

import numpy as np, matplotlib.pyplot as plt, pandas as pd, pymc3 as pm
import os, pickle, requests, socket
from glob import glob
import time as pytime

import matplotlib as mpl
mpl.use('Agg')

from numpy import array as nparr
from os.path import join
from collections import OrderedDict
from importlib.machinery import SourceFileLoader
from copy import deepcopy

from astropy.coordinates import SkyCoord
from astropy import units as u, constants as const
from astropy.io import fits
from astroquery.mast import Catalogs

from astrobase import imageutils as iu

from cdips.utils import today_YYYYMMDD, str2bool
from cdips.utils.pipelineutils import save_status, load_status
from cdips.utils.catalogs import (
    get_cdips_catalog, get_toi_catalog, get_exofop_ctoi_catalog
)
from cdips.utils.mamajek import (
    get_interp_mass_from_rstar, get_interp_rstar_from_teff
)
from cdips.utils import lcutils as lcu
from cdips.lcproc.find_planets import run_periodograms_and_detrend
import cdips.vetting.make_all_vetting_reports as mavr

from betty.io import given_priordict_make_priorfile
from betty.modelfitter import ModelFitter
from betty.posterior_table import make_posterior_table
import betty.plotting as bp

##########
# CONFIG #
##########

DATADIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/data/mcmc_fitting_identifiers'

mcmc_fails_path = join(DATADIR, 'KNOWN_MCMC_FAILS.csv')
mcmc_fails_df = pd.read_csv(mcmc_fails_path, sep=';')
skip_convergence_path = join(DATADIR, 'SKIP_CONVERGENCE_IDENTIFIERS.csv')
skip_convergence_df = pd.read_csv(skip_convergence_path)

KNOWN_MCMC_FAILS = list(nparr(mcmc_fails_df.source_id))
SKIP_CONVERGENCE_IDENTIFIERS = list(nparr(skip_convergence_df.source_id))

host = socket.gethostname()
if 'phtess' in host:
    CLASSIFXNDIR = "/home/lbouma/proj/cdips/results/vetting_classifications/Sector14_through_Sector26"
    resultsbase = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
    database = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/data/'
elif 'brik' in host:
    CLASSIFXNDIR = "/home/luke/Dropbox/proj/cdips/results/vetting_classifications/Sector14_through_Sector26"
    resultsbase = '/home/luke/Dropbox/proj/cdips/results/'
    database = '/home/luke/Dropbox/proj/cdips/data/'


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

    assert sector <= 26, '>=Year 3 will need adaptive exposure time in the fits'

    lcbasedir, resultsdir = _define_and_make_directories(
        sector, is_not_cdips_still_good=is_not_cdips_still_good)

    df, cdips_df, pfdf, supplementstatsdf, toidf, ctoidf = (
        _get_sector_metadata(sector, cdips_cat_vnum=cdips_cat_vnum)
    )

    lcpaths = _get_lcpaths(df, lcbasedir)

    for lcpath in lcpaths:

        _hdr = iu.get_header_keyword_list(
            lcpath, ['CAMERA','CCD'], ext=0
        )
        cam, ccd = _hdr['CAMERA'], _hdr['CCD']
        source_id = np.int64(lcpath.split('gaiatwo')[1].split('-')[0].lstrip('0'))
        lcname = os.path.basename(lcpath)

        #
        # make fitresults and samples directories
        #
        outdirs = [
            join(resultsdir,'fitresults',lcname.replace('.fits','')),
            join(resultsdir,'samples',lcname.replace('.fits',''))
        ]
        for outdir in outdirs:
            if not os.path.exists(outdir):
                LOGINFO(f'Made {outdir}')
                os.mkdir(outdir)

        #
        # collect metadata for this target star
        #
        mdf = cdips_df[cdips_df['source_id']==source_id]
        if len(mdf) != 1:
            errmsg = 'expected exactly 1 source match in CDIPS cat'
            raise AssertionError(errmsg)

        supprow = mavr._get_supprow(source_id, supplementstatsdf)
        suppfulldf = supplementstatsdf

        pfrow = pfdf.loc[pfdf['source_id']==source_id]
        if len(pfrow) != 1:
            errmsg = 'expected exactly 1 source match in period find df'
            raise AssertionError(errmsg)

        outpath = join(
            resultsdir, 'fitresults', lcname.replace('.fits',''),
            lcname.replace('.fits','_fitparameters.csv')
        )

        #
        # if you haven't already made the output parameter file (which requires
        # convergence) then start fitting.
        #
        if not os.path.exists(outpath):

            _fit_transit_model_single_sector(
                lcpath, outpath, mdf, source_id, supprow, suppfulldf, pfdf,
                pfrow, toidf, ctoidf, sector, nworkers,
                SKIP_CONVERGENCE_IDENTIFIERS, cdipsvnum=cdipsvnum,
                overwrite=overwrite
            )

        else:

            status_file = join(
                os.path.dirname(outpath), 'run_status.stat'
            )
            status = load_status(status_file)

            modelid = 'simpletransit'
            if str2bool(status[modelid]['is_converged']):
                LOGINFO(
                    f'{source_id} converged; already wrote ctoi csv.'
                )

            elif (
                not str2bool(status[modelid]['is_converged'])
                and np.int64(source_id) in SKIP_CONVERGENCE_IDENTIFIERS
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

def get_approx_transit_depth_from_ror(_df):

    ror = _df.loc['ror', 'median']
    b = _df.loc['b','median']
    u1 = _df.loc['u_star[0]','median']
    u2 = _df.loc['u_star[1]','median']

    arg = 1 - np.sqrt(1 - b**2)
    f0 = 1 - 2 * u1 / 6.0 - 2 * u2 / 12.0
    f = 1 - u1* arg - u2 * arg ** 2
    factor = f0 / f

    delta = ror**2 * (1/factor)

    return delta

def fit_results_to_ctoi_csv(msg_to_pass, ticid, ra, dec, m, summdf, outpath,
                            posterior_csvpath, toidf, ctoidf, teff, teff_err,
                            rstar, rstar_err, logg, logg_err, cdipsvnum=1):
    """
    args:

        m / summdf: PyMC3 model and summary DataFrame.

        outpath: csv to save to

        posterior_csvpath: csv file with raw posterior info

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

    extranote = f'{msg_to_pass}. ' if len(msg_to_pass)>0 else ''

    # fitted simpletransit
    # ['logg_star', 'log_jitter', 't0', 'period', 'tess_mean', 'r_star', 'rho_star',
    # 'log_ror', 'ror', 'r_pl', 'b', 'u_star[0]', 'u_star[1]']
    # derived
    #  ['tess_roughdepth', 'r_planet', 'a_Rs', 'cosi', 'sini', 'T_14', 'T_13']
    _df = pd.read_csv(posterior_csvpath, index_col=0)

    if 'tess_roughdepth' not in _df:
        approx_depth = get_approx_transit_depth_from_ror(_df)
        rp_rel_unc = _df.loc['r_planet','sd']/_df.loc['r_planet','mean']
        approx_depth_unc = 2*approx_depth*rp_rel_unc
    else:
        approx_depth = _df.loc['tess_roughdepth','mean']
        approx_depth_unc = _df.loc['tess_roughdepth','sd']

    period = _df.loc['period','mean']
    period_unc = _df.loc['period','sd']
    epoch = _df.loc['t0','mean']
    epoch_unc = _df.loc['t0','sd']

    b_lower, b_upper = _df.loc['b','hdi_3%'], _df.loc['b','hdi_97%']
    rp_lower, rp_upper = _df.loc['r_planet','hdi_3%'], _df.loc['r_planet','hdi_97%'] # rjup
    extranote += f'Rp=({rp_lower:.2f}-{rp_upper:.2f})Rj. '
    extranote += f'b={b_lower:.2f}-{b_upper:.2f} (3-97HDI). '

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
                extranote += "<2' from TOI{}, same period; diff TICID.".format(repr(toiname))

        else:
            #
            # match is within 6 pixels of a TOI, but has different period. 
            # assume i got it right, but add warning to note.
            #
            if len(toidf[toidf['TIC']==ticid])==0:
                sel = nparr(toiseps < spatial_cutoff)
                tdf = toidf[sel]
                toiname = tdf['Full TOI ID'].iloc[0]
                extranote += "<2' from TOI{} but diff period & TICID.".format(repr(toiname))

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
                extranote += "<2' from TOI{}, same TICID; diff period.".format(repr(toiname))

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
            extranote += "Same period+TICID as CTOI{}.".format(repr(ctoiname))
            # disp = tdf['TOI Disposition'].iloc[0] #NOTE: don't update disposition

        elif (len(ctoidf[ctoidf['TIC ID']==ticid])==0
            and
            len(ctoiseps[ctoisel])==0
        ):
            tdf = ctoidf[ctoisel]
            ctoiname = tdf['CTOI'].iloc[0]
            extranote += "<2' from CTOI{} (diff period+TICID).".format(repr(ctoiname))

        elif (len(ctoidf[ctoidf['TIC ID']==ticid])==0
            and
            len(ctoiseps[ctoisel])==1
        ):
            tdf = ctoidf[ctoisel]
            ctoiname = tdf['CTOI'].iloc[0]
            extranote += "<2' from CTOI{}, same period, diff TICID. Blend?".format(repr(ctoiname))
            raise NotImplementedError('manually fix {}'.format(extranote))

        elif (len(ctoidf[ctoidf['TIC ID']==ticid])==1
            and
            len(ctoiseps[ctoisel])==0
        ):
            tdf = ctoidf[ctoisel]
            ctoiname = tdf['CTOI'].iloc[0]
            target = 'TIC'+str(ctoiname)
            flag = 'newparams'
            extranote += "==TICID/CTOI{}; diff period. New PC?".format(repr(ctoiname))
            raise NotImplementedError('manually fix {}'.format(extranote))
            # disp = tdf['TOI Disposition'].iloc[0] #NOTE: don't update disposition

    ##########################################

    if len(extranote) > 1:
        notes = (
            f'CDIPS: see PDF report. {extranote}'
        )
    else:
        notes = 'CDIPS: see PDF report.'

    assert len(notes) < 120, f'Got len(notes)={len(notes)}, notes={notes}'

    # finally write
    d = {
        'target':target,
        'flag':flag,
        'disp':disp,
        'period':_df.loc['period','mean'],
        'period_unc':_df.loc['period','sd'],
        'epoch':_df.loc['t0','mean'],
        'epoch_unc':_df.loc['t0','sd'],
        'depth':1e6*approx_depth,
        'depth_unc':1e6*approx_depth_unc,
        'duration':_df.loc['T_14', 'mean'],
        'duration_unc':_df.loc['T_14', 'sd'],
        'inc':np.nan,
        'inc_unc':np.nan,
        'imp':_df.loc['b','mean'],
        'imp_unc':_df.loc['b','sd'],
        'r_planet':_df.loc['ror', 'mean'], # actually Rp/R*. poorly named.
        'r_planet_unc':_df.loc['ror', 'sd'],
        'ar_star':_df.loc['a_Rs','mean'],
        'a_rstar_unc':_df.loc['a_Rs','sd'],
        'radius':((_df.loc['r_planet','mean'])*u.Rjup).to(u.Rearth).value, #radius (R_Earth)
        'radius_unc':((_df.loc['r_planet','sd'])*u.Rjup).to(u.Rearth).value, #radius uncertainty
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
        'tag':f"{today_YYYYMMDD()}_bouma_cdips-v{str(cdipsvnum).zfill(2)}_00001",
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


def _get_sector_metadata(sector, cdips_cat_vnum=None, is_not_cdips_still_good=0):

    if is_not_cdips_still_good:
        classifxn_csv = join(
            CLASSIFXNDIR,
            "sector-{}_PCs_NOT_CDIPS_STILL_GOOD.csv".format(sector)
        )
    else:
        classifxn_csv = join(
            CLASSIFXNDIR,
            "sector-{}_PCs_CLEAR_THRESHOLD.csv".format(sector)
        )

    df = pd.read_csv(classifxn_csv, sep=';')

    cdips_df = get_cdips_catalog(ver=cdips_cat_vnum)

    pfpath = join(resultsbase,
              'cdips_lc_periodfinding/'
              'sector-{}/'.format(sector)+
              'initial_period_finding_results_supplemented.csv')
    pfdf = pd.read_csv(pfpath, sep=';')

    supppath = join(resultsbase,
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
        resultsdir = join(
            resultsbase, f'fit_gold/Year2_simpletransit_polyremove/sector-{sector}_NOT_CDIPS_STILL_GOOD'
        )
    else:
        resultsdir = join(
            resultsbase, f'fit_gold/Year2_simpletransit_polyremove/sector-{sector}_CLEAR_THRESHOLD'
        )
    dirs = [resultsdir,
            join(resultsdir,'fitresults'),
            join(resultsdir,'samples')
           ]
    for _d in dirs:
        if not os.path.exists(_d):
            LOGINFO(f'Making {_d}...')
            os.mkdir(_d)

    lcbasedir = join(lcbase, f'CDIPS_LCS/sector-{sector}/')

    return lcbasedir, resultsdir


def _get_lcpaths(df, lcbasedir):

    # vet_hlsp_cdips_tess_ffi_gaiatwo0003125263468681400320-0006-cam1-ccd1_tess_v01_llc.pdf
    lcnames = list(map(
        lambda x: x.replace('vet_','').replace('.pdf','.fits'),
        nparr(df['Name'])
    ))

    lcglobs = [
        join(lcbasedir, 'cam?_ccd?', lcname) for lcname in lcnames
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
                                     SKIP_CONVERGENCE_IDENTIFIERS,
                                     cdipsvnum=1, overwrite=1):
    try_mcmc = True
    identifier = source_id

    fit_savdir = os.path.dirname(outpath)
    chain_savdir = os.path.dirname(outpath).replace('fitresults', 'samples')

    modelid = 'simpletransit'
    starid = (
        os.path.basename(lcpath).
        replace('.fits','').
        replace('hlsp_cdips_tess_ffi_','')
    )

    exofopcsvpath = join(fit_savdir, f'{starid}_{modelid}_fitparameters.csv')
    if os.path.exists(exofopcsvpath):
        LOGINFO(f"Found {exofopcsvpath}, continue.")
        return 1

    #
    # get time & flux. (follows procedure from do_initial_period_finding).
    #
    hdul = fits.open(lcpath)
    hdr = hdul[0].header

    APNAME = 'PCA1'
    source_id, time, mag, xcc, ycc, ra, dec, _, _ = (
        lcu.get_lc_data(lcpath, mag_aperture=APNAME)
    )
    err_mag = hdul[1].data['IRE1']

    dtr_method, break_tolerance, window_length = 'best', 0.5, 0.5
    dtr_dict = {'method':dtr_method,
                'break_tolerance':break_tolerance,
                'window_length':window_length}

    r, dtr_time, dtr_flux, dtr_stages_dict = run_periodograms_and_detrend(
        source_id, time, mag, dtr_dict, return_extras=True
    )

    # save initial (orbit-edge-masked) time and flux, w/out detrending.
    outdf = pd.DataFrame({
        'time': dtr_stages_dict['time'],
        'flux_'+APNAME: dtr_stages_dict['flux'],
    })
    lc_csvpath = join(fit_savdir, f'{starid}_{modelid}_rawlc.csv')
    outdf.to_csv(lc_csvpath, index=False)
    LOGINFO(f'Wrote {lc_csvpath}')

    from betty.helpers import p2p_rms

    # define the time, flux, flux_err to be used in the fitting.
    if modelid == 'simpletransit':

        # # if you wanted to use the detrended light curve (distorted due to
        # # notch), it would be the three below.  instead, we use the PCA light
        # # curve data, and trim it to local windows around each transit (with
        # # local polynomials removed).
        # time = deepcopy(dtr_time)
        # flux = deepcopy(dtr_flux)
        # flux_err = p2p_rms(flux)*np.ones_like(flux)

        # get the PCA light curve data
        time = deepcopy(dtr_stages_dict['time'])
        flux = deepcopy(dtr_stages_dict['flux'])
        flux_err = p2p_rms(flux)*np.ones_like(flux)

        # get the trimming window priors
        period_val = r['tls_period']
        t0_val = r['tls_t0']
        tdur_val = r['tls_duration']

        # given n*tdur omitted on each side of either transit, there is P-2*ntdur
        # space between each time group.
        if tdur_val < (2/24):
            n_tdurs = 5.0
        elif tdur_val < (2.5/24):
            n_tdurs = 4.5
        elif tdur_val < (3/24):
            n_tdurs = 4
        else:
            n_tdurs = 3.5
        mingap = period_val - 2.1*n_tdurs*tdur_val
        while mingap < 0:
            n_tdurs -= 0.5
            mingap = period_val - 2.1*n_tdurs*tdur_val
        msg = f"P={period_val}, N={n_tdurs}, Tdur={tdur_val}"
        assert mingap > 0, msg

        # do the "local polynomial removal" around each transit window.
        from cdips.lcproc.detrend import transit_window_polynomial_remover
        outpath = join(fit_savdir, f'{starid}.png')
        d = transit_window_polynomial_remover(
            time, flux, flux_err, t0_val,
            period_val, tdur_val, n_tdurs=n_tdurs,
            method='poly_2', plot_outpath=outpath
        )

        time = np.hstack([d[f'time_{ix}'] for ix in range(d['ngroups'])])
        flux = np.hstack([d[f'flat_flux_{ix}'] for ix in range(d['ngroups'])])
        flux_err = np.hstack([d[f'flux_err_{ix}'] for ix in range(d['ngroups'])])

    elif modelid == 'localpolytransit':
        # localpolytransit fits the raw light curve, w/out detrending
        time = deepcopy(dtr_stages_dict['time'])
        flux = deepcopy(dtr_stages_dict['flux'])
        flux_err = p2p_rms(flux)*np.ones_like(flux)

    #
    # define the paths. get the stellar parameters, and do the fit!
    #
    try:
        teff, teff_err, rstar, rstar_err, logg, logg_err = (
            get_teff_rstar_logg(hdr)
        )
    except (NotImplementedError, ValueError) as e:
        LOGERROR(e)
        LOGERROR(f'did not get rstar for {source_id}. MUST MANUALLY FIX.')
        try_mcmc = False

    #
    # initialize status file
    #
    status_file = join(fit_savdir, 'run_status.stat')
    if not os.path.exists(status_file):
        save_status(
            status_file, modelid,
            {'is_converged':False, 'n_steps_run':0, 'mean_rhat':99}
        )
    status = load_status(status_file)[modelid]

    #
    # initialize prior; cache it to know what you did.
    #
    prior_path = join(
        fit_savdir, f'{starid}_{modelid}_priors.py'
    )
    if not os.path.exists(prior_path):
        # period/t0: guess period known 1% precision TLS
        period_val, period_err = r['tls_period'], 0.01*r['tls_period']
        t0_val, t0_err = r['tls_t0'], 30/(60*24)
        # Rp/R*: log-uniform; lower 10x lower than TLS depth
        depth = 1-r['tls_depth']
        ror_val = depth**(0.5)
        ror_lower = ror_val/10**0.5
        # stellar params
        rstar_val, rstar_err = rstar, rstar_err
        if pd.isnull(rstar_err):
            rstar_err = 0.3*rstar # 30% relative unc assumed by default
        logg_val, logg_err = logg, logg_err
        if pd.isnull(logg_err):
            logg_err = 0.2*logg # 20% relative unc assumed by default

        priordict = {
        'period': ('Normal', period_val, period_err),
        't0': ('Normal', t0_val, t0_err),
        'log_ror': ('Uniform', np.log(ror_lower), np.log(1), np.log(ror_val)),
        'b': ('ImpactParameter', 0.5),
        'u_star': ('QuadLimbDark',),
        'r_star': ('Normal', rstar_val, rstar_err),
        'logg_star': ('Normal', logg_val, logg_err),
        'tess_mean': ('Normal', 1, 0.1),
        'log_jitter': ('Normal', r"\log\langle \sigma_f \rangle", 2.0),
        }
        if modelid == 'localpolytransit':
            del priordict['tess_mean'] # window-specific a0/a1/a2 to be constructed

        _init_priordict = deepcopy(priordict)
        given_priordict_make_priorfile(priordict, prior_path)

    else:
        prior_mod = SourceFileLoader('prior', prior_path).load_module()
        priordict = prior_mod.priordict

        _init_priordict = deepcopy(priordict)
        initkeylist = deepcopy(list(_init_priordict.keys()))
        for k in initkeylist:
            if k.startswith('tess_'):
                del _init_priordict[k]

    #
    # LOGIC:
    #
    # if not converged and no steps previously run:
    #   run 2000 steps. write status file.
    # 
    # reload status file.
    # if not converged, but based on Rhat seems like it will (Rhat<1.1):
    #   run 10000 samples, write status file.
    #
    # reload status file.
    # if still not converged, but Rhat<1.1, add it to the list of things we're
    # OK with not converging.  print a warning.
    #
    # reload status file.
    # if not converged:
    #   print a warning.
    #

    flux_err = np.nanmedian(flux_err)*np.ones_like(time)
    tess_texp = np.nanmedian(np.diff(time))

    datasets = OrderedDict()
    if modelid == 'simpletransit':
        datasets['tess'] = [
            time.astype(np.float64),
            flux.astype(np.float64),
            flux_err.astype(np.float64),
            tess_texp
        ]

    elif modelid == 'localpolytransit':
        # In this model, we fit local windows.  Define those windows, and the
        # priors on the fitted parameters.

        from betty.helpers import _subset_cut, _quicklcplot
        from astrobase.lcmath import find_lc_timegroups

        tdur_val = r['tls_duration'] # units: days
        period_val, period_err = r['tls_period'], 0.01*r['tls_period']
        t0_val, t0_err = r['tls_t0'], 30/(60*24)

        if tdur_val < (2/24):
            n_tdurs = 5.0
        elif tdur_val < (2.5/24):
            n_tdurs = 4.5
        elif tdur_val < (3/24):
            n_tdurs = 4
        else:
            n_tdurs = 3.5

        # given n*tdur omitted on each side of either transit, there is P-2*ntdur
        # space between each time group.
        mingap = period_val - 2.1*n_tdurs*tdur_val
        while mingap < 0:
            n_tdurs -= 0.5
            mingap = period_val - 2.1*n_tdurs*tdur_val
        msg = f"P={period_val}, N={n_tdurs}, Tdur={tdur_val}"
        assert mingap > 0, msg

        # trim
        outpath = join(fit_savdir, f'{starid}_{modelid}_rawlc.png')
        _quicklcplot(time, flux, flux_err, outpath)
        time, flux, flux_err = _subset_cut(
            time, flux, flux_err, n=n_tdurs, t0=t0_val,
            per=period_val, tdur=tdur_val
        )
        outpath = join(fit_savdir, f'{starid}_{modelid}_rawtrimlc.png')
        _quicklcplot(time, flux, flux_err, outpath)

        ngroups, groupinds = find_lc_timegroups(time, mingap=mingap)

        for ix, g in enumerate(groupinds):
            tess_texp = np.nanmedian(np.diff(time[g]))
            datasets[f'tess_{ix}'] = [
                time[g].astype(np.float64),
                flux[g].astype(np.float64),
                flux_err[g].astype(np.float64),
                tess_texp
            ]

        if 'tess_0_mean' not in priordict.keys():
            for n, (name, (x, y, yerr, texp)) in enumerate(datasets.items()):
                # mean + a1*(time-midtime) + a2*(time-midtime)^2.
                priordict[f'{name}_mean'] = ('Normal', 1, 0.1)
                priordict[f'{name}_a1'] = ('Uniform', -0.1, 0.1)
                priordict[f'{name}_a2'] = ('Uniform', -0.1, 0.1)

        given_priordict_make_priorfile(priordict, prior_path)

    else:
        raise NotImplementedError

    pklpath = join(chain_savdir, f'{starid}_{modelid}.pkl')
    msg_to_pass = ''

    if identifier in KNOWN_MCMC_FAILS:
        LOGWARNING(f'WRN! identifier {identifier} requires manual fixing.')
        try_mcmc = False

    if (
        not str2bool(status['is_converged'])
        and int(status['n_steps_run']) == 0
        and try_mcmc
    ):

        n_mcmc_steps = 2000
        m = ModelFitter(
            modelid, datasets, priordict, plotdir=fit_savdir,
            pklpath=pklpath, overwrite=0, N_samples=n_mcmc_steps,
            N_cores=os.cpu_count()
        )
        _update_status(m, priordict, status_file, modelid, n_mcmc_steps)

    status = load_status(status_file)[modelid]
    if (
        not str2bool(status['is_converged'])
        and int(status['n_steps_run']) != 10000
        and float(status['mean_rhat']) < 1.2
        and try_mcmc
    ):
        n_mcmc_steps = 10000
        msg = (
            f"{identifier}: got <Rhat>={status['mean_rhat']} after 2k steps."
            f"Continuing with {n_mcmc_steps} steps."
        )
        LOGINFO(msg)
        m = ModelFitter(
            modelid, datasets, priordict, plotdir=fit_savdir,
            pklpath=pklpath, overwrite=1, N_samples=n_mcmc_steps,
            N_cores=os.cpu_count()
        )
        _update_status(m, priordict, status_file, modelid, n_mcmc_steps)

    # If we failed to get convergence, then skip the convergence requirement to
    # proceed.

    status = load_status(status_file)[modelid]
    if (
        not str2bool(status['is_converged'])
        and int(status['n_steps_run']) == 10000
    ):
        if np.int64(identifier) not in SKIP_CONVERGENCE_IDENTIFIERS:
            with open(skip_convergence_path, mode='a') as f:
                f.write('\n'+str(source_id))
            LOGINFO(f'Appended {source_id} to {skip_convergence_path}')
            SKIP_CONVERGENCE_IDENTIFIERS.append(
                np.int64(source_id)
            )

        if float(status['mean_rhat']) > 1.01:
            msg_to_pass = f"Not converged <R>={float(status['mean_rhat']):.3f}"
        else:
            msg_to_pass = ''

    status = load_status(status_file)[modelid]
    if (
        str2bool(status['is_converged'])
        or np.int64(identifier) in SKIP_CONVERGENCE_IDENTIFIERS
    ):
        # load the pickle file of sampling results
        m = ModelFitter(
            modelid, datasets, priordict, plotdir=fit_savdir,
            pklpath=pklpath, overwrite=0, N_samples=42,
            N_cores=os.cpu_count()
        )
    else:
        LOGERROR(
            'ERR! {} did not converge, after {} steps. MUST MANUALLY FIX.'.
            format(identifier, status['n_steps_run'])
        )
        return 0

    #
    # visualize results + summary stats
    #
    summdf = pm.summary(m.trace, var_names=list(priordict), round_to=10,
                        kind='stats', stat_funcs={'median':np.nanmedian},
                        extend=True)

    fitindiv = 1 if modelid == 'simpletransit' else 0
    fitindivpanels = 1 if modelid == 'localpolytransit' else 0
    phaseplot = 1
    cornerplot = 1
    posttable = 1

    posterior_texpath = join(fit_savdir, f'{starid}_{modelid}_posteriortable.tex')
    if posttable and not os.path.exists(posterior_texpath):
        make_posterior_table(
            pklpath, priordict, posterior_texpath, modelid, makepdf=1
        )

    outpath = join(fit_savdir, f'{starid}_{modelid}_phaseplot.png')
    if phaseplot and not os.path.exists(outpath):
        bp.plot_phasefold(
            m, summdf, outpath, modelid=modelid, inppt=1, binsize_minutes=30
        )

    outpath = join(fit_savdir, f'{starid}_{modelid}_fitindivpanels.png')
    if fitindivpanels and not os.path.exists(outpath):
        bp.plot_fitindivpanels(
            m, summdf, outpath, modelid=modelid, singleinstrument='tess'
        )

    outpath = join(fit_savdir, f'{starid}_{modelid}_fitindiv.png')
    if fitindiv and not os.path.exists(outpath):
        bp.plot_fitindiv(m, summdf, outpath, modelid=modelid)

    outpath = join(fit_savdir, f'{starid}_{modelid}_cornerplot.png')
    if cornerplot and not os.path.exists(outpath):
        if modelid == 'simpletransit':
            bp.plot_cornerplot(list(priordict), m, outpath)
        elif modelid == 'localpolytransit':
            bp.plot_cornerplot(list(_init_priordict), m, outpath)

    #
    # if converged or in the list of IDs for which its fine to skip convegence
    # (because by-eye, the fits are converged), convert fit results to ctoi csv
    # format
    #
    status = load_status(status_file)[modelid]

    ticid = int(hdr['TICID'])
    ra, dec = hdr['RA_OBJ'], hdr['DEC_OBJ']
    LOGINFO(f'{identifier} converged. writing ctoi csv.')
    posterior_csvpath = posterior_texpath.replace(".tex", "_raw.csv")
    fit_results_to_ctoi_csv(msg_to_pass, ticid, ra, dec, m, summdf, exofopcsvpath,
                            posterior_csvpath, toidf, ctoidf, teff, teff_err,
                            rstar, rstar_err, logg, logg_err,
                            cdipsvnum=cdipsvnum)


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
                rstar_err = 0.25*rstar
                mstar = get_interp_mass_from_rstar(rstar)

            if not isinstance(rstar, float):
                raise NotImplementedError('got unexpected value for rstar! '
                                          'manual debug required')

            _Mstar = mstar*u.Msun
            _Rstar = rstar*u.Rsun

            if type(logg)==np.ma.core.MaskedConstant:
                logg = np.log10((const.G * _Mstar / _Rstar**2).cgs.value)
                logg_err = 0.25*logg

        if np.isfinite(rstar) and not np.isfinite(rstar_err):
            rstar_err = 0.25*rstar

        if not np.isfinite(logg) or not np.isfinite(logg_err):
            mstar = get_interp_mass_from_rstar(rstar)
            _Mstar = mstar*u.Msun
            _Rstar = rstar*u.Rsun

        if not np.isfinite(logg):
            logg = np.log10((const.G * _Mstar / _Rstar**2).cgs.value)

        if not np.isfinite(logg_err):
            logg_err = 0.25*logg

    else:
        raise ValueError('bad xmatch for {}'.format(hdr['GAIA-ID']))

    return teff, teff_err, rstar, rstar_err, logg, logg_err


def _update_status(m, priordict, status_file, modelid, n_mcmc_steps):

    EPS=0.01
    is_converged = np.all(
        (pm.summary(m.trace, var_names=list(priordict))['r_hat']) < 1+EPS
    )
    mean_rhat = np.mean(
        pm.summary(m.trace, var_names=list(priordict))['r_hat']
    )
    status = {'is_converged':is_converged, 'n_steps_run':n_mcmc_steps,
              'mean_rhat':mean_rhat}
    save_status(status_file, modelid, status)


if __name__=="__main__":

    sectors = range(14,27)

    for sector in sectors:

        cdips_cat_vnum = 0.6
        is_not_cdips_still_good = False

        main(sector=sector, cdips_cat_vnum=cdips_cat_vnum,
             is_not_cdips_still_good=is_not_cdips_still_good)
