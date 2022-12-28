"""
detrend.py - Luke Bouma (bouma.luke@gmail) - Jul 2019, Nov 2020

Contents:

Very useful:

    clean_rotationsignal_tess_singlesector_light_curve: masks orbit edges,
        sigma slide clip, detrends, re-sigma slide clip.

    _run_notch: a wrapper to Aaron Rizzuto's Notch implementation.

    _run_locor: a wrapper to Aaron Rizzuto's LOCOR implementation.

    detrend_flux: a wrapper to wotan pspline or biweight detrending (given
        vectors of time and flux).

    transit_window_polynomial_remover: given a light curve and good guesses for
        {t0,P,t_14}, trim it to windows around each transit, mask out the
        points in-transit, and fit (chi-squared) local polynomials to each.

PCA / "shared systematic trend" detrending:

    detrend_systematics: removes systematic trends, by wrapping prepare_pca,
        get_dtrvecs, and calculate_linear_model_mag.

    prepare_pca: given TFA template stars, calculates PCA eigenvectors, and the
        "optimal" number to use according to a particular heuristic.

    get_dtrvecs: given PCA eigenvectors and a lightcurve, construct the vectors
        that will actually be used in decorrelation.

    calculate_linear_model_mag:  given a set of basis vectors in a linear model
        for a target light curve (y), calculate the coefficients and apply the
        linear model prediction.

Helper functions for the above:

    eigvec_smooth_fn: a wrapper to wotan biweight detrending, with hard tuning
        for eigenvector smoothing in PCA detrending.

    insert_nans_given_rstfc: NaN insertion for PCA prep.

    compute_scores: factor analysis and cross-validation PCA score helper.
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
import matplotlib
matplotlib.use("AGG")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits

from datetime import datetime
import os, shutil
from glob import glob

from numpy import array as nparr, all as npall, isfinite as npisfinite
import numpy.lib.recfunctions as rfn
from collections import OrderedDict
from copy import deepcopy

from astrobase import imageutils as iu
from astrobase.lcmath import find_lc_timegroups

import cdips.lcproc.mask_orbit_edges as moe

from astropy.timeseries import LombScargle

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# NOTE: some code duplication with cdips.testing.check_dependencies
from wotan import flatten, version, slide_clip
from wotan.pspline import pspline
wotanversion = version.WOTAN_VERSIONING
wotanversiontuple = tuple(wotanversion.split('.'))
assert int(wotanversiontuple[0]) >= 1
assert int(wotanversiontuple[1]) >= 9

def clean_rotationsignal_tess_singlesector_light_curve(
    time, mag, magisflux=False, dtr_dict=None, lsp_dict=None,
    maskorbitedge=True, lsp_options={'period_min':0.1, 'period_max':20},
    verbose=True, slide_clip_lo=20, slide_clip_hi=3):
    """
    The goal of this function is to remove a stellar rotation signal from a
    single TESS light curve (ideally one without severe insturmental
    systematics) while preserving transits.

    "Cleaning" by default is taken to mean the sequence of mask_orbit_edge ->
    slide_clip -> detrend -> slide_clip.  "Detrend" can mean any of the Wotan
    flatteners, Notch, or LOCOR. "slide_clip" means apply windowed
    sigma-clipping removal.

    Args:
        time, mag (np.ndarray): time and magnitude (or flux) vectors

        magisflux (bool): True if the "mag" vector is a flux already

        dtr_dict (optional dict): dictionary containing arguments passed to
        Wotan, Notch, or LOCOR. Relevant keys should include:

            'dtr_method' (str): one of: ['best', 'notch', 'locor', 'pspline',
            'biweight', 'none']

            'break_tolerance' (float): number of days past which a segment of
            light curve is considered a "new segment".

            'window_length' (float): length of sliding window in days

        lsp_dict (optional dict): dictionary containing Lomb Scargle
        periodogram information, which is used in the "best" method for
        choosing between LOCOR or Notch detrending.  If this is not passed,
        it'll be constructed here after the mask_orbit_edge -> slide_clip
        steps.

        lsp_options: contains keys period_min and period_max, used for the
        internal Lomb Scargle periodogram search.

        maskorbitedge (bool): whether to apply the initial "mask_orbit_edge"
        step. Probably would only want to be false if you had already done it
        elsewhere.

    Returns:
        search_time, search_flux, dtr_stages_dict (np.ndarrays and dict): light
            curve ready for TLS or BLS style periodograms; and a dictionary of
            the different processing stages (see comments for details of
            `dtr_stages_dict` contents).
    """

    dtr_method = _get_detrending_method(dtr_dict)

    #
    # convert mag to flux and median-normalize
    #
    if magisflux:
        flux = mag
    else:
        f_x0 = 1e4
        m_x0 = 10
        flux = f_x0 * 10**( -0.4 * (mag - m_x0) )

    flux /= np.nanmedian(flux)

    #
    # ignore the times near the edges of orbits for TLS.
    #
    if maskorbitedge:
        _time, _flux = moe.mask_orbit_start_and_end(
            time, flux, raise_expectation_error=False, verbose=verbose,
            orbitgap=1, orbitpadding=6/(24)
        )
    else:
        _time, _flux = time, flux

    #
    # sliding sigma clip asymmetric [20,3]*MAD, about median. use a 3-day
    # window, to give ~100 to 150 data points at 30-minute cadence. mostly
    # avoids big flares, provided restrictive slide_clip_lo and slide_clip_hi
    # are given.
    #
    clip_window = 3
    clipped_flux = slide_clip(
        _time, _flux, window_length=clip_window, low=slide_clip_lo,
        high=slide_clip_hi, method='mad', center='median'
    )
    sel0 = ~np.isnan(clipped_flux) & (clipped_flux != 0)

    #
    # for "best" or LOCOR detrending, you need to know the stellar rotation
    # period.  so, if it hasn't already been run, run the LS periodogram here.
    # in `lsp_dict`, cache the LS peak period, amplitude, and FAP.
    #
    if (not isinstance(lsp_dict, dict)) and (dtr_method in ['locor', 'best']):

        period_min = lsp_options['period_min']
        period_max = lsp_options['period_max']

        ls = LombScargle(_time[sel0], clipped_flux[sel0], clipped_flux[sel0]*1e-3)
        freq, power = ls.autopower(
            minimum_frequency=1/period_max, maximum_frequency=1/period_min
        )
        ls_fap = ls.false_alarm_probability(power.max())
        best_freq = freq[np.argmax(power)]
        ls_period = 1/best_freq
        theta = ls.model_parameters(best_freq)
        ls_amplitude = theta[1]

        lsp_dict = {}
        lsp_dict['ls_period'] = ls_period
        lsp_dict['ls_amplitude'] = np.abs(ls_amplitude)
        lsp_dict['ls_fap'] = ls_fap

    if not isinstance(dtr_dict, dict):
        dtr_dict = {}
        dtr_dict['method'] = dtr_method

    #
    # apply the detrending call based on the method given
    #

    dtr_method_used = dtr_method

    if dtr_method in ['pspline','biweight','none']:

        if 'break_tolerance' not in dtr_dict:
            dtr_dict['break_tolerance'] = None
        if 'window_length' not in dtr_dict:
            dtr_dict['window_length'] = None

        flat_flux, trend_flux = detrend_flux(
            _time[sel0], clipped_flux[sel0], break_tolerance=dtr_dict['break_tolerance'],
            method=dtr_dict['method'], cval=None,
            window_length=dtr_dict['window_length'], edge_cutoff=None
        )

    elif dtr_method == 'notch':

        flat_flux, trend_flux, notch = _run_notch(
            _time[sel0], clipped_flux[sel0], dtr_dict, verbose=verbose
        )

    elif dtr_method == 'locor':

        flat_flux, trend_flux, notch = _run_locor(
            _time[sel0], clipped_flux[sel0], dtr_dict, lsp_dict
        )

    elif dtr_method == 'best':

        # for stars with Prot < 1 day, use LOCOR.
        # for stars with Prot > 1 day, use Notch.
        PERIOD_CUTOFF = 1.0

        if lsp_dict['ls_period'] > PERIOD_CUTOFF:
            flat_flux, trend_flux, notch = _run_notch(
                _time[sel0], clipped_flux[sel0], dtr_dict, verbose=verbose
            )
            dtr_method_used += '-notch'
        elif (
            lsp_dict['ls_period'] < PERIOD_CUTOFF and
            lsp_dict['ls_period'] > 0
        ):
            flat_flux, trend_flux, notch = _run_locor(
                _time[sel0], clipped_flux[sel0], dtr_dict, lsp_dict
            )
            dtr_method_used += '-locor'
        else:
            raise NotImplementedError(f"Got LS period {lsp_dict['ls_period']}")

    #
    # re-apply sliding sigma clip asymmetric [20,3]*MAD, about median, after
    # detrending.
    #
    clip_window = 3
    clipped_flat_flux = slide_clip(
        _time[sel0], flat_flux, window_length=clip_window, low=slide_clip_lo,
        high=slide_clip_hi, method='mad', center='median'
    )
    sel1 = ~np.isnan(clipped_flat_flux) & (clipped_flat_flux != 0)

    search_flux = clipped_flat_flux[sel1]
    search_time = _time[sel0][sel1]

    dtr_stages_dict = {
        # initial (orbit-edge-masked) time and flux.
        'time': _time,
        'flux': _flux,
        # after initial window sigma_clip on flux, what is left?
        'clipped_flux': clipped_flux,
        # non-nan indices from clipped_flux
        'sel0': sel0,
        # after detrending, what is left?
        'flat_flux': flat_flux,
        # after window sigma_clip on flat_flux, what is left?
        'clipped_flat_flux': clipped_flat_flux,
        # non-nan indices from clipped_flat_flux
        'sel1': sel1,
        # what does the detrending algorithm give as the "trend"?
        'trend_flux': trend_flux,
        'trend_time': _time[sel0],
        # what method was used? if "best", gives "best-notch" or "best-locor"
        'dtr_method_used': dtr_method_used,
        # times and fluxes used
        'search_time': search_time,
        'search_flux': search_flux
    }
    if isinstance(lsp_dict, dict):
        # in most cases, cache the LS period, amplitude, and FAP
        dtr_stages_dict['lsp_dict'] = lsp_dict

    return search_time, search_flux, dtr_stages_dict


def _get_detrending_method(dtr_dict):
    """
    Get the string representation of the detrending method to be used. default
    to "best".
    """

    allowed_methods = ['best', 'notch', 'locor', 'pspline', 'biweight', 'none']

    if isinstance(dtr_dict, dict):
        if 'method' in dtr_dict.keys():
            dtr_method = dtr_dict['method']
        else:
            dtr_method = 'best'
    else:
        dtr_method = 'best'

    if dtr_method not in allowed_methods:
        raise NotImplementedError(
            f'{dtr_method} is not an implemented detrending method.'
        )

    return dtr_method


def _run_notch(TIME, FLUX, dtr_dict, verbose=False):

    from notch_and_locor.core import sliding_window

    #
    # HARD-CODE notch options (nb. could also let them be options via
    # dtr_dict).
    #
    # Set to True to do a full fit over time and arclength (default False).
    use_arclength = False
    # Internally, keep as "False" to use wdat.fcor as the flux.
    use_raw = False
    # BIC difference between transit and no-transit model required to
    # select the transit model
    min_deltabic = -1.0
    # By default (resolvabletrans == False), a grid of transit durations is
    # searched. [0.75, 1.0, 2.0, 4.0] hours.  If this is set to be True,
    # the 45 minute one is dropped.
    resolvable_trans = False
    # show_progress: if True, puts out a TQDM bar
    show_progress = verbose

    # Format "data" into recarray format needed for notch.
    N_points = len(TIME)
    data = np.recarray(
        (N_points,),
        dtype=[('t',float), ('fraw',float), ('fcor',float), ('s',float),
               ('qual',int), ('divisions',float)]
    )
    data.t = TIME
    data.fcor = FLUX
    data.fraw[:] = 0
    data.s[:] = 0
    data.qual[:] = 0

    # Run notch
    if verbose:
        LOGINFO('Beginning notch run...')
    fittimes, depth, detrend, polyshape, badflag = (
        sliding_window(
            data, windowsize=dtr_dict['window_length'],
            use_arclength=use_arclength, use_raw=use_raw,
            deltabic=min_deltabic, resolvable_trans=resolvable_trans,
            show_progress=show_progress
        )
    )
    if verbose:
        LOGINFO('Completed notch run.')

    assert len(fittimes) == len(TIME)

    # Store everything in a common format recarray
    N_points = len(detrend)
    notch = np.recarray(
        (N_points, ), dtype=[
            ('t', float), ('detrend', float), ('polyshape', float),
            ('notch_depth', float), ('deltabic', float), ('bicstat', float),
            ('badflag', int)
        ]
    )

    notch.t = data.t
    notch.notch_depth = depth[0].copy()
    notch.deltabic    = depth[1].copy()
    notch.detrend     = detrend.copy()
    notch.badflag     = badflag.copy()
    notch.polyshape   = polyshape.copy()

    bicstat = notch.deltabic-np.median(notch.deltabic)
    notch.bicstat = 1- bicstat/np.max(bicstat)

    #
    # Convert to my naming scheme.
    #
    flat_flux = notch.detrend
    trend_flux = notch.polyshape

    return flat_flux, trend_flux, notch


def _run_locor(TIME, FLUX, dtr_dict, lsp_dict):
    """
    NOTE: lsp_dict is created here if None is passed.
    """

    from notch_and_locor.core import rcomb

    # Format "data" into recarray format needed for notch.
    N_points = len(TIME)
    data = np.recarray(
        (N_points,),
        dtype=[('t',float), ('fraw',float), ('fcor',float), ('s',float),
               ('qual',int), ('divisions',float)]
    )
    data.t = TIME
    data.fcor = FLUX
    data.fraw[:] = 0
    data.s[:] = 0
    data.qual[:] = 0

    # Get rotation period, if not available.  In most cases, it should be
    # passed in via lsp_dict.
    if not isinstance(lsp_dict, dict):
        LOGINFO(
            "Did not get period from lsp_dict: finding via Lomb Scargle. "
            "WARNING: it's better to feed via lsp_dict for cacheing speeds. "
        )

        ls = LombScargle(TIME, FLUX, FLUX*1e-3)
        period_min, period_max = 0.1, 10
        freq, power = ls.autopower(minimum_frequency=1/period_max,
                                   maximum_frequency=1/period_min)
        ls_fap = ls.false_alarm_probability(power.max())
        best_freq = freq[np.argmax(power)]
        ls_period = 1/best_freq
        theta = ls.model_parameters(best_freq)
        ls_amplitude = theta[1]

        # NOTE: this just forces LOCOR to run irrespective of the exact
        # ls_period, ls_amplitude, ls_fap, or color. In other words, it
        # doesn't deal with the question of whether the star is young.  You
        # should do that elsewhere, and probably only be running LOCOR for
        # stars with Prot <~ a few days.

        lsp_dict['ls_period'] = ls_period
        lsp_dict['ls_amplitude'] = np.abs(ls_amplitude)
        lsp_dict['ls_fap'] = ls_fap

    wsize = lsp_dict['ls_period']

    #
    # minimum rotation period to bunch rotations together to run LOCoR.
    # 2 days works robustly for K2 long cadence data. TESS we shall see.
    #
    alias_num = 2.0

    # Run LOCOR (Locally Optimized Combination of Rotations)
    fittimes, depth, detrend, polyshape, badflag = (
        rcomb(
            data, wsize, aliasnum=alias_num
        )
    )

    assert len(fittimes) == len(TIME)

    # store everything in a common format recarray
    N_points = len(detrend)
    locor = np.recarray(
        (N_points, ),
        dtype=[('t', float), ('detrend', float), ('badflag', int)]
    )
    locor.detrend = detrend.copy()
    locor.badflag = badflag.copy()
    locor.polyshape = polyshape.copy()
    locor.t = data.t

    #
    # Convert to my naming scheme.
    #
    flat_flux = locor.detrend
    trend_flux = locor.polyshape

    return flat_flux, trend_flux, locor


def detrend_flux(time, flux, break_tolerance=0.5, method='pspline', cval=None,
                 window_length=None, edge_cutoff=None):
    """
    Apply the wotan flatten function. Implemented methods include pspline,
    biweight, and median.

    Args:
        time, flux (np.ndarray): array of times and fluxes.

        break_tolerance (float): maximum time past which light curve is split
        into timegroups, each of which is detrended individually.

        method (str): 'pspline', 'biweight', 'median'.

        cval (float): the wotan 'biweight' tuning parameter.

        window_length (float): length of the window in days.

        edge_cutoff (float): how much of the edge to remove. Only works for
        'median' method.

    Returns:
        flat_flux, trend_flux (np.ndarray): flattened array, and the trend
        vector that was divided out.

    See also:
        https://wotan.readthedocs.io/en/latest/
    """

    # Initial pre-processing: verify that under break_tolerance, time and flux
    # do not have any sections with <=6 points. Spline detrending routines do
    # not like fitting lines.
    N_groups, group_inds = find_lc_timegroups(time, mingap=break_tolerance)
    SECTION_CUTOFF = 6
    for g in group_inds:
        if len(time[g]) <= SECTION_CUTOFF:
            time[g], flux[g] = np.nan, np.nan

    try:
        if method == 'pspline':

            # Detrend each time segment individually, to prevent overfitting
            # based on the `max_splines` parameter.

            flat_fluxs, trend_fluxs = [], []
            for g in group_inds:
                tgtime, tgflux = time[g], flux[g]

                t_min, t_max = np.min(tgtime), np.max(tgtime)
                t_baseline = t_max - t_min
                transit_timescale = 6/24.

                # e.g., for a 25 day segment, we want max_splines to be ~100,
                # i.e., 1 spline point every 6 hours.  this helps prevent
                # overfitting.
                max_splines = int(t_baseline / transit_timescale)

                # a 2-sigma cutoff is standard, but there's no obvious reason for
                # this being the case. generally, anything far from the median
                # shouldn't go into the fit.
                stdev_cut = 1.5

                edge_cutoff = 0

                # note: major API update in wotan v1.6
                # pspline(time, flux, edge_cutoff, max_splines, stdev_cut, return_nsplines, verbose)
                _trend_flux, _nsplines = pspline(
                    tgtime, tgflux, edge_cutoff, max_splines, stdev_cut, False, False
                )

                _flat_flux = tgflux/_trend_flux

                flat_fluxs.append(_flat_flux)
                trend_fluxs.append(_trend_flux)

            flat_flux = np.hstack(flat_fluxs)
            trend_flux = np.hstack(trend_fluxs)


        elif method == 'biweight':
            flat_flux, trend_flux = flatten(
                time, flux, method='biweight', return_trend=True,
                break_tolerance=break_tolerance, window_length=window_length,
                cval=cval
            )

        elif method == 'median':
            flat_flux, trend_flux = flatten(
                time, flux, method='median', return_trend=True,
                break_tolerance=break_tolerance, window_length=window_length,
                edge_cutoff=edge_cutoff
            )

        elif method == 'none':
            flat_flux = flux
            trend_flux = None


        else:
            raise NotImplementedError

    except ValueError as e:
        msg = (
            'WRN! {}. Probably have a short segment. Trying to nan it out.'
            .format(repr(e))
        )
        LOGWARNING(msg)

        SECTION_CUTOFF = min([len(time[g]) for g in group_inds])
        for g in group_inds:
            if len(time[g]) <= SECTION_CUTOFF:
                time[g], flux[g] = np.nan, np.nan

        # NOTE: code duplication here
        if method == 'pspline':
            # matched detrending to do_initial_period_finding
            flat_flux, trend_flux = flatten(time, flux,
                                            method='pspline',
                                            return_trend=True,
                                            break_tolerance=break_tolerance,
                                            robust=True)
        elif method == 'biweight':
            # another option:
            flat_flux, trend_flux = flatten(time, flux,
                                            method='biweight',
                                            return_trend=True,
                                            break_tolerance=break_tolerance,
                                            window_length=window_length,
                                            cval=cval)
        else:
            raise NotImplementedError

    return flat_flux, trend_flux


def detrend_systematics(lcpath, max_n_comp=5, infodict=None,
                        enforce_hdrfmt=True):
    """
    Wraps functions in lcproc.detrend for all-in-one systematics removal, using
    a tuned variant of PCA.

    See doc/20201109_injectionrecovery_completeness_goldenvariability.txt for a
    verbose explanation of the options that were explored, and the assumptions
    that were ultimately made for this "tuned variant".

    Returns:
        Tuple of primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs
    """

    if infodict is None:
        from cdips.utils import given_lcpath_get_infodict
        infodict = given_lcpath_get_infodict(lcpath)

    eigveclist, smooth_eigveclist, n_comp_df = prepare_pca(
        infodict['CAMERA'], infodict['CCD'],
        infodict['SECTOR'], infodict['PROJID']
    )

    sysvecnames = ['BGV']
    dtrvecs, sysvecs, ap, primaryhdr, data, eigenvecs = (
        get_dtrvecs(lcpath, eigveclist, smooth_eigveclist,
                    sysvecnames=sysvecnames)
    )
    time, y = data['TMID_BJD'], data[f'IRM{ap}']

    n_components = min([int(n_comp_df[f'fa_cv_ap{ap}']), max_n_comp])
    n_components += len(sysvecnames)

    model_mag, n_comp = calculate_linear_model_mag(
        y, dtrvecs, n_components, method='LinearRegression'
    )

    try:
        data[f'PCA{ap}'] = model_mag
    except KeyError:
        data = np.array(data)
        _d = rfn.append_fields(data, f'PCA{ap}', model_mag, dtypes='>f8')
        data = _d.data

    primaryhdr[f'PCA{ap}NCMP'] = (
        n_comp,
        f'N principal components PCA{ap}'
    )

    if enforce_hdrfmt and 'TESSMAG' not in primaryhdr:
        #
        # Clean up headers on non-HLSP light curves for ease of use downstream.
        #
        alreadyhas = ['Gaia-ID', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                      'phot_rp_mean_mag', 'teff_val', 'PM_RA[mas/yr]',
                      'PM_Dec[mas/year]']
        for a in alreadyhas:
            assert a in primaryhdr

        primaryhdr.set('RA_OBJ', primaryhdr['RA[deg]'])
        primaryhdr.set('DEC_OBJ', primaryhdr['Dec[deg]'])

        Tmag_pred = (primaryhdr['phot_g_mean_mag']
                    - 0.00522555 * (primaryhdr['phot_bp_mean_mag'] - primaryhdr['phot_rp_mean_mag'])**3
                    + 0.0891337 * (primaryhdr['phot_bp_mean_mag'] - primaryhdr['phot_rp_mean_mag'])**2
                    - 0.633923 * (primaryhdr['phot_bp_mean_mag'] - primaryhdr['phot_rp_mean_mag'])
                    + 0.0324473)

        # a bit janky, but better than dealing with a TIC8 crossmatch for speed
        LOGWARNING('Forcing TESSMAG to be Tmag_pred')
        primaryhdr.set('TESSMAG', Tmag_pred)


    return primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs


def compute_scores(X, n_components):
    pca = PCA()
    fa = FactorAnalysis()
    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
        fa.n_components = n
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=5)))

    return pca_scores, fa_scores


def insert_nans_given_rstfc(mag, mag_rstfc, full_rstfc):
    """
    args:

        mag: the vector of magnitudes that needs nans inserted

        mag_rstfc: RSTFC frame id vector of same length as mag

        full_rstfc: RSTFC frame id vector of larger length, that will be
        matched against.

    returns:

        full_mag: vector of magnitudes with nans inserted, of same length as
        full_rstfc
    """

    if not len(full_rstfc) >= len(mag_rstfc):

        raise AssertionError('full_rstfc needs to be the big one')

    if len(full_rstfc) == len(mag_rstfc):

        return mag

    else:
        # input LC has too few frame stamps (for whatever reason -- often
        # because NaNs are treated differently by VARTOOLS TFA or other
        # detrenders, so they are omitted). The following code finds entries in
        # the TFA lightcurve that are missing frameids, and put nans in those
        # cases. It does this by first making an array of NaNs with length
        # equal to the original RAW data. It then puts the magnitude values at
        # the appropriate indices, through the frame-id matching ("RSTFC" ids).

        inarr = np.in1d(full_rstfc, mag_rstfc)

        inds_to_put = np.argwhere(inarr).flatten()

        full_mag = (
            np.ones_like(np.arange(len(full_rstfc),
                                   dtype=np.float))*np.nan
        )

        full_mag[ inds_to_put ] = mag

        wrn_msg = (
            '{} WRN!: found missing frameids. added NaNs'.
            format(datetime.utcnow().isoformat())
        )
        LOGWARNING(wrn_msg)

        return full_mag


def prepare_pca(cam, ccd, sector, projid, N_to_make=20):
    """
    This function:

        * calculates light curve principal components using TFA template
        stars given in "trendlist_tfa_ap[1-3].txt"

        * writes each set to /statsdir/pca_data/principal_component_ap[1-3].txt

        * calculates the optimal number of principal components to use based on
        a factor analysis cross-validation.

        * writes the optimal number, for each aperture, to
        /statsdir/pca_data/optimal_n_components.csv

        * makes plots showing the effect of using different numbers of trend
        stars for N_to_make random stars from the cam/ccd/sector/projid
        combination.

    Args:

        cam,ccd,sector,projid: ints

        N_to_make: integer number of plots showing the effects of adding more
        principal components to the fit.

    Returns:

        (eigveclist, optimal_n_comp_df):

            eigveclist = [eigenvecs_ap1, eigenvecs_ap2, eigenvecs_ap3] where
            each element is a np.ndarray.

            optimal_n_comp_df: dataframe.

    Using them, do linear least squares (or a variant thereof) to get
    components matched to each LC.
    """

    lcdir = ('/nfs/phtess2/ar0/TESS/FFI/LC/FULL/s{}/ISP_{}-{}-{}/'.
             format(str(sector).zfill(4), cam, ccd, projid))

    statsdir = os.path.join(lcdir, 'stats_files')

    pcadir = os.path.join(statsdir, 'pca_data')
    if not os.path.exists(pcadir):
        os.mkdir(pcadir)

    csvpath = os.path.join(pcadir, 'optimal_n_components.csv')

    comppaths = [
        os.path.join(pcadir, f'principal_component_ap{ap}.txt')
        for ap in range(1,4)
    ]
    smoothpaths = [
        os.path.join(pcadir, f'principal_component_ap{ap}_smoothed.txt')
        for ap in range(1,4)
    ]
    comppaths_exist = np.all([os.path.exists(p) for p in comppaths])
    smoothpaths_exist = np.all([os.path.exists(p) for p in smoothpaths])

    if os.path.exists(csvpath) and comppaths_exist and smoothpaths_exist:

        eigveclist = [np.genfromtxt(f) for f in comppaths]
        smooth_eigveclist = [np.genfromtxt(f) for f in smoothpaths]
        n_comp_df = pd.read_csv(csvpath)

        return eigveclist, smooth_eigveclist, n_comp_df

    #
    # path, x, y. space-separated. for all ~30k light curves to do TFA on.
    #
    tfalclist_path = os.path.join(statsdir,'lc_list_tfa.txt')
    #
    # key, time
    #
    datestfa_path = os.path.join(statsdir,'dates_tfa.txt')

    eigveclist, smooth_eigveclist, optimal_n_comp = [], [], {}
    for ap in [1,2,3]:

        #
        # path, x, y. space-separated. for 200 TFA template stars.
        #
        trendname = 'trendlist_tfa_ap{}.txt'.format(ap)
        trendlisttfa = os.path.join(statsdir,trendname)

        df_template_stars = pd.read_csv(trendlisttfa, sep=' ', header=None,
                                        names=['path','x','y'])

        df_dates = pd.read_csv(datestfa_path, sep=' ', header=None,
                               names=['rstfc','btjd'])

        lcpaths = glob(os.path.join(lcdir, '*_llc.fits'))

        #
        # prepare data as a (N_template_stars x N_times) matrix. We have N~=200
        # template light curves, with K=N_times measurements in each. Think of
        # these as N vectors in a K-dimensional space. I.e. the flux at each
        # time point is a "measured feature". So we have N samples (light
        # curves), and K features (points per light curve).
        #

        mags = nparr(
            list(
                map(iu.get_data_keyword,
                    nparr(df_template_stars['path']), # file,
                    np.repeat('IRM{}'.format(ap), len(df_template_stars)), # keyword
                    np.repeat(1, len(df_template_stars)) # extension
                   )
            )
        )
        mag_rstfc = nparr(
            list(
                map(iu.get_data_keyword,
                    nparr(df_template_stars['path']), # file,
                    np.repeat('RSTFC', len(df_template_stars)), # keyword
                    np.repeat(1, len(df_template_stars)) # extension
                   )
            )
        )

        #
        # construct a "mean time" vector in order to smooth the eigenvectors --
        # call it mean_tmid_bjd.  require that average time-difference across
        # the reference light curves is not more than one second.
        #
        tmid_bjd =  nparr(
            list(
                map(iu.get_data_keyword,
                    nparr(df_template_stars['path']), # file,
                    np.repeat('TMID_BJD', len(df_template_stars)), # keyword
                    np.repeat(1, len(df_template_stars)) # extension
                   )
            )
        )
        TOLERANCE_SEC = 1
        assert (
            np.abs(np.diff(tmid_bjd, axis=0).mean(axis=0)).max() * 24*60*60
            <
            TOLERANCE_SEC
        )
        mean_tmid_bjd = tmid_bjd.mean(axis=0)

        #
        # for the fit, require that for each tempalte light curve is only made
        # of finite values. this might drop a row or two.
        #
        fmags = mags[~np.isnan(mags).any(axis=1)]

        #
        # subtract mean, as is standard in PCA.
        #
        mean_mags = np.nanmean(fmags, axis=1)
        X = fmags - mean_mags[:, None]

        pca = PCA()
        pca.fit(X)

        #
        # (200 x N_times) eigenvector matrix. these are basis vectors for the
        # original data. can use factor analysis components instead, without
        # much differnece.
        #

        eigenvecs = pca.components_

        #
        # Save all 200 PCA eigenvectors.
        #
        comppath = os.path.join(pcadir, f'principal_component_ap{ap}.txt')
        if not os.path.exists(comppath):
            np.savetxt(comppath, eigenvecs)
            LOGINFO(f'saved {comppath}')
        else:
            LOGINFO(f'found {comppath}')

        eigveclist.append(eigenvecs)

        #
        # Smooth the eigenvectors and cache the result as well.
        #
        smooth_eigenvecs = []
        for e in eigenvecs:
            smooth_eigenvec = eigvec_smooth_fn(mean_tmid_bjd, e)
            smooth_eigenvecs.append(smooth_eigenvec-1)
        smooth_eigenvecs = np.array(smooth_eigenvecs)
        assert not np.any(pd.isnull(smooth_eigenvecs))
        assert smooth_eigenvecs.shape == eigenvecs.shape

        smoothpath = os.path.join(pcadir, f'principal_component_ap{ap}_smoothed.txt')
        if not os.path.exists(smoothpath):
            np.savetxt(smoothpath, smooth_eigenvecs)
            LOGINFO(f'saved {smoothpath}')
        else:
            LOGINFO(f'found {smoothpath}')

        smooth_eigveclist.append(eigenvecs)

        #
        # plot a sequence of reconstructions, for a set of random light curves
        #
        for i in range(N_to_make):

            if ap != 2:
                continue
            savpath = os.path.join(pcadir,
                                   'test_reconstruction_{}_ap{}.png'.
                                   format(i, ap))
            if os.path.exists(savpath):
                continue

            np.random.seed(i)
            this_lcpath = np.random.choice(lcpaths, size=1)[0]
            mag = iu.get_data_keyword(this_lcpath, 'IRM{}'.format(ap))
            mag_rstfc = iu.get_data_keyword(this_lcpath, 'RSTFC')

            if np.all(pd.isnull(mag)):
                while np.all(pd.isnull(mag)):
                    next_lcpath = np.random.choice(lcpaths, size=1)[0]
                    mag = iu.get_data_keyword(next_lcpath, 'IRM{}'.format(ap))
                    mag_rstfc = iu.get_data_keyword(next_lcpath, 'RSTFC')

            mean_mag = np.nanmean(mag)
            mag = mag - mean_mag
            mag = mag[~pd.isnull(mag)]

            component_list = [1, 2, 4, 8, 12, 16, 20]

            plt.close('all')
            f, axs = plt.subplots(nrows=len(component_list), ncols=2,
                                  sharex=True, figsize=(8,9))

            for n_components, ax, ax_r in zip(component_list, axs[:,0], axs[:,1]):
                #
                # eigenvecs shape: 200 x N_times
                #
                # model: 
                # y(w, x) = w_0 + w_1 x_1 + ... + w_p x_p.
                #
                # X is matrix of (n_samples, n_features).
                #

                # either linear regression or bayesian ridge regression seems fine
                reg = LinearRegression(fit_intercept=True)
                #reg = BayesianRidge(fit_intercept=True)

                # n.b., you need to simultaneously cast the eigenvectors and
                # magnitudes to the same shape of nans [?!]
                y = mag
                _X = eigenvecs[:n_components, :]

                try:
                    reg.fit(_X.T, y)
                except Exception as e:
                    LOGEXCEPTION(e)
                    LOGEXCEPTION(n_components)
                    continue

                model_mag = reg.intercept_ + (reg.coef_ @ _X)

                # given "true" (full) RSTFC list, and the actual list
                # ("rstfc" above), need a function that gives model mags
                # (or data mags) with nans in correct place

                time = nparr(df_dates['btjd'])
                full_rstfc = nparr(df_dates['rstfc'])

                full_mag = insert_nans_given_rstfc(mag, mag_rstfc, full_rstfc)

                full_model_mag = insert_nans_given_rstfc(
                    model_mag, mag_rstfc, full_rstfc
                )

                ax.scatter(time, full_mag + mean_mag, c='k', alpha=0.9,
                           zorder=2, s=1, rasterized=True, linewidths=0)
                ax.plot(time, full_model_mag + mean_mag, c='C0', zorder=1,
                        rasterized=True, lw=0.5, alpha=0.7 )

                txt = '{} components'.format(n_components)
                ax.text(0.02, 0.02, txt, ha='left', va='bottom',
                        fontsize='medium', transform=ax.transAxes)

                ax_r.scatter(time, full_mag-full_model_mag, c='k', alpha=0.9, zorder=2,
                             s=1, rasterized=True, linewidths=0)
                ax_r.plot(time, full_model_mag-full_model_mag, c='C0', zorder=1,
                          rasterized=True, lw=0.5, alpha=0.7)

            for a in axs[:,0]:
                a.set_ylabel('raw mag')
            for a in axs[:,1]:
                a.set_ylabel('resid')
            for a in axs.flatten():
                a.get_yaxis().set_tick_params(which='both', direction='in')
                a.get_xaxis().set_tick_params(which='both', direction='in')

            f.text(0.5,-0.01, 'BJDTDB [days]', ha='center')

            f.tight_layout(h_pad=0, w_pad=0.5)
            f.savefig(savpath, dpi=350, bbox_inches='tight')
            LOGINFO('made {}'.format(savpath))

        #
        # make plot to find optimal number of components. write output to...
        # /statsdir/pca_data/optimal_n_components.csv
        # see
        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
        #
        n_components = np.arange(0,50,1)

        pca_scores, fa_scores = compute_scores(X, n_components)

        plt.close('all')
        f,ax = plt.subplots(figsize=(4,3))

        ax.plot(n_components, pca_scores, label='PCA CV score')
        ax.plot(n_components, fa_scores, label='FA CV score')

        ax.legend(loc='best')
        ax.set_xlabel('N components')
        ax.set_ylabel('Cross-validation score')

        f.tight_layout(pad=0.2)
        savpath = os.path.join(pcadir, 'test_optimal_n_components.png')
        f.savefig(savpath, dpi=350, bbox_inches='tight')
        LOGINFO('made {}'.format(savpath))

        #
        # write the factor analysis maximum to a dataframe
        #
        n_components_pca_cv = n_components[np.argmax(pca_scores)]
        n_components_fa_cv = n_components[np.argmax(fa_scores)]

        LOGINFO('n_components_pca_cv: {}'.format(n_components_pca_cv))
        LOGINFO('n_components_fa_cv: {}'.format(n_components_fa_cv))

        optimal_n_comp['pca_cv_ap{}'.format(ap)] = n_components_pca_cv
        optimal_n_comp['fa_cv_ap{}'.format(ap)] = n_components_fa_cv

    optimal_n_comp_df = pd.DataFrame(optimal_n_comp, index=[0])
    optimal_n_comp_df.to_csv(csvpath, index=False)
    LOGINFO('made {}'.format(csvpath))

    return eigveclist, smooth_eigveclist, optimal_n_comp_df


def get_dtrvecs(lcpath, eigveclist, smooth_eigveclist, sysvecnames=['BGV'],
                use_smootheigvecs=True, ap=None):
    """
    Given a CDIPS light curve file, and the PCA eigenvectors for this
    sector/cam/ccd, construct the vectors to "detrend" or "decorrelate" against
    (presumably via a linear model).

    Args:
        lcpath: CDIPS light curve file ("calibration", or HLSP format)

        eigveclist: list of np.ndarray PCA eigenvectors, length 3, calculated
        by a call to cdips.lcutils.detrend.prepare_pca.

        smooth_eigveclist: as above, but smoothed using `eigvec_smooth_fn`

        sysvecnames: list of vector names to also be decorrelated against.
        E.g., ['BGV', 'CCDTEMP', 'XIC', 'YIC']. Default is just ['BGV'].

        use_smootheigvecs: whether or not to smooth the PCA eigenvectors, using
        a windowed biweight filter.

        ap (int): 1, 2, 3, or None.  If None, tries to get best aperture.

    Returns:
        tuple containing:
        dtrvecs (np.ndarray), sysvecs (np.ndarray), ap (the optimal aperture),
        data (the entire lcpath data table), eigenvecs (np.ndarray chosen from
        eigveclist given the optimal aperture), smooth_eigenvecs (np.ndarray).
    """

    from cdips.utils.lcutils import get_best_ap_number_given_lcpath

    if DEBUG:
        _t = datetime.utcnow().isoformat()
        LOGDEBUG(f'{_t}: ap {ap}, lcpath {lcpath}: started get_dtrvecs')

    hdul = fits.open(lcpath)
    primaryhdr, hdr, data = (
        hdul[0].header, hdul[1].header, hdul[1].data
    )
    hdul.close()
    if ap is None:
        try:
            best_ap = get_best_ap_number_given_lcpath(lcpath)
        except TypeError:
            best_ap = 1
        ap = np.nanmin([best_ap, 2])
    else:
        pass

    ##########################################
    # eigenvecs shape: N_templates x N_times
    #
    # model: 
    # y(w, x) = w_0 + w_1 x_1 + ... + w_p x_p.
    #
    # X is matrix of (n_samples, n_features), so each template is a "sample",
    # and each time is a "feature". Analogous to e.g., doing PCA to reconstruct
    # a spectrum, and you want each wavelength bin to be a feature.
    ##########################################
    eigenvecs = eigveclist[ap-1]

    if np.any(pd.isnull(eigenvecs)):
        raise ValueError('got nans in eigvecs. bad!')

    if use_smootheigvecs:
        smooth_eigenvecs = smooth_eigveclist[ap-1]
    else:
        smooth_eigenvecs = None

    use_sysvecs = True if isinstance(sysvecnames, list) else False

    if DEBUG:
        _t = datetime.utcnow().isoformat()
        LOGDEBUG(f'{_t} ap {ap}, lcpath {lcpath}: begin scaling')

    if use_sysvecs:

        # Use the (0-1 scaled) systematic vectors directly. Don't smooth.
        sysvecs = np.vstack(
            [
                MinMaxScaler().fit_transform(
                    data[s][:,None].astype(np.float64)
                ).flatten()
                -
                np.nanmedian(MinMaxScaler().fit_transform(
                    data[s][:,None].astype(np.float64)
                ).flatten())
                for s in sysvecnames
            ]
        )

        if use_smootheigvecs:
            dtrvecs = np.vstack([sysvecs, smooth_eigenvecs])
        else:
            dtrvecs = np.vstack([sysvecs, eigenvecs])

    else:

        sysvecs = None

        if use_smootheigvecs:
            dtrvecs = smooth_eigenvecs
        else:
            dtrvecs = eigenvecs

    if DEBUG:
        _t = datetime.utcnow().isoformat()
        LOGDEBUG(f'{_t} ap {ap}, lcpath {lcpath}: end scaling')

    return dtrvecs, sysvecs, ap, primaryhdr, data, eigenvecs, smooth_eigenvecs


def eigvec_smooth_fn(time, eigenvec, window_length=1, cval=6):
    """
    lower cval: includes less data
    the biweight a cval of 6 includes data up to 4 standard deviations (6
    median absolute deviations) from the central location and has an
    efficiency of 98%
    """

    _, smoothed = flatten(
        time-np.nanmin(time), 1+eigenvec,
        break_tolerance=0.5,
        method='biweight', window_length=window_length, cval=cval, edge_cutoff=0,
        return_trend=True
    )

    return smoothed


def calculate_linear_model_mag(y, basisvecs, n_components,
                               method='LinearRegression', verbose=False):
    """
    Given a set of basis vectors in a linear model for a target light curve
    (y), calculate the coefficients and apply the linear model prediction.

    `basisvecs` needs to have shape: N_templates x N_times

    model:
    y(w, x) = w_0 + w_1 x_1 + ... + w_p x_p.

    where X is matrix of (n_samples, n_features), so each template is a
    "sample", and each time is a "feature". Analogous to e.g., doing PCA to
    reconstruct a spectrum, and you want each wavelength bin to be a feature.

    Allow methods:

        'LinearRegression': ordinary least squares.

        'RidgeCV': ridge regression, which is ordinary least squares, plus an
        L2 norm with a regularization coefficient. The regularization is solved
        for via a cross-validation grid search.

    Returns:
        out_mag, n_comp: the model light curve (supposedly with instrumental
        systematic removed), and the number of PCA components used during the
        regression.
    """

    if method == 'LinearRegression':
        reg = LinearRegression(fit_intercept=True)
    elif method == 'RidgeCV':
        reg = RidgeCV(alphas=np.logspace(-10, 10, 210), fit_intercept=True)

    y2 = y
    y2[y2 == np.inf] = np.nan
    y2[y2 == -np.inf] = np.nan

    if np.all(pd.isnull(y2)):
        # if all nan, job is easy
        out_mag = y
        n_comp = 'nan'

    elif np.sum(~pd.isnull(y2)) <= 10:
        # fewer than 10 non-nan points.  treat as all nan for PCA.
        out_mag = np.ones_like(y)*np.nan
        n_comp = 'nan'

    elif np.sum(~pd.isnull(y2)) > 10 and np.any(pd.isnull(y2)):

        #
        # if some nan in target light curve, tricky. strategy: impose the
        # same nans onto the eigenvectors. then fit without nans. then put
        # nans back in.
        #
        mean_mag = np.nanmean(y2)
        std_mag = np.nanstd(y2)
        norm_mag = (y2[~pd.isnull(y2)] - mean_mag)/std_mag

        _X = basisvecs[:n_components, :]

        _X = _X[:, ~pd.isnull(y2)]

        reg.fit(_X.T, norm_mag)

        # see note below in the else statement about how the prediction step
        # works.
        model_mag = mean_mag + std_mag*reg.predict(_X.T)

        #
        # now insert nans where they were in original target light curve.
        # typically nans occur in groups. insert by groups. this is a
        # tricky procedure, so test after to ensure nan indice in the model
        # and target light curves are the same.
        #
        naninds = np.argwhere(pd.isnull(y2)).flatten()

        ngroups, groups = find_lc_timegroups(naninds, mingap=1)

        for group in groups:

            thisgroupnaninds = naninds[group]

            model_mag = np.insert(model_mag,
                                  np.min(thisgroupnaninds),
                                  [np.nan]*len(thisgroupnaninds))

        np.testing.assert_(
            model_mag.shape == y.shape
        )
        np.testing.assert_(
            (len((model_mag-y2)[pd.isnull(model_mag-y2)])
             ==
             len(y[pd.isnull(y2)])
            ),
            'got different nan indices in model mag than target mag'
        )

        #
        # save result as IRM mag - PCA model mag + mean IRM mag
        #
        out_mag = y - model_mag + mean_mag
        n_comp = n_components

    else:
        # otherwise, just directly fit the principal components
        mean_mag = np.nanmean(y)
        std_mag = np.nanstd(y)
        norm_mag = (y - mean_mag)/std_mag

        _X = basisvecs[:n_components, :]

        reg.fit(_X.T, norm_mag)

        # NOTE: in ordinary least squares, the line below is the same as
        # `model_mag = reg.intercept_ + (reg.coef_ @ _X)`. But the sklearn
        # interface is sick, and using reg.predict is easier than doing the
        # extra terms for other regressors.
        model_mag = mean_mag + std_mag * reg.predict(_X.T)

        out_mag = y - model_mag + mean_mag
        n_comp = n_components

    if verbose:
        if method == 'RidgeCV':
            try:
                LOGINFO(f'RidgeCV alpha: {reg.alpha_:.2e}')
            except AttributeError:
                pass

    return out_mag, n_comp


def transit_mask(t, period, duration, T0):
    """
    :t: *(array)* Time series of the data (in units of days)
    :period: *(float)* Transit period e.g. from results: ``period``
    :duration: *(float)* Transit duration e.g. from results: ``duration``
    :T0: *(float)* Mid-transit of first transit e.g. from results: ``T0``

    Returns

    :intransit: *(numpy array mask)* A numpy array mask (of True/False values)
    for each data point in the time series. ``True`` values are in-transit.

    Example usage:

    ::

        intransit = transit_mask(t, period, duration, T0)
        print(intransit)
        >>> [False False False ...]
        plt.scatter(t[in_transit], y[in_transit], color='red')  # in-transit points in red
        plt.scatter(t[~in_transit], y[~in_transit], color='blue')  # other points in blue
    """
    mask = np.abs((t - T0 + 0.5*period) % period - 0.5*period) < 0.5 * duration
    return mask


def transit_window_polynomial_remover(
    time, flux, flux_err, t0, period, tdur, n_tdurs=5., method='poly_2',
    plot_outpath=None, N_model_points=1000, drop_badtransits=False
    ):
    """
    Given a light curve and good guesses for {t0,P,t_14}, trim it to windows
    around each transit, mask out the points in-transit, and fit (chi-squared)
    local polynomials to each window.  Optionally, drop transits for which the
    fit is determined to be "bad".

    Args:
        time, flux, flux_err (np.ndarray): self-described.

        t0, period, tdur (float): self-described.  units: days.  tdur=t_14.

        n_tdurs (float): n where selecting: [ t0 - n*tdur, t + n*tdur ]

        method (str): 'poly_{N}' where N is Nth-order polynomial. E.g., poly_2,
        to fit a local quadratic.

        plot_outpath (None or str): if str, e.g,
        '/path/to/TIC123_somestr.png', light curve plots before and after
        trimming will be made (TIC123_somestr_rawlc.png and *rawtrimlc.png).

        drop_badtransits (bool or dict): False or a dict.  Example dict would
        be {'min_pts_in_transit':1, 'drop_worst_rms_percentile':0.9,
        'badtimewindows':[(1,2),(1001,1002)], 'x0':2457000}, which would require at least 1
        point in transit, would then also drop the top 10% of transits
        according to the RMS of the residual after subtracting the local
        quadratic, and would drop times (in the same time system as `time`)
        between 1-2 days, and 1001-1002 days.  If 'x0' is present, it'll be
        applied as an offset to the times.

    Returns:

        OrderedDict with keys:

        'time_{ix}', 'flux_{ix}', 'flux_err_{ix}': each {ix} is a transit
            window, this is the data.

        'trend_flux_{ix}', 'flat_flux_{ix}': the model, evaluated at time_{ix}.

        'coeffs_{ix}': coefficients of the model

        'mod_time_{ix}', 'mod_flux_{ix}': the model, evaluated over
            N_model_points=1000 points in each transit window.  Used for
            visualization purposes.

        'ngroups', and 'groupinds'.
    """

    # check types
    for p in [time, flux, flux_err]:
        assert isinstance(p, np.ndarray)
    for p in [t0, period, tdur]:
        assert isinstance(p, (int, float))
    if isinstance(plot_outpath, str):
        assert plot_outpath.endswith('.png')
    assert method.startswith('poly_')
    assert drop_badtransits == False or isinstance(drop_badtransits, dict)

    # trim
    from betty.helpers import _subset_cut, _quicklcplot

    if isinstance(plot_outpath, str):
        outpath = plot_outpath.replace('.png', '_rawlc.png')
        _quicklcplot(time, flux, flux_err, outpath)

    time, flux, flux_err = _subset_cut(
        time, flux, flux_err, n=n_tdurs, t0=t0, per=period, tdur=tdur
    )

    # drop bad times, if present
    if isinstance(drop_badtransits, dict):
        if 'badtimewindows' in drop_badtransits.keys():
            if 'x0' in drop_badtransits.keys():
                x0 = drop_badtransits['x0']
            else:
                x0 = 0

            keep = np.ones_like(time).astype(bool)
            for window in drop_badtransits['badtimewindows']:
                t_lower = min(window) + x0
                t_upper = max(window) + x0
                keep &= ~( (time > t_lower) & (time < t_upper) )

            LOGINFO(f"Dropping windows: {drop_badtransits['badtimewindows']}")
            LOGINFO(f'Before drop, had {len(keep)} points')
            LOGINFO(f'After drop, had {len(keep[keep])} points')
            LOGINFO(42*'#')

            time = time[keep]
            flux = flux[keep]
            flux_err = flux_err[keep]

    if isinstance(plot_outpath, str):
        outpath = plot_outpath.replace('.png', '_rawtrimlc.png')
        _quicklcplot(time, flux, flux_err, outpath)

    # construct mask
    from astrobase.lcmath import find_lc_timegroups

    mingap = period - 2.01*n_tdurs*tdur
    msg = f"P={period}, N={n_tdurs}, Tdur={tdur}"
    assert mingap > 0, msg
    ngroups, groupinds = find_lc_timegroups(time, mingap=mingap)
    if isinstance(drop_badtransits, dict):
        init_ngroups = deepcopy(ngroups)
        init_groupinds = deepcopy(groupinds)

    # drop groups with less than a certain number of points *in-transit*.
    if isinstance(drop_badtransits, dict):
        N_in_transits = []
        for ix, g in enumerate(groupinds):
            _t, _f, _e = (
                time[g].astype(np.float64),
                flux[g].astype(np.float64),
                flux_err[g].astype(np.float64),
            )
            in_transit = transit_mask(_t, period, tdur, t0)
            N_in_transit = len(_t[in_transit])
            N_in_transits.append(N_in_transit)

        N_in_transits = np.array(N_in_transits)
        N_CUT = drop_badtransits['min_pts_in_transit']
        ok_transits = N_in_transits >= N_CUT

        LOGINFO(f'Initially got {ngroups} transit windows...')
        LOGINFO(f'Cut #1: Requiring >= {N_CUT} points in transit...')
        LOGINFO(f'yields {np.sum(ok_transits)} transit windows...')

        # apply the actual mask
        ngroups = int(np.sum(ok_transits))
        groupinds = [g for g,m in zip(groupinds,ok_transits) if m]
        assert len(groupinds) == ngroups

    # drop groups for which the polynomial fit yields RELATIVELY bad results --
    # i.e., relative to the other transits
    import numpy.polynomial.polynomial as poly

    if isinstance(drop_badtransits, dict):

        oot_stdevs = []
        for ix, g in enumerate(groupinds):

            _t, _f, _e = (
                time[g].astype(np.float64),
                flux[g].astype(np.float64),
                flux_err[g].astype(np.float64),
            )
            in_transit = transit_mask(_t, period, tdur, t0)

            # chi-squared local polynomial to the points out of transit, in this
            # window.
            _tmid = np.nanmedian(_t)

            poly_order = int(method.split("_")[1])
            coeffs = poly.polyfit(_t[~in_transit]-_tmid, _f[~in_transit], poly_order)

            _trend_flux = poly.polyval(_t - _tmid, coeffs)
            _flat_flux = _f / _trend_flux

            stdev_oot_flat_flux = np.nanstd( _flat_flux[~in_transit] )
            oot_stdevs.append(stdev_oot_flat_flux)

        oot_stdevs = np.array(oot_stdevs)
        pctile = drop_badtransits['drop_worst_rms_percentile']
        CUT_PERCENTILE = np.nanpercentile(oot_stdevs, pctile)
        ok_transits = oot_stdevs <= CUT_PERCENTILE

        LOGINFO(f'Cut #2: Requiring <={pctile:.1f} RMS pctile on OOT flat flux...')
        LOGINFO(f'Translates to stdev(oot) <= {CUT_PERCENTILE:.6f}...')
        LOGINFO(f'yields {np.sum(ok_transits)} transit windows...')

        # apply the actual mask
        ngroups = int(np.sum(ok_transits))
        groupinds = [g for g,m in zip(groupinds,ok_transits) if m]
        assert len(groupinds) == ngroups

    #
    # finally, construct the OrderedDict containing the trimmed data and the
    # models over the transit windows
    #
    datadict = OrderedDict()

    for ix, g in enumerate(groupinds):

        _t, _f, _e = (
            time[g].astype(np.float64),
            flux[g].astype(np.float64),
            flux_err[g].astype(np.float64),
        )
        in_transit = transit_mask(_t, period, tdur, t0)

        # chi-squared local polynomial to the points out of transit, in this
        # window.
        _tmid = np.nanmedian(_t)

        poly_order = int(method.split("_")[1])
        coeffs = poly.polyfit(_t[~in_transit]-_tmid, _f[~in_transit], poly_order)

        _trend_flux = poly.polyval(_t - _tmid, coeffs)
        _flat_flux = _f / _trend_flux

        x_mod = np.linspace(_t.min(), _t.max(), N_model_points)
        y_mod = poly.polyval(x_mod - _tmid, coeffs)

        datadict[f'time_{ix}'] = _t
        datadict[f'flux_{ix}'] = _f
        datadict[f'flux_err_{ix}'] = _e
        datadict[f'trend_flux_{ix}'] = _trend_flux
        datadict[f'flat_flux_{ix}'] = _flat_flux
        datadict[f'coeffs_{ix}'] = coeffs
        datadict[f'mod_time_{ix}'] = x_mod
        datadict[f'mod_flux_{ix}'] = y_mod

    datadict['ngroups'] = int(ngroups)
    datadict['groupinds'] = groupinds
    if isinstance(drop_badtransits,dict):
        datadict['init_ngroups'] = init_ngroups
        datadict['groupinds'] = groupinds

    return datadict
