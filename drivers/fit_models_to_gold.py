import os, pickle, h5py, json, shutil, requests
from glob import glob
from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=False)

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from numpy import array as nparr
from astropy.io import fits
from astropy import units as u, constants as const

from astrobase import periodbase, lcfit
from astrobase.services.tic import tic_single_object_crossmatch
from astrobase.varbase.transits import get_snr_of_dip
from astrobase.varbase.transits import estimate_achievable_tmid_precision
from astrobase.plotbase import plot_phased_magseries
from astrobase.lcmath import sigclip_magseries
from astrobase.varbase.transits import get_transit_times
from astrobase.lcfit.utils import make_fit_plot

from astroquery.mast import Catalogs
from astroquery.vizier import Vizier

from numpy.polynomial.legendre import Legendre
from scipy import optimize
from scipy.interpolate import interp1d

from astropy import units as u
from astropy.coordinates import SkyCoord

from cdips.lcproc import detrend as dtr
from cdips.lcproc import mask_orbit_edges as moe
from cdips.plotting import vetting_pdf as vp
from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils import today_YYYYMMDD

import make_vetting_multipg_pdf as mvp

"""
MCMC fit Mandel-Agol transits to gold above -- these parameters are used for
paper and CTOIs.

(pulls from tessorbitaldecay/measure_transit_times_from_lightcurve.py repo)
"""

CLASSIFXNDIR = "/home/lbouma/proj/cdips/results/vetting_classifications"

def main(overwrite=0, sector=7, nworkers=40, cdipsvnum=1, cdips_cat_vnum=0.3):

    lcbasedir, tfasrdir, resultsdir = _define_and_make_directories(sector)

    df, cdips_df, pfdf, supplementstatsdf, toidf, ctoidf = _get_data(
        sector, cdips_cat_vnum=cdips_cat_vnum)

    tfa_sr_paths = _get_lcpaths(df, tfasrdir)

    for tfa_sr_path in tfa_sr_paths:

        sourceid = int(tfa_sr_path.split('gaiatwo')[1].split('-')[0].lstrip('0'))
        mdf = cdips_df[cdips_df['source_id']==sourceid]
        if len(mdf) != 1:
            errmsg = 'expected exactly 1 source match in CDIPS cat'
            raise AssertionError(errmsg)

        hdul = fits.open(tfa_sr_path)
        hdr = hdul[0].header
        cam, ccd = hdr['CAMERA'], hdr['CCD']
        hdul.close()

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

        # fitresults, samples: each object's sector light curve gets a separate
        # fit. fitresults  contains csvs and jpgs to assess.
        outdirs = [
            os.path.join(resultsdir,'fitresults',lcname.replace('.fits','')),
            os.path.join(resultsdir,'samples',lcname.replace('.fits',''))
        ]
        for outdir in outdirs:
            if not os.path.exists(outdir):
                os.mkdir(outdir)

        supprow = mvp._get_supprow(sourceid, supplementstatsdf)
        suppfulldf = supplementstatsdf

        pfrow = pfdf.loc[pfdf['source_id']==sourceid]
        if len(pfrow) != 1:
            errmsg = 'expected exactly 1 source match in period find df'
            raise AssertionError(errmsg)

        outpath = os.path.join(resultsdir,'fitresults',
                               lcname.replace('.fits',''),
                               lcname.replace('.fits','_fitparameters.csv'))

        if not os.path.exists(outpath):

            _fit_given_cdips_lcpath(tfa_sr_path, lcpath, outpath, mdf,
                                    sourceid, supprow, suppfulldf, pfdf, pfrow,
                                    toidf, ctoidf, sector, nworkers,
                                    cdipsvnum=cdipsvnum, overwrite=overwrite)


def _get_data(sector, cdips_cat_vnum=0.3):

    classifxn_csv = os.path.join(CLASSIFXNDIR,
                                 "sector-{}_UNANIMOUS_GOLD.csv".format(sector))
    df = pd.read_csv(classifxn_csv, sep=';')

    cdips_df = ccl.get_cdips_catalog(ver=cdips_cat_vnum)

    pfpath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
              'cdips/results/cdips_lc_periodfinding/'
              'sector-{}/'.format(sector)+
              'initial_period_finding_results_supplemented.csv')
    pfdf = pd.read_csv(pfpath)

    supppath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
                'cdips_lc_stats/sector-{}/'.format(sector)+
                'supplemented_cdips_lc_statistics.txt')
    supplementstatsdf = pd.read_csv(supppath, sep=';')

    toipath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
              'cdips/data/toi-plus-2019-06-25.csv')
    toidf = pd.read_csv(toipath, sep=',')

    ctoipath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                'cdips/data/ctoi-exofop-2019-06-25.txt')
    ctoidf = pd.read_csv(ctoipath, sep='|')

    return df, cdips_df, pfdf, supplementstatsdf, toidf, ctoidf

def _define_and_make_directories(sector):
    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'fit_gold/'
        'sector-{}'.format(sector)
    )
    dirs = [resultsdir,
            os.path.join(resultsdir,'fitresults'),
            os.path.join(resultsdir,'samples')
           ]
    for _d in dirs:
        if not os.path.exists(_d):
            os.mkdir(_d)

    tfasrdir = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                'CDIPS_LCS/sector-{}_TFA_SR'.format(sector))
    lcbasedir =  ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
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

def _fit_given_cdips_lcpath(tfa_sr_path, lcpath, outpath, mdf, sourceid,
                            supprow, suppfulldf, pfdf, pfrow, toidf, ctoidf,
                            sector, nworkers, cdipsvnum=1, overwrite=1):
    #
    # read and re-detrend lc if needed
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

    time, flux, err = sigclip_magseries(time, flux, err, magsarefluxes=True,
                                        sigclip=[50,5])

    if is_pspline_dtr:
        flux, _ = dtr.detrend_flux(time, flux)

    #
    # define all the paths
    #
    tlsfit_savfile = outpath.replace(
        '_fitparameters.csv',
        '_tlsfit.png'
    )
    fit_savdir = os.path.dirname(outpath)
    chain_savdir = os.path.dirname(outpath).replace('fitresults','samples')

    #
    # acquire estimates of teff, rstar, logg from TIC
    #
    ra, dec = hdr['RA_OBJ'], hdr['DEC_OBJ']
    targetcoord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    radius = 10.0*u.arcsec
    try:
        stars = Catalogs.query_region(
            "{} {}".format(float(targetcoord.ra.value), float(targetcoord.dec.value)),
            catalog="TIC",
            radius=radius
        )
    except requests.exceptions.ConnectionError:
        print('ERR! TIC query failed. trying again...')
        time.sleep(30)
        stars = Catalogs.query_region(
            "{} {}".format(float(targetcoord.ra.value), float(targetcoord.dec.value)),
            catalog="TIC",
            radius=radius
        )

    Tmag_pred = (hdr['phot_g_mean_mag']
                - 0.00522555 * (hdr['phot_bp_mean_mag'] - hdr['phot_rp_mean_mag'])**3
                + 0.0891337 * (hdr['phot_bp_mean_mag'] - hdr['phot_rp_mean_mag'])**2
                - 0.633923 * (hdr['phot_bp_mean_mag'] - hdr['phot_rp_mean_mag'])
                + 0.0324473)
    Tmag_cutoff = 1

    selstars = stars[np.abs(stars['Tmag'] - Tmag_pred)<Tmag_cutoff]

    if len(selstars)>=1:

        seltab = selstars[np.argmin(selstars['dstArcSec'])]

        if not int(seltab['GAIA']) == int(hdr['GAIA-ID']):
            # TODO: should just instead query selstars by Gaia ID u want...
            raise ValueError

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
            rstar_err = 0.3*rstar
            # get mass given rstar, so that you can get logg
            mamadf = pd.read_csv('../data/Mamajek_Rstar_Mstar_Teff_SpT.txt',
                                 delim_whitespace=True)
            if rstar != 'NaN':
                mamarstar, mamamstar = nparr(mamadf['Rsun'])[::-1], nparr(mamadf['Msun'])[::-1]
                isbad = np.insert(np.diff(mamamstar) <= 0, False, 0)
                fn_mass_to_rstar = interp1d(mamamstar[~isbad], mamarstar[~isbad],
                                            kind='quadratic', bounds_error=False,
                                            fill_value='extrapolate')
                radiusval = rstar
                fn = lambda mass: fn_mass_to_rstar(mass) - radiusval
                mass_guess = mamamstar[np.argmin(np.abs(mamarstar - radiusval))]
                try:
                    mass_val = optimize.newton(fn, mass_guess)
                except RuntimeError:
                    mass_val = mass_guess
                mstar = mass_val
            else:
                raise NotImplementedError('need rstar somehow')
            _Mstar = mstar*u.Msun
            _Rstar = rstar*u.Rsun
            logg = np.log10((const.G * _Mstar / _Rstar**2).cgs.value)
            logg_err = 0.3*logg

        ticid = int(hdr['TICID'])

    else:
        raise ValueError('bad xmatch for {}'.format(tfa_sr_path))

    mafr, tlsr = given_light_curve_fit_transit(time, flux, err, teff, teff_err,
                                               rstar, rstar_err, logg,
                                               logg_err, outpath, fit_savdir,
                                               chain_savdir, nworkers=nworkers,
                                               n_transit_durations=5,
                                               n_mcmc_steps=1000,
                                               tlsfit_savfile=tlsfit_savfile,
                                               overwrite=overwrite)

    # convert fit results to ctoi csv format
    fit_results_to_ctoi_csv(ticid, ra, dec, mafr, tlsr, outpath, toidf, ctoidf,
                            teff, teff_err, rstar, rstar_err, logg, logg_err,
                            cdipsvnum=cdipsvnum)




def fit_phased_transit_mandelagol_and_line(
    time, flux, err,
    tlsr,
    teff, teff_err, rstar, rstar_err, logg, logg_err,
    outpath,
    fit_savdir,
    chain_savdir,
    n_transit_durations=5,
    nworkers=40,
    n_mcmc_steps=1,
    overwriteexistingsamples=True,
    mcmcprogressbar=True
):
    """
    tlsr: result dictionary from TLS.
    teff, rstr, logg: usual units, from wherever.
    outpath: .csv file with fit parameters to save.
    fit_savdir: directory where other plots are saved.
    chain_savdir: MCMC chains go here.
    overwriteexistingsamples: if false, and finds pickle file with saved
    parameters, no sampling is done.
    """

    # initial guesses mostly from TLS results.
    incl = 85
    b = 0.2
    period = tlsr['period']
    T_dur = tlsr['duration']
    rp_by_rstar = tlsr['rp_rs']
    a_by_rstar = (
        (period*u.day)/(np.pi*T_dur*u.day) * (1-b**2)**(1/2)
    ).cgs.value
    t0 = tlsr['T0']
    u_linear, u_quad = get_limb_darkening_initial_guesses(teff, logg)

    # isolate each transit to within +/- n_transit_durations
    tmids_obsd = tlsr['transit_times']

    t_Is = tmids_obsd - T_dur/2
    t_IVs = tmids_obsd + T_dur/2

    # focus on the times around transit
    t_starts = t_Is - n_transit_durations*T_dur
    t_ends = t_IVs + n_transit_durations*T_dur

    # fit only +/- n_transit_durations near the transit data. don't try to fit
    # OOT or occultation data.
    sel_inds = np.zeros_like(time).astype(bool)
    for t_start,t_end in zip(t_starts, t_ends):
        these_inds = (time > t_start) & (time < t_end)
        if np.any(these_inds):
            sel_inds |= these_inds

    # to construct the phase-folded light curve, fit a line to the OOT flux
    # data, and use the parameters of the best-fitting line to "rectify" each
    # lightcurve. Note that an order 1 legendre polynomial == a line, so we'll
    # use that implementation.
    out_fluxs, in_fluxs, fit_fluxs, time_list, intra_inds_list, err_list = (
        [], [], [], [], [], []
    )
    for t_start,t_end in zip(t_starts, t_ends):
        this_window_inds = (time > t_start) & (time < t_end)
        tmid = t_start + (t_end-t_start)/2
        # flag out slightly more than expected "in transit" points
        prefactor = 1.05
        transit_start = tmid - prefactor*T_dur/2
        transit_end = tmid + prefactor*T_dur/2

        this_window_intra = (
            (time[this_window_inds] > transit_start) &
            (time[this_window_inds] < transit_end)
        )
        this_window_oot = ~this_window_intra

        this_oot_time = time[this_window_inds][this_window_oot]
        this_oot_flux = flux[this_window_inds][this_window_oot]

        if len(this_oot_flux) == len(this_oot_time) == 0:
            continue
        try:
            p = Legendre.fit(this_oot_time, this_oot_flux, 1)
            coeffs = p.coef
            this_window_fit_flux = p(time[this_window_inds])

            time_list.append( time[this_window_inds] )
            out_fluxs.append( flux[this_window_inds] / this_window_fit_flux )
            fit_fluxs.append( this_window_fit_flux )
            in_fluxs.append( flux[this_window_inds] )
            intra_inds_list.append( (time[this_window_inds]>transit_start) &
                                    (time[this_window_inds]<transit_end) )
            err_list.append( err[this_window_inds] )
        except np.linalg.LinAlgError:
            print('WRN! Legendre.fit failed, b/c bad data for this transit. '
                  'Continue.')
            continue

    # make plots to verify that this procedure is working.
    ix = 0
    for _time, _flux, _fit_flux, _out_flux, _intra in zip(
        time_list, in_fluxs, fit_fluxs, out_fluxs, intra_inds_list):

        outdir = fit_savdir
        savpath = (
            os.path.splitext(outpath)[0]+'_t{:s}.png'.format(str(ix).zfill(3))
        )

        if os.path.exists(savpath):
            print('found & skipped making {}'.format(savpath))
            ix += 1
            continue

        plt.close('all')
        fig, (a0,a1) = plt.subplots(nrows=2, sharex=True, figsize=(6,6))

        a0.scatter(_time, _flux, c='k', alpha=0.9, label='data', zorder=1,
                   s=10, rasterized=True, linewidths=0)
        a0.scatter(_time[_intra], _flux[_intra], c='r', alpha=1,
                   label='in-transit (for fit)', zorder=2, s=10, rasterized=True,
                   linewidths=0)

        a0.plot(_time, _fit_flux, c='b', zorder=0, rasterized=True, lw=2,
                alpha=0.4, label='linear fit to OOT')

        a1.scatter(_time, _out_flux, c='k', alpha=0.9, rasterized=True,
                   s=10, linewidths=0)
        a1.plot(_time, _fit_flux/_fit_flux, c='b', zorder=0, rasterized=True,
                lw=2, alpha=0.4, label='linear fit to OOT')

        xlim = a1.get_xlim()

        for a in [a0,a1]:
            a.hlines(1, np.min(_time)-10, np.max(_time)+10, color='gray',
                     zorder=-2, rasterized=True, alpha=0.2, lw=1,
                     label='flux=1')

        a1.set_xlabel('time-t0 [days]')
        a0.set_ylabel('relative flux')
        a1.set_ylabel('residual')
        a0.legend(loc='best', fontsize='x-small')
        for a in [a0, a1]:
            a.get_yaxis().set_tick_params(which='both', direction='in')
            a.get_xaxis().set_tick_params(which='both', direction='in')
            a.set_xlim(xlim)

        fig.tight_layout(h_pad=0, w_pad=0)
        fig.savefig(savpath, dpi=300, bbox_inches='tight')
        print('saved {:s}'.format(savpath))
        ix += 1

    sel_time = np.concatenate(time_list)
    sel_flux = np.concatenate(out_fluxs)
    fit_flux = np.concatenate(fit_fluxs)
    sel_err = np.concatenate(err_list)
    assert len(sel_flux) == len(sel_time) == len(sel_err)

    # model = transit only (no line). "transit" as defined by BATMAN has flux=1
    # out of transit.
    fittype = 'mandelagol'
    initfitparams = {'t0':t0,
                     'period':period,
                     'sma':a_by_rstar,
                     'rp':rp_by_rstar,
                     'incl':incl
                    }
    fixedparams = {'ecc':0.,
                   'omega':90.,
                   'limb_dark':'quadratic',
                   'u':[u_linear,u_quad]}
    priorbounds = {'t0':(t0 - period/10, t0 + period/10),
                   'period':(period-1e-1, period+1e-1),
                   'sma':(a_by_rstar/5, 5*a_by_rstar),
                   'rp':(rp_by_rstar/3, 3*rp_by_rstar),
                   'incl':(65, 90)
                   #'u_linear':(u_linear-0.1, u_linear+0.1), # narrow to remain "physical"
                   #'u_quad':(u_quad-0.1, u_quad+0.1)
                   }
    cornerparams = {'t0':t0,
                    'period':period,
                    'sma':a_by_rstar,
                    'rp':rp_by_rstar,
                    'incl':incl}#,
                    #'u_linear':u_linear,
                    #'u_quad':u_quad }

    ndims = len(initfitparams)

    ##########################################################
    # FIRST: run the fit using the errors given in the data. #
    ##########################################################
    mandelagolfit_plotname = (
        os.path.splitext(os.path.basename(outpath))[0]+
        '_phased_{:s}_fit_{:d}d_dataerrs.png'.format(fittype, ndims)
    )
    corner_plotname = (
        os.path.splitext(os.path.basename(outpath))[0]+
        '_phased_corner_{:s}_fit_{:d}d_dataerrs.png'.format(fittype, ndims)
    )
    sample_plotname = (
        os.path.splitext(os.path.basename(outpath))[0]+
        '_phased_{:s}_fit_samples_{:d}d_dataerrs.h5'.format(fittype, ndims)
    )

    mandelagolfit_savfile = os.path.join(fit_savdir, mandelagolfit_plotname)
    corner_savfile = os.path.join(fit_savdir, corner_plotname)
    if not os.path.exists(chain_savdir):
        try:
            os.mkdir(chain_savdir)
        except:
            raise AssertionError('you need to save chains')
    samplesavpath = os.path.join(chain_savdir, sample_plotname)

    print('beginning {:s}'.format(samplesavpath))

    plt.close('all')

    fitparamdir = fit_savdir
    if not os.path.exists(fitparamdir):
        os.mkdir(fitparamdir)
    maf_savpath = ( os.path.join(
        fitparamdir,
        (os.path.splitext(os.path.basename(outpath))[0]
        +"_phased_{:s}_fit_dataerrs.pickle".format(fittype))
    ) )

    if os.path.exists(maf_savpath) and not overwriteexistingsamples:
        maf_data_errs = pickle.load(open(maf_savpath, 'rb'))

    else:
        maf_data_errs = lcfit.transits.mandelagol_fit_magseries(
                        sel_time, sel_flux, sel_err,
                        initfitparams, priorbounds, fixedparams,
                        trueparams=cornerparams, magsarefluxes=True,
                        sigclip=None, plotfit=mandelagolfit_savfile,
                        plotcorner=corner_savfile,
                        samplesavpath=samplesavpath, nworkers=nworkers,
                        n_mcmc_steps=n_mcmc_steps, exp_time_minutes=30,
                        eps=1e-6, n_walkers=500, skipsampling=False,
                        overwriteexistingsamples=overwriteexistingsamples,
                        mcmcprogressbar=mcmcprogressbar)
        with open(maf_savpath, 'wb') as f:
            pickle.dump(maf_data_errs, f, pickle.HIGHEST_PROTOCOL)
            print('saved {:s}'.format(maf_savpath))

    fitfluxs = maf_data_errs['fitinfo']['fitmags']
    fitepoch = maf_data_errs['fitinfo']['fitepoch']
    fiterrors = maf_data_errs['fitinfo']['finalparamerrs']
    fitepoch_perr = fiterrors['std_perrs']['t0']
    fitepoch_merr = fiterrors['std_merrs']['t0']

    # Winn (2010) eq 14 gives the transit duration
    k = maf_data_errs['fitinfo']['finalparams']['rp']
    t_dur_day = (
        (period*u.day)/np.pi * np.arcsin(
            1/a_by_rstar * np.sqrt(
                (1 + k)**2 - b**2
            ) / np.sin((incl*u.deg))
        )
    ).to(u.day*u.rad).value

    per_point_cadence = 2*u.min
    npoints_in_transit = (
        int(np.floor(((t_dur_day*u.day)/per_point_cadence).cgs.value))
    )

    # use the whole LC's RMS as the "noise"
    snr, _, empirical_errs = get_snr_of_dip(
        sel_time, sel_flux, sel_time, fitfluxs,
        magsarefluxes=True, atol_normalization=1e-2,
        transitdepth=k**2, npoints_in_transit=npoints_in_transit)

    sigma_tc_theory = estimate_achievable_tmid_precision(
        snr, t_ingress_min=0.05*t_dur_day*24*60,
        t_duration_hr=t_dur_day*24)

    print('mean fitepoch err: {:.2e}'.format(
          np.mean([fitepoch_merr, fitepoch_perr])))
    print('mean fitepoch err / theory err = {:.2e}'.format(
          np.mean([fitepoch_merr, fitepoch_perr]) / sigma_tc_theory))
    print('mean error from data lightcurve ='+
          '{:.2e}'.format(np.mean(sel_err))+
          '\nmeasured empirical RMS = {:.2e}'.format(empirical_errs))

    empirical_err = np.ones_like(sel_err)*empirical_errs

    # THEN: rerun the fit using the empirically determined errors
    # (measured from RMS of the transit-model subtracted lightcurve).
    mandelagolfit_plotname = (
        os.path.splitext(os.path.basename(outpath))[0]+
        '_phased_{:s}_fit_{:d}d_empiricalerrs.png'.format(fittype, ndims)
    )
    corner_plotname = (
        os.path.splitext(os.path.basename(outpath))[0]+
        '_phased_corner_{:s}_fit_{:d}d_empiricalerrs.png'.format(fittype, ndims)
    )
    sample_plotname = (
        os.path.splitext(os.path.basename(outpath))[0]+
        '_phased_{:s}_fit_samples_{:d}d_empiricalerrs.h5'.format(fittype, ndims)
    )

    mandelagolfit_savfile = os.path.join(fit_savdir, mandelagolfit_plotname)
    corner_savfile = os.path.join(fit_savdir, corner_plotname)
    samplesavpath = os.path.join(chain_savdir, sample_plotname)

    plt.close('all')

    print('beginning {:s}'.format(samplesavpath))

    maf_savpath = ( os.path.join(
        fitparamdir,
        (os.path.splitext(os.path.basename(outpath))[0]
         +"_phased_{:s}_fit_empiricalerrs.pickle".format(fittype))
    ) )

    if os.path.exists(maf_savpath) and not overwriteexistingsamples:
        maf_empc_errs = pickle.load(open(maf_savpath, 'rb'))

    else:
        maf_empc_errs = lcfit.transits.mandelagol_fit_magseries(
                        sel_time, sel_flux, empirical_err,
                        initfitparams, priorbounds, fixedparams,
                        trueparams=cornerparams, magsarefluxes=True,
                        sigclip=None, plotfit=mandelagolfit_savfile,
                        plotcorner=corner_savfile,
                        samplesavpath=samplesavpath, nworkers=nworkers,
                        n_mcmc_steps=n_mcmc_steps, exp_time_minutes=30,
                        eps=1e-6, n_walkers=500, skipsampling=False,
                        overwriteexistingsamples=overwriteexistingsamples,
                        mcmcprogressbar=mcmcprogressbar)

        with open(maf_savpath, 'wb') as f:
            pickle.dump(maf_empc_errs, f, pickle.HIGHEST_PROTOCOL)
            print('saved {:s}'.format(maf_savpath))

    # fitfluxs, fittimes = _get_interp_fitfluxs(maf_empc_errs, sel_time)
    # now plot the phased lightcurve
    fitfluxs = maf_empc_errs['fitinfo']['fitmags']
    fittimes = maf_empc_errs['magseries']['times']
    fitperiod = maf_empc_errs['fitinfo']['finalparams']['period']
    fitepoch = maf_empc_errs['fitinfo']['finalparams']['t0']

    outfile = ( os.path.join(
        fit_savdir,
        (os.path.splitext(os.path.basename(outpath))[0]
         +"_phased_{:s}_fit_empiricalerrs.png".format(fittype))
    ) )

    plot_phased_magseries(sel_time, sel_flux, fitperiod, magsarefluxes=True,
                          errs=None, normto=False, epoch=fitepoch,
                          outfile=outfile, sigclip=False, phasebin=0.01,
                          phasewrap=True, phasesort=True,
                          plotphaselim=[-.4,.4], plotdpi=400,
                          modelmags=fitfluxs, modeltimes=fittimes,
                          xaxlabel='Time from mid-transit [days]',
                          yaxlabel='Relative flux', xtimenotphase=True)
    print('made {}'.format(outfile))

    # fit parameters are accessed like
    # maf_empc_errs['fitinfo']['finalparams']['sma'],
    return maf_empc_errs


def _get_interp_fitfluxs(mafr, sel_time):

    medianparams = mafr['fitinfo']['finalparams']
    fixedparams = mafr['fitinfo']['fixedparams']

    from astrobase.lcfit.transits import _get_value, _transit_model

    per = _get_value('period', medianparams, fixedparams)
    t0 = _get_value('t0', medianparams, fixedparams)
    rp = _get_value('rp', medianparams, fixedparams)
    sma = _get_value('sma', medianparams, fixedparams)
    incl = _get_value('incl', medianparams, fixedparams)
    ecc = _get_value('ecc', medianparams, fixedparams)
    omega = _get_value('omega', medianparams, fixedparams)
    limb_dark = _get_value('limb_dark', medianparams, fixedparams)
    try:
        u = fixedparams['u']
    except Exception:
        u = [medianparams['u_linear'], medianparams['u_quad']]

    # sample model over 10k points
    times = np.linspace(np.nanmin(sel_time),
                        np.nanmax(sel_time),
                        num=int(1e4))

    fit_params, fit_m = _transit_model(times, t0, per, rp, sma, incl, ecc,
                                       omega, u, limb_dark,
                                       exp_time_minutes=30,
                                       supersample_factor=20)
    fitmags = fit_m.light_curve(fit_params)

    return fitmags, times


def get_limb_darkening_initial_guesses(teff, logg):
    '''
    CITE: Claret 2017, whose coefficients we're parsing
    '''

    metallicity = 0 # solar

    # get the Claret quadratic priors for TESS bandpass
    # the selected table below is good from Teff = 1500 - 12000K, logg = 2.5 to
    # 6. We choose values computed with the "r method", see
    # http://vizier.u-strasbg.fr/viz-bin/VizieR-n?-source=METAnot&catid=36000030&notid=1&-out=text
    if not 2300 < teff < 12000:
        if teff < 15000:
            print('WRN! using 12000K atmosphere LD coeffs even tho teff={}'.
                  format(teff))
        else:
            print('got teff error')
            import IPython; IPython.embed()
    if not 2.5 < logg < 6:
        if teff < 15000:
            # is B star; assume star is like B6V, mamajek table
            _Mstar = 4*u.Msun
            _Rstar = 2.9*u.Rsun
            logg = np.log10((const.G * _Mstar / _Rstar**2).cgs.value)
        else:
            print('got logg error')
            import IPython; IPython.embed()

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/A+A/600/A30')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    t = catalogs[1]
    sel = (t['Type'] == 'r')
    df = t[sel].to_pandas()

    # since we're using these as starting guesses, not even worth
    # interpolating. just use the closest match!
    # each Teff gets 8 logg values. first, find the best teff match.
    foo = df.iloc[(df['Teff']-teff).abs().argsort()[:8]]
    # then, among those best 8, get the best logg match.
    bar = foo.iloc[(foo['logg']-logg).abs().argsort()].iloc[0]

    # TODO: should probably determine these coefficients by INTERPOLATING.
    # (especially in cases when you're FIXING them, rather than letting them
    # float).
    print('WRN! skipping interpolation for Claret coefficients.')
    print('WRN! data logg={:.3f}, teff={:.1f}'.format(logg, teff))
    print('WRN! Claret logg={:.3f}, teff={:.1f}'.
          format(bar['logg'],bar['Teff']))

    u_linear = bar['aLSM']
    u_quad = bar['bLSM']

    return float(u_linear), float(u_quad)



def given_light_curve_fit_transit(time, flux, err, teff, teff_err, rstar,
                                  rstar_err, logg, logg_err, outpath,
                                  fit_savdir, chain_savdir, nworkers=40,
                                  n_transit_durations=5, n_mcmc_steps=1,
                                  tlsfit_savfile=None,
                                  overwrite=1):
    """
    maybe to astrobase?
    """

    # run tls to get initial parameters.
    tlsp = periodbase.tls_parallel_pfind(time, flux, err, magsarefluxes=True,
                                         tls_rstar_min=0.1, tls_rstar_max=10,
                                         tls_mstar_min=0.1, tls_mstar_max=5.0,
                                         tls_oversample=8, tls_mintransits=1,
                                         tls_transit_template='default',
                                         nbestpeaks=5, sigclip=None,
                                         nworkers=nworkers)

    tlsr = tlsp['tlsresult']
    t0, per = tlsr.T0, tlsr.period

    # make plot of TLS fit
    if isinstance(tlsfit_savfile, str):

        make_fit_plot(tlsr['folded_phase'], tlsr['folded_y'], None,
                      tlsr['model_folded_model'], per, t0, t0, tlsfit_savfile,
                      model_over_lc=False, magsarefluxes=True,
                      fitphase=tlsr['model_folded_phase'])
        print('made {}'.format(tlsfit_savfile))

    # fit the phased transit, within N durations of the transit itself, to
    # determine a/Rstar, inclination, quadratric LD terms, period, and epoch.
    # returns mandel-agol fit results dict.
    mafr = fit_phased_transit_mandelagol_and_line(
        time, flux, err,
        tlsr,
        teff, teff_err, rstar, rstar_err, logg, logg_err,
        outpath,
        fit_savdir,
        chain_savdir,
        n_transit_durations=n_transit_durations,
        nworkers=nworkers,
        n_mcmc_steps=n_mcmc_steps,
        overwriteexistingsamples=overwrite,
        mcmcprogressbar=True
    )

    return mafr, tlsr


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
