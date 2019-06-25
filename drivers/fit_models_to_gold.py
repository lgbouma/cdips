import os, argparse, pickle, h5py, json, shutil
from glob import glob
from parse import search

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=False)

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from numpy import array as nparr
from astropy.io import fits
from astropy import units as u, constants as const

from astrobase.varbase import lcfit
from astrobase import astrotess as at
from astrobase.periodbase import kbls
from astrobase.varbase.trends import smooth_magseries_ndimage_medfilt
from astrobase import lcmath
from astrobase.services.tic import tic_single_object_crossmatch
from astrobase.varbase.transits import get_snr_of_dip
from astrobase.varbase.transits import estimate_achievable_tmid_precision
from astrobase.plotbase import plot_phased_magseries
from astrobase.lcmath import sigclip_magseries

from astrobase.varbase.transits import get_transit_times

from numpy.polynomial.legendre import Legendre

from astropy import units as u
from astropy.coordinates import SkyCoord

from cdips.lcproc import detrend as dtr
from cdips.lcproc import mask_orbit_edges as moe
from cdips.plotting import vetting_pdf as vp

"""
MCMC fit Mandel-Agol transits to gold above -- these parameters are used for
paper and CTOIs.

(pulls from tessorbitaldecay/measure_transit_times_from_lightcurve.py repo)
"""

CLASSIFXNDIR = "/home/lbouma/proj/cdips/results/vetting_classifications"

def main(overwrite=0, sector=6, cdipsvnum=1, nworkers=40, cdipsvnum=1,
         cdips_cat_vnum=0.3):

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

        supprow = _get_supprow(sourceid, supplementstatsdf)
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
                                    toidf, ctoidf, sector, nworkers)


def _get_data(sector, cdips_cat_vnum=0.3):

    classifxn_csv = os.path.join(CLASSIFXNDIR,
                                 "sector-{}_UNANIMOUS_GOLD.csv".format(sector))
    df = pd.read_csv(classifxn_csv, sep=';')

    cdipscatpath = ('/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
                    'OC_MG_FINAL_GaiaRp_lt_16_v{}.csv'.format(cdips_cat_vnum))
    cdips_df = pd.read_csv(cdipscatpath, sep=';')

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
                'cdips/data/ctoi-exofop-2019-06-25.csv')
    ctoidf = pd.read_csv(toipath, sep='|')

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

def  _fit_given_cdips_lcpath(tfa_sr_path, lcpath, outpath, mdf, sourceid,
                             supprow, suppfulldf, pfdf, pfrow, toidf, ctoidf,
                             sector, nworkers):
    # read lc
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

    # finally read for the transit fitting call (?)
    given_light_curve_fit_transit(time, flux, err)


    pass


def given_light_curve_fit_transit(time, flux, err):
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

    t0, per = tlsp['tlsresult'].T0, tlsp['tlsresult'].period
    midtimes = np.array([t0 + ix*per for ix in range(-100,100)])
    obsd_midtimes = midtimes[ (midtimes > np.nanmin(time)) &
                             (midtimes < np.nanmax(time)) ]

    # TODO: continue from here!!
    #FIXME
    #FIXME
    #FIXME

    fitd = kbls.bls_stats_singleperiod(time, flux, err, per,
                                       magsarefluxes=True, sigclip=None,
                                       perioddeltapercent=5)

    bls_period = fitd['period']
    #  plot the BLS model.
    from astrobase.lcfit.utils import make_fit_plot
    make_fit_plot(fitd['phases'], fitd['phasedmags'], None, fitd['blsmodel'],
                  fitd['period'], fitd['epoch'], fitd['epoch'], blsfit_savfile,
                  magsarefluxes=True)

    ingduration_guess = fitd['transitduration']*0.2
    transitparams = [fitd['period'], fitd['epoch'], fitd['transitdepth'],
                     fitd['transitduration'], ingduration_guess
                    ]

    # fit a trapezoidal transit model; plot the resulting phased LC.
    trapfit = lcfit.traptransit_fit_magseries(time, flux, err_flux,
                                              transitparams,
                                              magsarefluxes=True, sigclip=None,
                                              plotfit=trapfit_savfile)

    period = trapfit['fitinfo']['finalparams'][0]
    t0 = trapfit['fitinfo']['fitepoch']
    transitduration_phase = trapfit['fitinfo']['finalparams'][3]
    tdur = period * transitduration_phase

    # isolate each transit to within +/- n_transit_durations
    tmids, t_starts, t_ends = (
        get_transit_times(fitd, time, n_transit_durations, trapd=trapfit)
    )

    rp = np.sqrt(trapfit['fitinfo']['finalparams'][2])

    if read_literature_params:

        litdir = "../data/literature_physicalparams/{:d}/".format(ticid)
        if not os.path.exists(litdir):
            os.mkdir(litdir)
        litpath = os.path.join(litdir, 'params.csv')

        if not os.path.exists(litpath):
            # attempt to get physical parameters of planet -- period, a/Rstar, and
            # inclination -- for the initial guesses.
            from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
            eatab = NasaExoplanetArchive.get_confirmed_planets_table()

            pl_coords = eatab['sky_coord']
            tcoord = SkyCoord(targetcoordstr, frame='icrs', unit=(u.deg, u.deg))

            print('got match w/ separation {}'.format(
                np.min(tcoord.separation(pl_coords).to(u.arcsec))))
            pl_row = eatab[np.argmin(tcoord.separation(pl_coords).to(u.arcsec))]

            # all dimensionful
            period = pl_row['pl_orbper'].value
            incl = pl_row['pl_orbincl'].value
            semimaj_au = pl_row['pl_orbsmax']
            rstar = pl_row['st_rad']
            a_by_rstar = (semimaj_au / rstar).cgs.value

            litdf = pd.DataFrame(
                {'period_day':period,
                 'a_by_rstar':a_by_rstar,
                 'inclination_deg':incl
                }, index=[0]
            )
            # get the fixed physical parameters from the data. period_day,
            # a_by_rstar, and inclination_deg are comma-separated in this file.
            litdf.to_csv(litpath, index=False, header=True, sep=',')
            litdf = pd.read_csv(litpath, sep=',')
        else:
            litdf = pd.read_csv(litpath, sep=',')

        # NOTE: only period is used, for now
        litparams = tuple(map(float,
            [litdf['period_day'],litdf['a_by_rstar'],litdf['inclination_deg']])
        )
    else:
        raise AssertionError

    # fit the phased transit, within N durations of the transit itself, to
    # determine a/Rstar, inclination, quadratric LD terms, period, and epoch.
    fit_abyrstar, fit_incl, fit_ulinear, fit_uquad = (
        fit_phased_transit_mandelagol_and_line(
            sectornum,
            t_starts, t_ends, time, flux, err_flux, lcfile, fitd, trapfit,
            bls_period, litparams, ticid, fit_savdir, chain_savdir, nworkers,
            n_phase_mcmc_steps, overwriteexistingsamples, mcmcprogressbar)
    )


if __name__=="__main__":
    main()
