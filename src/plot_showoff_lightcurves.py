"""
(aspect ~6x3 total)
WASP-18 phase-folded                            phase-folded rotating blanco1 star
a small planet candidate (phase-folded)         phase-folded EB in blanco 1
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, subprocess, itertools, pickle
from datetime import datetime

from numpy import array as nparr

from astropy.io import fits
from glob import glob

from astrobase import periodbase, checkplot, lcmath
from astrobase.periodbase import kbls
from astrobase.varbase import transits
from astrobase.varbase.trends import (
    smooth_magseries_ndimage_medfilt as medfilt,
    smooth_magseries_savgol as savgol )

from astrobase.lcmath import phase_magseries
from astrobase.lcmath import phase_bin_magseries

from astrobase.lcfit.transits import traptransit_fit_magseries

from scipy.interpolate import interp1d

def check_lightcurve(lcfile):

    outdir = os.path.dirname(lcfile)

    hdulist = fits.open(lcfile)
    data = hdulist[1].data

    ycols = ['IRM1','IRM2','IRM3','TFA1','TFA2','TFA3']
    xcol = 'TMID_BJD'

    for ycol in ycols:

        outname = '{}_vs_{}_{}.png'.format(
            ycol, xcol, os.path.basename(lcfile))
        outpath = os.path.join(outdir, outname)
        #if os.path.exists(outpath):
        #    continue

        f,ax=plt.subplots(figsize=(5,2))

        ax.scatter(data[xcol], data[ycol], c='k', alpha=0.5, zorder=1, s=2,
                   rasterized=True, linewidths=0)

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

        ylim = ax.get_ylim()
        ax.set_ylim((max(ylim),min(ylim)))

        f.savefig(outpath, bbox_inches='tight', dpi=300)
        print('made {}'.format(outpath))


def _divmodel_plot(time, flux, flux_to_div, savpath):

    f, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,6))

    axs[0].scatter(time, flux, c='k', s=1.5, rasterized=True, lw=0, zorder=2)
    axs[0].plot(time, flux_to_div, c='C0', zorder=1)

    axs[1].scatter(time, flux/flux_to_div, c='k', s=1.5, rasterized=True, lw=0,
                   zorder=2)
    axs[1].plot(time, flux_to_div/flux_to_div, c='C0', zorder=1)

    axs[0].set_ylabel('flux')
    axs[1].set_ylabel('flux / model')
    axs[1].set_xlabel('time [days]')
    f.tight_layout(h_pad=0, w_pad=0)
    f.savefig(savpath, dpi=400, bbox_inches='tight')

def _get_flux_from_mags(time, mag, err_mag):
    # must be finite
    # and scaled to relative flux units

    mag_0 = 10  # these numbers do not matter; we care about relative flux
    flux_0 = 1000
    flux = flux_0 * 10**(-0.4 * (mag - mag_0))
    # sigma_flux = dg/d(mag) * sigma_mag, for g=f0 * 10**(-0.4*(mag-mag0)).
    err_flux = np.abs(
        -0.4 * np.log(10) * flux_0 * 10**(-0.4*(mag-mag_0)) * err_mag
    )

    fluxmedian = np.nanmedian(flux)
    flux /= fluxmedian
    err_flux /= fluxmedian

    finds = np.isfinite(flux) & np.isfinite(err_flux)
    flux = flux[finds]
    err_flux = err_flux[finds]
    if len(time) > len(flux):
        # hack to match indices on first iteration
        time = time[finds]

    return time, flux, err_flux


def transit_process_raw_lc(lcfile, mingap=0.5, windowsize=48*2, extra_maskfrac=0.03):
    """
    process raw lightcurve (IRM2) to show off transits.  turn it to relative
    flux units. apply a median filter with a windowsize of whatever's given.

        mingap (float): units of days

        windowsize (int): units of n_cadences, to be applied in the median
        filter.

    return:

        time, whitened_flux, err_flux, time_oot, flux_oot, err_flux_oot, flux
    """

    hdulist = fits.open(lcfile)
    data = hdulist[1].data

    time, mag, err_mag = data['TMID_BJD'], data['IRM2'], data['IRE2']

    time, flux, err_flux = _get_flux_from_mags(time, mag, err_mag)

    ##########################################
    outdir = os.path.dirname(lcfile)
    blsfit_savpath = os.path.join(
        outdir, 'BLS_{}.png'.format(os.path.basename(lcfile)))
    trapfit_savpath = os.path.join(
        outdir, 'TRAPEZOIDAL_{}.png'.format(os.path.basename(lcfile)))
    inout_savpath = os.path.join(
        outdir, 'INOUTTRANSIT_{}.png'.format(os.path.basename(lcfile)))

    time_oot, flux_oot, err_flux_oot = (
        transits.given_lc_get_out_of_transit_points(
        time, flux, err_flux, blsfit_savpath=blsfit_savpath,
        trapfit_savpath=trapfit_savpath,
        in_out_transit_savpath=inout_savpath, nworkers=8,
        extra_maskfrac=extra_maskfrac)
    )

    # for detrending, need to split into orbits.  get time groups, and filter
    # each one
    ngroups, groups = lcmath.find_lc_timegroups(time_oot, mingap=mingap)
    assert ngroups == 2

    tg_smooth_oot_flux = []
    for group in groups:
        tg_smooth_oot_flux.append( medfilt(flux_oot[group], windowsize) )

    flux_oot_smooth = np.concatenate(tg_smooth_oot_flux)

    func = interp1d(time_oot, flux_oot_smooth)
    flux_to_div = func(time)

    divsavpath = os.path.join(
        outdir, 'DIVMODEL_{}.png'.format(os.path.basename(lcfile)))
    _divmodel_plot(time, flux, flux_to_div, divsavpath)

    whitened_flux = flux/flux_to_div

    return (
        time, whitened_flux, err_flux, time_oot, flux_oot, err_flux_oot, flux,
        outdir
    )


def stellarvar_process_raw_lc(lcfile, mingap=0.5, windowsize=48*3):
    """
    process raw lightcurve (IRM2) to show off stellar variability.  turn it to
    relative flux units. stitch consistent normalization across orbits. (also,
    median filter with a BIG window).  get the stellingwerf period and t0 as
    well.

        mingap (float): units of days

    return:

        time, whitened_flux, err_flux, flux, outdir, spdm
    """
    if windowsize % 2 == 0:
        windowsize += 1

    hdulist = fits.open(lcfile)
    data = hdulist[1].data

    time, mag, err_mag = data['TMID_BJD'], data['IRM2'], data['IRE2']

    time, flux, err_flux = _get_flux_from_mags(time, mag, err_mag)

    # for detrending, need to split into orbits.  get time groups, and filter
    # each one
    ngroups, groups = lcmath.find_lc_timegroups(time, mingap=mingap)
    assert ngroups == 2

    tg_smooth_flux = []
    for group in groups:
        #tg_smooth_flux.append( medfilt(flux[group], windowsize) )
        tg_smooth_flux.append( savgol(flux[group], windowsize, polyorder=1) )

    flux_smooth = np.concatenate(tg_smooth_flux)

    func = interp1d(time, flux_smooth)
    flux_to_div = func(time)

    outdir = os.path.dirname(lcfile)
    divsavpath = os.path.join(
        outdir, 'DIVMODEL_{}.png'.format(os.path.basename(lcfile)))
    _divmodel_plot(time, flux, flux_to_div, divsavpath)

    whitened_flux = flux/flux_to_div

    spdm = periodbase.stellingwerf_pdm(time,whitened_flux,err_flux,
                                       magsarefluxes=True, nworkers=8)

    return time, whitened_flux, err_flux, flux, outdir, spdm





def wasp18_fine_tuning(lcfile, time, whitened_flux, err_flux, time_oot, flux_oot,
                       err_flux_oot, outdir):
    #WASP-18 specific things

    outlier_times = time_oot[flux_oot<(np.median(flux_oot)-4*np.std(flux_oot))]
    badtime = (
        ((time - 2.4583e6) > 65) & ((time - 2.4583e6) < 67.5)
        |
        np.in1d(time,outlier_times)
    )
    sel = ~badtime
    time = time[sel]
    whitened_flux = whitened_flux[sel]
    err_flux = err_flux[sel]

    blsfit_savpath = os.path.join(
        outdir, 'WHITENEDBLS_{}.png'.format(os.path.basename(lcfile)))
    trapfit_savpath = os.path.join(
        outdir, 'WHITENEDTRAPEZOIDAL_{}.png'.format(os.path.basename(lcfile)))
    inout_savpath = os.path.join(
        outdir, 'WHITENEDINOUTTRANSIT_{}.png'.format(os.path.basename(lcfile)))
    _, _, _ = (
        transits.given_lc_get_out_of_transit_points(
        time, whitened_flux, err_flux, blsfit_savpath=blsfit_savpath,
        trapfit_savpath=trapfit_savpath, in_out_transit_savpath=inout_savpath,
        nworkers=8)
    )

    return time, whitened_flux, err_flux


def toi_fine_tuning(lcfile, time, whitened_flux, err_flux, time_oot, flux_oot,
                    err_flux_oot, outdir):
    #TOI324... specific things

    badtime = (
        ((time - 2.4583e6) > 80)
    )
    sel = ~badtime
    time = time[sel]
    whitened_flux = whitened_flux[sel]
    err_flux = err_flux[sel]

    blsfit_savpath = os.path.join(
        outdir, 'WHITENEDBLS_{}.png'.format(os.path.basename(lcfile)))
    trapfit_savpath = os.path.join(
        outdir, 'WHITENEDTRAPEZOIDAL_{}.png'.format(os.path.basename(lcfile)))
    inout_savpath = os.path.join(
        outdir, 'WHITENEDINOUTTRANSIT_{}.png'.format(os.path.basename(lcfile)))
    _, _, _ = (
        transits.given_lc_get_out_of_transit_points(
        time, whitened_flux, err_flux, blsfit_savpath=blsfit_savpath,
        trapfit_savpath=trapfit_savpath, in_out_transit_savpath=inout_savpath,
        nworkers=8)
    )

    return time, whitened_flux, err_flux


def get_trapd_given_timeseries(lcfile, time, flux, err_flux, magsarefluxes=True,
                               nworkers=8, sigclip=1000):

    # first, run BLS to get an initial epoch and period.
    endp = 1.05*(np.nanmax(time) - np.nanmin(time))/2

    blsdict = kbls.bls_parallel_pfind(time, flux, err_flux,
                                      magsarefluxes=magsarefluxes, startp=0.1,
                                      endp=endp, maxtransitduration=0.3,
                                      nworkers=nworkers, sigclip=sigclip)

    blsd = kbls.bls_stats_singleperiod(time, flux, err_flux,
                                       blsdict['bestperiod'],
                                       magsarefluxes=True, sigclip=sigclip,
                                       perioddeltapercent=5)
    ingduration_guess = blsd['transitduration'] * 0.2  # a guesstimate.
    transitparams = [
        blsd['period'], blsd['epoch'], blsd['transitdepth'],
        blsd['transitduration'], ingduration_guess
    ]

    # fit a trapezoidal transit model; plot the resulting phased LC.
    savdir = os.path.dirname(lcfile)
    trapfit_savpath = os.path.join(savdir, 'TRAPD_'+os.path.basename(lcfile)+'.png')
    trapd = traptransit_fit_magseries(time, flux, err_flux,
                                      transitparams,
                                      magsarefluxes=magsarefluxes,
                                      sigclip=sigclip,
                                      plotfit=trapfit_savpath)
    # best-guesses at epoch, period for phase-folding...

    return trapd


#FIXME implement
def plot_phase(times, fluxs, fit_period, fit_t0, ax, s=4, alpha=0.3):

    phzd = phase_magseries(times, fluxs, fit_period, fit_t0,
                           wrap=True, sort=True)

    phase = phzd['phase']
    phz_flux = phzd['mags']

    #ax.scatter(phase, phz_flux, c='k', alpha=0.2, zorder=1, s=6,
    #           rasterized=True, linewidths=0)
    ax.scatter(phase, phz_flux, c='k', alpha=alpha, zorder=1, s=s,
               rasterized=True, linewidths=0)

    # # NOTE: could optionally bin...
    # binsize = 0.003
    # bin_phzd = phase_bin_magseries(phase, phz_flux, binsize=binsize)
    # bin_phase = bin_phzd['binnedphases']
    # bin_fluxs = bin_phzd['binnedmags']


    # a0.plot(bin_phase*fit_period*24, bin_fluxs, alpha=1, mew=0.5,
    #         zorder=8, label='binned', markerfacecolor='yellow',
    #         markersize=8, marker='.', color='black', lw=0, rasterized=True)



def main(checklcs=0, processlcs=0, makequadplot=0):

    wasp18file = ( '../data/showoff_lightcurves/wasp18/'
                  '4955371367334610048_llc.fits')
    blanco1files = glob('../data/showoff_lightcurves/blanco_1/*.fits')
    toifiles = glob('../data/showoff_lightcurves/toi_324/*.fits')
    toi324file = toifiles[0]

    if checklcs:
        check_lightcurve(wasp18file)
        for blanco1file in blanco1files:
            check_lightcurve(blanco1file)
        for toifile in toifiles:
            check_lightcurve(toifile)

    if processlcs:

        # retrieve the LC for
        # blanco_1/2320822863006024832_llc_spdm_blsp_checkplot.png
        fpath = [f for f in blanco1files if '2320822863006024832' in f][0]
        time, whitened_flux, err_flux, flux, outdir, spdm = (
             stellarvar_process_raw_lc(fpath, windowsize=48*12)
        )
        period = spdm['bestperiod']
        t0 = np.nanmedian(time)

        assert 0

        # retrieve and clean wasp-18
        (time, whitened_flux, err_flux,
         time_oot, flux_oot, err_flux_oot, flux, outdir) = (
             transit_process_raw_lc(wasp18file)
        )
        time, whitened_flux, err_flux = wasp18_fine_tuning(
            wasp18file, time, whitened_flux, err_flux, time_oot, flux_oot,
            err_flux_oot, outdir)
        trapd = get_trapd_given_timeseries(wasp18file, time, whitened_flux,
                                           err_flux)
        period, t0 = (trapd['fitinfo']['finalparams'][0],
                      trapd['fitinfo']['finalparams'][1])

        # ditto toi 324
        (time, whitened_flux, err_flux,
         time_oot, flux_oot, err_flux_oot, flux, outdir) = (
             transit_process_raw_lc(toi324file, extra_maskfrac=0.3, windowsize=48)
         )
        time, whitened_flux, err_flux = toi_fine_tuning(
            toi324file, time, whitened_flux, err_flux, time_oot, flux_oot,
            err_flux_oot, outdir)
        trapd = get_trapd_given_timeseries(toi324file, time, whitened_flux,
                                           err_flux)

    if makequadplot:

        fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(6,3))

        # retrieve and clean wasp-18
        datapklfile = '../data/showoff_lightcurves/wasp18/plot_data.pickle'
        if not os.path.exists(datapklfile):
            (time, whitened_flux, err_flux,
             time_oot, flux_oot, err_flux_oot, flux, outdir) = (
                 transit_process_raw_lc(wasp18file, windowsize=48*2, extra_maskfrac=0.1)
            )
            time, whitened_flux, err_flux = wasp18_fine_tuning(
                wasp18file, time, whitened_flux, err_flux, time_oot, flux_oot,
                err_flux_oot, outdir)
            trapd = get_trapd_given_timeseries(wasp18file, time, whitened_flux,
                                               err_flux)
            period, t0 = (trapd['fitinfo']['finalparams'][0],
                          trapd['fitinfo']['finalparams'][1])
            outd = {'time':time,'flux':whitened_flux,'period':period,'t0':t0}
            with open(datapklfile,'wb') as f:
                pickle.dump(outd, f, pickle.HIGHEST_PROTOCOL)
                print('dumped to {}'.format(datapklfile))
        else:
            d = pickle.load(open(datapklfile,'rb'))
            time = d['time']
            whitened_flux = d['flux']
            period = d['period']
            t0 = d['t0']

        plot_phase(time, whitened_flux, period, t0, axs[0,0])
        axs[0,0].text(0.96,0.06,'P={:.2f}d'.format(period),
                      transform=axs[0,0].transAxes,ha='right',va='bottom')

        # ok now TOI 324
        datapklfile = '../data/showoff_lightcurves/toi_324/plot_data.pickle'
        if not os.path.exists(datapklfile):
            (time, whitened_flux, err_flux,
             time_oot, flux_oot, err_flux_oot, flux, outdir) = (
                 transit_process_raw_lc(toi324file, extra_maskfrac=0.3, windowsize=48)
            )
            time, whitened_flux, err_flux = toi_fine_tuning(
                toi324file, time, whitened_flux, err_flux, time_oot, flux_oot,
                err_flux_oot, outdir)
            trapd = get_trapd_given_timeseries(toi324file, time, whitened_flux,
                                               err_flux)
            period, t0 = (trapd['fitinfo']['finalparams'][0],
                          trapd['fitinfo']['finalparams'][1])
            outd = {'time':time,'flux':whitened_flux,'period':period,'t0':t0}
            with open(datapklfile,'wb') as f:
                pickle.dump(outd, f, pickle.HIGHEST_PROTOCOL)
                print('dumped to {}'.format(datapklfile))
        else:
            d = pickle.load(open(datapklfile,'rb'))
            time = d['time']
            whitened_flux = d['flux']
            period = d['period']
            t0 = d['t0']

        plot_phase(time, whitened_flux, period, t0, axs[1,0],s=5)
        axs[1,0].text(0.96,0.06,'P={:.2f}d'.format(period),
                      transform=axs[1,0].transAxes,ha='right',va='bottom')

        # now the blanco-1 mega-rotator
        datapklfile = '../data/showoff_lightcurves/blanco_1/2320822863006024832_plot_data.pickle'
        if not os.path.exists(datapklfile):
            fpath = [f for f in blanco1files if '2320822863006024832' in f][0]
            time, whitened_flux, err_flux, flux, outdir, spdm = (
                 stellarvar_process_raw_lc(fpath, windowsize=48*12)
            )
            period = spdm['bestperiod']
            t0 = np.nanmedian(time)
            outd = {'time':time,'flux':whitened_flux,'period':period,'t0':t0}
            with open(datapklfile,'wb') as f:
                pickle.dump(outd, f, pickle.HIGHEST_PROTOCOL)
                print('dumped to {}'.format(datapklfile))
        else:
            d = pickle.load(open(datapklfile,'rb'))
            time = d['time']
            whitened_flux = d['flux']
            period = d['period']
            t0 = d['t0']

        plot_phase(time, whitened_flux, period, t0, axs[0,1])
        axs[0,1].text(0.96,0.94,'P={:.2f}d'.format(period),
                      transform=axs[0,1].transAxes,ha='right',va='top')

        # finally, the rapid M dwarf rotator
        datapklfile = '../data/showoff_lightcurves/mdwarf_rotators/4742436853122727040_plot_data.pickle'
        if not os.path.exists(datapklfile):
            fpath = '../data/showoff_lightcurves/mdwarf_rotators/4742436853122727040_llc.fits'
            time, whitened_flux, err_flux, flux, outdir, _ = (
                 stellarvar_process_raw_lc(fpath, windowsize=48*12)
            )
            spdm = periodbase.stellingwerf_pdm(time,whitened_flux,err_flux,
                                               magsarefluxes=True, nworkers=8,
                                               startp=0.1,endp=0.2,stepsize=1e-4,autofreq=False)

            #the largest three flux points are visible outliers
            badfluxs = whitened_flux[np.argpartition(whitened_flux, -3)[-3:]]
            badinds = np.in1d(whitened_flux, badfluxs)
            okinds = ~badinds
            time = time[okinds]
            whitened_flux = whitened_flux[okinds]

            period = spdm['bestperiod']
            t0 = np.nanmedian(time)
            outd = {'time':time,'flux':whitened_flux,'period':period,'t0':t0}
            with open(datapklfile,'wb') as f:
                pickle.dump(outd, f, pickle.HIGHEST_PROTOCOL)
                print('dumped to {}'.format(datapklfile))
        else:
            d = pickle.load(open(datapklfile,'rb'))
            time = d['time']
            whitened_flux = d['flux']
            period = d['period']
            t0 = d['t0']

        plot_phase(time, whitened_flux, period, t0, axs[1,1])
        axs[1,1].text(0.96,0.06,'P={:.2f}d'.format(period),
                      transform=axs[1,1].transAxes,ha='right',va='bottom')


        # TWEAK THINGS
        fig.text(0.5,0, 'Phase', ha='center')
        fig.text(0,0.5, 'Relative flux', va='center', rotation=90)

        fig.tight_layout(h_pad=0.1, w_pad=0.1)

        axs[0,0].set_ylim((0.988,1.001))
        axs[0,0].set_xlim((-0.7,0.7))

        axs[1,0].set_ylim((0.970,1.009))
        axs[1,0].set_xlim((-0.7,0.7))

        axs[0,1].set_ylim((0.982,1.018))
        axs[0,1].set_xlim((-1,1))

        axs[1,1].set_ylim((0.93,1.07))
        axs[1,1].set_xlim((-1,1))

        for ax in axs.flatten():
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.get_yaxis().set_tick_params(which='both', direction='in')
            ax.get_xaxis().set_tick_params(which='both', direction='in')

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize('small')
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize('small')

        fig.tight_layout(h_pad=0.35, w_pad=0.85, pad=0.95)
        outpath = '../results/showoff_lightcurves/showoff_quadplot.png'
        fig.savefig(outpath, bbox_inches='tight', dpi=400)
        print('made {}'.format(outpath))


if __name__=="__main__":

    checklcs = 0
    processlcs = 0
    makequadplot = 1

    main(checklcs=checklcs, processlcs=processlcs, makequadplot=makequadplot)
