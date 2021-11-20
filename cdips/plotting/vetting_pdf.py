"""
Contents:

Plotting functions for CDIPS vetting reports:
    two_periodogram_checkplot
    plot_raw_pca_bkgd
    scatter_increasing_ap_size
    transitcheckdetails
    centroid_plots
    plot_group_neighborhood
    plot_neighborhood_only

Helpers:
    _get_full_infodict
    _insert_newlines
    _get_a_given_P_and_Mstar

Deprecated:
    cluster_membership_check
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

from glob import glob
import os, textwrap, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from copy import deepcopy

from astrobase import periodbase, checkplot
from astrobase.checkplot.png import _make_phased_magseries_plot
from astrobase.lcmath import sigclip_magseries, phase_magseries
from astrobase.lcfit.eclipses import gaussianeb_fit_magseries
from astrobase.plotbase import skyview_stamp

from cdips.lcproc import detrend as dtr
from cdips.lcproc import mask_orbit_edges as moe
from cdips.utils.catalogs import get_cdips_pub_catalog_entry

from astropy.io import fits
from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.wcs import WCS
import astropy.visualization as vis
from astropy.io.votable import from_table, writeto, parse

from scipy import optimize
from scipy.interpolate import interp1d
from numpy import array as nparr

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs

import multiprocessing as mp
nworkers = mp.cpu_count()

def _given_mag_get_flux(mag, err_mag=None):
    # wrapper to avoid breaking dependencies
    from cdips.utils.lcutils import _given_mag_get_flux
    return _given_mag_get_flux(mag, err_mag=err_mag)


def _insert_newlines(string, every=64):
    lines = []
    for i in range(0, len(string), every):
        lines.append(string[i:i+every])
    return '\n'.join(lines)


def _get_a_given_P_and_Mstar(period, mstar):
    return (
        const.G * mstar / (4*np.pi*np.pi) * period**2
    )**(1/3.)


def two_periodogram_checkplot(lc, hdr, supprow, pfrow,
                              mask_orbit_edges=True, fluxap=None,
                              nworkers=32):

    time, mag = lc['TMID_BJD'], lc[fluxap]

    #
    # detrend the light curve
    #
    dtr_dict = {'method':'best', 'break_tolerance':0.5, 'window_length':0.5}
    lsp_options = {'period_min':0.1, 'period_max':20}
    search_time, search_flux, dtr_stages_dict = (
        dtr.clean_rotationsignal_tess_singlesector_light_curve(
            time, mag, magisflux=False, dtr_dict=dtr_dict,
            lsp_dict=None, maskorbitedge=mask_orbit_edges, lsp_options=lsp_options,
            verbose=False
        )
    )

    #
    # run the periodograms
    #
    time = deepcopy(search_time)
    flux = deepcopy(search_flux)
    err = np.ones_like(flux)*1e-4

    spdm = periodbase.stellingwerf_pdm(time, flux, err, magsarefluxes=True,
                                       startp=0.1, endp=19, nworkers=nworkers)

    try:
        tlsp = periodbase.tls_parallel_pfind(time, flux, err, magsarefluxes=True,
                                             tls_rstar_min=0.1, tls_rstar_max=5,
                                             tls_mstar_min=0.1, tls_mstar_max=3.0,
                                             tls_oversample=8, tls_mintransits=1,
                                             tls_transit_template='default',
                                             nbestpeaks=5, sigclip=[50.,5.],
                                             nworkers=nworkers)
    except ValueError as e:
        LOGERROR(f'tls_parallel_pfind error:\n{e}\n. Escaping...')
        fig = plt.Figure(figsize=(30,24))
        return fig, np.nan, np.nan

    objectinfo = {}
    keys = ['objectid','ra','decl','pmra','pmdecl','teff','gmag']
    hdrkeys = ['Gaia-ID', 'RA_OBJ', 'DEC_OBJ', 'PM_RA[mas/yr]',
               'PM_Dec[mas/year]', 'teff_val', 'phot_g_mean_mag']
    for k,hk in zip(keys,hdrkeys):
        if hk in hdr:
            objectinfo[k] = hdr[hk]
        else:
            objectinfo[k] = np.nan

    import matplotlib as mpl
    mpl.rcParams['axes.titlesize'] = 'xx-large'
    mpl.rcParams['axes.labelsize'] = 'xx-large'

    fig = checkplot.twolsp_checkplot_png(tlsp, spdm, time, flux, err,
                                         magsarefluxes=True,
                                         objectinfo=objectinfo,
                                         varepoch='min',
                                         sigclip=[50.,5.], plotdpi=100,
                                         phasebin=3e-2, phasems=6.0,
                                         phasebinms=14.0, unphasedms=6.0,
                                         figsize=(30,24), returnfigure=True,
                                         circleoverlay=1.5*21, yticksize=20)

    axs = fig.get_axes()

    for ax in axs:
        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')

    return fig, tlsp, spdm, dtr_stages_dict


def plot_raw_pca_bkgd(time, rawmag, pcamag, bkgdval, ap_index, supprow,
                      pfrow,
                      savpath=None, obsd_midtimes=None, xlabel='BJDTDB',
                      customstr='', returnfig=True, dtrdict=None,
                      figsize=(30,24)):
    """
    Plot 4 row, 1 column plot with rows of:
        * raw flux vs time
        * pca flux vs time.
        * detrended flux vs time.
        * bkgd val vs time.

    args:
        time, rawmag, pcamag, pcamag (np.ndarray)

        ap_index (int): integer, e.g., "2" for aperture #2.

    kwargs:
        obsd_midtimes (np.ndarray): times for which to underplot lines, to show
        the ephemeris.

        customstr: string that goes on top right of plot, under a line that
        quotes the RMS

        dtrdict: via detrend.clean_rotationsignal_tess_singlesector_light_curve
    """

    rawflux = _given_mag_get_flux(rawmag)
    pcaflux = _given_mag_get_flux(pcamag)

    trend_time, trend_flux = dtrdict['trend_time'], dtrdict['trend_flux']
    search_time, search_flux = dtrdict['search_time'], dtrdict['search_flux']

    ##########################################

    plt.close('all')
    nrows = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=figsize)

    axs = axs.flatten()

    apstr = 'AP{:d}'.format(ap_index)
    stagestrs = ( ['RM{:d}'.format(ap_index),
                   'PCA{:d}'.format(ap_index),
                   'DTR{:d}'.format(ap_index),
                   'BKGD{:d}'.format(ap_index)] )

    yvals = [rawflux,pcaflux,search_flux,bkgdval]
    xvals = [time,time,search_time,time]
    nums = list(range(len(yvals)))

    if DEBUG:
        df = pd.DataFrame({
            'time':time,
            'rawflux':rawflux,
            'pcaflux':pcaflux,
            'search_time':search_time,
            'search_flux':search_flux,
            'bkgdval':bkgdval
        })
        source_id = pfrow.source_id.squeeze()
        outpath = 'example_data_{}.csv'.format(source_id)
        df.to_csv(
            outpath, index=False
        )
        LOGINFO('made {}'.format(outpath))

    for ax, yval, xval, txt, num in zip(axs, yvals, xvals, stagestrs, nums):

        ax.scatter(xval, yval, c='black', alpha=0.9, zorder=2, s=50,
                   rasterized=True, linewidths=0)

        if 'PCA' in txt and len(stagestrs)==4:
            ax.scatter(trend_time, trend_flux, c='red', alpha=0.9, zorder=1,
                       s=20, rasterized=True, linewidths=0)

        if num in [0]:
            txt_x, txt_y = 0.99, 0.98
            mag = yval
            stdmmag = np.nanstd(mag)*1e3
            if stdmmag > 0.1:
                stattxt = '$\sigma$ = {:.1f} mmag{}'.format(stdmmag, customstr)
                ndigits = 2
            elif stdmmag > 0.01:
                stattxt = '$\sigma$ = {:.2f} mmag{}'.format(stdmmag, customstr)
                ndigits = 3
            else:
                stattxt = '$\sigma$ = {:.3f} mmag{}'.format(stdmmag, customstr)
                ndigits = 4
            _ = ax.text(txt_x, txt_y, stattxt, horizontalalignment='right',
                    verticalalignment='top', fontsize='xx-large', zorder=3,
                    transform=ax.transAxes)
        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')

        if isinstance(obsd_midtimes, np.ndarray):
            ylim = ax.get_ylim()
            ax.vlines(obsd_midtimes, min(ylim), max(ylim), color='orangered',
                      linestyle='--', zorder=1, lw=2, alpha=0.3)
            ax.set_ylim((min(ylim), max(ylim)))

    if not isinstance(obsd_midtimes, np.ndarray):
        for ax in axs:
            ylim = ax.get_ylim()
            ax.set_ylim((min(ylim), max(ylim)))

    axs[-1].set_xlabel(xlabel, fontsize='xx-large')
    axs[-1].xaxis.get_offset_text().set_fontsize('xx-large')

    # make the y label
    ax_hidden = fig.add_subplot(111, frameon=False)
    ax_hidden.tick_params(labelcolor='none', top=False, bottom=False,
                          left=False, right=False)

    axs[0].set_ylabel(f'raw flux IRM{ap_index:d}', fontsize='xx-large', labelpad=27)
    axs[1].set_ylabel(f'pca flux PCA{ap_index:d}', fontsize='xx-large', labelpad=27)
    axs[2].set_ylabel(f'detrended flux DTR{ap_index:d}', fontsize='xx-large', labelpad=27)
    axs[3].set_ylabel('background [ADU]', fontsize='xx-large', labelpad=27)

    if not savpath:
        savpath = 'temp_{:s}.png'.format(apstr)

    fig.tight_layout(h_pad=0.)
    if returnfig:
        return fig
    else:
        fig.savefig(savpath, dpi=250, bbox_inches='tight')
        LOGINFO(f'made plot: {savpath}')


def scatter_increasing_ap_size(lc, pfrow, infodict=None, obsd_midtimes=None,
                               xlabel='BJDTDB', customstr='', figsize=(30,24),
                               returnfigdict=True, auto_depth_vs_apsize=False):
    """
    Plot 3 row, 1 column plot with rows of:
        * TFASR ap 1 (smallest)
        * TFASR ap 2 (detection)
        * TFASR ap 3 (biggest).
    We use flux instead of mags, for easiest comparison of depths.

    Similar args/kwargs to plot_raw_tfa_bkgd
    """

    plt.close('all')
    nrows = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=figsize)

    axs = axs.flatten()

    stagestrs = ['PCA1','PCA2','PCA3']

    times, fluxs = [], []

    for i in stagestrs:

        pcamag = lc[i]
        time = lc['TMID_BJD']

        dtr_dict = {'method':'best', 'break_tolerance':0.5, 'window_length':0.5}
        lsp_options = {'period_min':0.1, 'period_max':20}
        search_time, search_flux, dtr_stages_dict = (
            dtr.clean_rotationsignal_tess_singlesector_light_curve(
                time, pcamag, magisflux=False, dtr_dict=dtr_dict,
                lsp_dict=None, maskorbitedge=True,
                lsp_options=lsp_options, verbose=False
            )
        )

        times.append(search_time)
        fluxs.append(search_flux)

    nums = list(range(len(fluxs)))

    _plottimes, _plotfluxs = [], []
    for ax, stime, sflux, txt, num in zip(axs, times, fluxs, stagestrs, nums):

        ax.scatter(stime, sflux, c='black', alpha=0.9, zorder=2, s=50,
                   rasterized=True, linewidths=0)
        _plottimes.append(stime)
        _plotfluxs.append(sflux)

        if num in [0]:
            txt_x, txt_y = 0.99, 0.98
            stdppt = np.nanstd(sflux)*1e3
            if stdppt > 0.1:
                stattxt = '$\sigma$ = {:.1f} ppt{}'.format(stdppt, customstr)
                ndigits = 2
            elif stdppt > 0.01:
                stattxt = '$\sigma$ = {:.2f} ppt{}'.format(stdppt, customstr)
                ndigits = 3
            else:
                stattxt = '$\sigma$ = {:.3f} ppt{}'.format(stdppt, customstr)
                ndigits = 4
            _ = ax.text(txt_x, txt_y, stattxt, horizontalalignment='right',
                    verticalalignment='top', fontsize='xx-large', zorder=3,
                    transform=ax.transAxes)

        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')

        if isinstance(infodict, dict):
            # show transit depth in detection aperture (ap2). depth is saved as
            # 1-delta.
            ax.axhline(infodict['depth'], lw=2, alpha=0.3, color='C0')

    depth_vs_apsize_increasing = False
    tcounts, depths, depth_uncs = [], [], []
    if auto_depth_vs_apsize:
        tlsps = []
        for t,f in zip(_plottimes, _plotfluxs):
            try:
                tlsp = periodbase.tls_parallel_pfind(
                    t, f, np.ones_like(f)*1e-4, magsarefluxes=True,
                    tls_rstar_min=0.1, tls_rstar_max=5, tls_mstar_min=0.1,
                    tls_mstar_max=3.0, tls_oversample=8, tls_mintransits=1,
                    tls_transit_template='default', nbestpeaks=5, sigclip=[50.,5.],
                    nworkers=nworkers
                )
                tlsps.append(tlsp)
            except ValueError as e:
                LOGINFO(e)
                return fig, None

        for tlsp in tlsps:
            tcounts.append(
                tlsp['tlsresult']['distinct_transit_count']
            )
            depths.append(
                1-tlsp['tlsresult']['depth']
            )
            depth_uncs.append(
                np.nanmedian(tlsp['tlsresult']['transit_depths_uncertainties'])
            )

        a1_to_a2_increase = depths[0] + 2*depth_uncs[0] < depths[1]
        a2_to_a3_increase = depths[1] + 2*depth_uncs[1] < depths[2]

        # For this test, require at least three transits, an increasing depth,
        # and finite uncertainties in each aperture.
        if (
            np.all(nparr(tcounts) >= 3)
            and a1_to_a2_increase
            and a2_to_a3_increase
            and (depth_uncs[0] > 1e-10)
            and (depth_uncs[1] > 1e-10)
            and (depth_uncs[2] > 1e-10)
        ):
            depth_vs_apsize_increasing = True

    apdict = {
        'depth_vs_apsize_increasing': depth_vs_apsize_increasing,
        'tcounts': tcounts,
        'depths': depths,
        'depth_uncs': depth_uncs
    }

    if not isinstance(obsd_midtimes, np.ndarray):
        for ax in axs:
            ylim = ax.get_ylim()
            ax.set_ylim((min(ylim), max(ylim)))

    ylims = []
    # make ylims same on all subplots to make by-eye check easier.
    for ax in axs:
        ylims.append(ax.get_ylim())
    ylims = nparr(ylims)
    ymin = np.min(ylims[:,0])
    ymax = np.max(ylims[:,1])
    for ax in axs:
        if isinstance(obsd_midtimes, np.ndarray):
            ax.vlines(obsd_midtimes, ymin, ymax, color='orangered',
                      linestyle='--', zorder=1, lw=2, alpha=0.3)
            ax.set_ylim((ymin, ymax))

    axs[-1].set_xlabel(xlabel, fontsize='xx-large')
    axs[-1].xaxis.get_offset_text().set_fontsize('xx-large')

    # make the y label
    ax_hidden = fig.add_subplot(111, frameon=False)
    ax_hidden.tick_params(labelcolor='none', top=False, bottom=False,
                          left=False, right=False)
    ax_hidden.set_title('detrended flux vs time. top: smallest aperture. middle: '+
                        'detection (medium) aperture. bottom: biggest',
                        fontsize='xx-large')

    fig.tight_layout(h_pad=1)
    if returnfigdict:
        return fig, apdict
    else:
        raise NotImplementedError
        fig.savefig(savpath, dpi=250, bbox_inches='tight')
        LOGINFO('%sZ: made plot: %s' % (datetime.utcnow().isoformat(), savpath))


def _get_full_infodict(tlsp, hdr, mdf):

    # odd_even_mismatch: stdevns btwn odd and even transit depths
    d = {
        'period':tlsp['tlsresult']['period'],
        't0':tlsp['tlsresult']['T0'],
        'duration':tlsp['tlsresult']['duration'],
        'depth':tlsp['tlsresult']['depth'],
        'rp_rs':tlsp['tlsresult']['rp_rs'], #Rp/Rstar, different bc LD
        'snr':tlsp['tlsresult']['snr'],
        'snr_pink_per_transit':tlsp['tlsresult']['snr_pink_per_transit'],
        'odd_even_mismatch':tlsp['tlsresult']['odd_even_mismatch'],
        'sde':tlsp['tlsresult']['SDE']
    }

    # dict from CDIPS star catalog with the following keys:
    #  ['cluster', 'dec', 'dist', 'ext_catalog_name', 'parallax',
    #   'phot_bp_mean_mag', 'phot_g_mean_mag', 'phot_rp_mean_mag', 'pmdec',
    #   'pmra', 'ra', 'reference', 'source_id']
    c = mdf.T.to_dict().popitem()[1]

    h = {
        'teff':hdr['TICTEFF'],
        'rstar':hdr['TICRAD'],
        'mstar':hdr['TICMASS'],
        'AstExcNoiseSig':hdr['AstExcNoiseSig'], # N-sigma of astrometric detection
        'TICID':hdr['TICID'],
        'TESSMAG':hdr['TESSMAG'],
        'TICDIST':hdr['TICDIST'],
        'TICCONT':hdr['TICCONT']
    }

    if h['rstar'] != 'nan':
        rp = (d['rp_rs'] *  h['rstar']*u.Rsun).to(u.Rearth).value
        d['rp'] = '{:.2f}'.format(rp)
    else:
        d['rp'] = 'nan'

    if h['teff'] != 'nan':
        h['teff'] = '{:.0f}'.format(h['teff'])
    if h['rstar'] != 'nan':
        h['rstar'] = '{:.2f}'.format(h['rstar'])
    if h['mstar'] != 'nan':
        h['mstar'] = '{:.2f}'.format(h['mstar'])
    if h['TICCONT'] != 'nan':
        h['TICCONT'] = '{:.2f}'.format(h['TICCONT'])

    if h['rstar'] != 'nan' and h['mstar'] != 'nan':
        mstar = float(h['mstar'])
        rstar = float(h['rstar'])*u.Rsun
        a = _get_a_given_P_and_Mstar(d['period']*u.day, mstar*u.Msun)
        tdur_circ = (rstar*(d['period']*u.day)/(np.pi*a)).to(u.hr).value
        h['circduration'] = '{:.1f}'.format(tdur_circ)

    elif h['rstar'] != 'nan' and h['mstar'] == 'nan':

        # if you have only Rstar, use that to estimate stellar mass (assuming
        # it's a dwarf), and therefore the circular transit duration timescale
        mamadf = pd.read_csv('../data/LITERATURE_DATA/Mamajek_Rstar_Mstar_Teff_SpT.txt',
                             delim_whitespace=True)

        #
        # mass monotonically decreases, but radius does not (in Mamajek's
        # table). so do the interpolation as a function that given mass, gives
        # rstar. then invert it by numerically attempting to find the root. if
        # this fails, just take a "by-eye" interpolation value as Mstar.
        #
        mamarstar, mamamstar = nparr(mamadf['Rsun'])[::-1], nparr(mamadf['Msun'])[::-1]

        isbad = np.insert(np.diff(mamamstar) <= 0, False, 0)
        fn_mass_to_rstar = interp1d(mamamstar[~isbad], mamarstar[~isbad],
                                    kind='quadratic', bounds_error=False,
                                    fill_value='extrapolate')

        radiusval = float(h['rstar'])
        fn = lambda mass: fn_mass_to_rstar(mass) - radiusval

        mass_guess = mamamstar[np.argmin(np.abs(mamarstar - radiusval))]
        try:
            mass_val = optimize.newton(fn, mass_guess)
        except RuntimeError:
            mass_val = mass_guess

        mstar = mass_val
        rstar = float(h['rstar'])*u.Rsun

        a = _get_a_given_P_and_Mstar(d['period']*u.day, mstar*u.Msun)
        tdur_circ = (rstar*(d['period']*u.day)/(np.pi*a)).to(u.hr).value

        h['mstar'] = '{:.2f}'.format(mstar)
        h['circduration'] = '{:.1f}'.format(tdur_circ)

    else:
        h['mstar'] = 'nan'
        h['circduration'] = 'nan'

    megad = {**d, **c, **h}

    if megad['cluster'] == np.nan:
        megad['cluster'] = 'N/A'

    return megad


def transitcheckdetails(stime, sflux, tlsp, mdf, hdr, supprow,
                        pfrow, dtrdict,
                        obsd_midtimes=None,
                        figsize=(30,24), returnfig=True, sigclip=[50,3]):

    d = _get_full_infodict(tlsp, hdr, mdf)

    # [period (time), epoch (time), pdepth (mags), pduration (phase),
    # psdepthratio, secondaryphase]
    initebparams = [d['period'], d['t0'], 1-d['depth'],
                    d['duration']/d['period'], 0.2, 0.5 ]
    ebfitd = gaussianeb_fit_magseries(stime, sflux, np.ones_like(sflux)*1e-4,
                                      initebparams, sigclip=None,
                                      plotfit=False, magsarefluxes=True,
                                      verbose=True)
    fitparams = ebfitd['fitinfo']['finalparams']
    fitparamerrs = ebfitd['fitinfo']['finalparamerrs']

    if fitparams is None:
        fitparams = [np.nan]*42
    if fitparamerrs is None:
        fitparamerrs = [np.nan]*42
    psdepthratio = 1/fitparams[4]
    psdepthratioerr = psdepthratio*(fitparamerrs[4]/fitparams[4])

    #
    # if primary/secondary ratio is >~ 6, could be a planet. (Most extreme case
    # is probably KELT-9b, with delta_tra/delta_occ of ~= 10).
    # if primary/secondary ratio is very near 1, could be a planet, phased at 2x the
    # true orbital period.
    #
    # so it's really only in the primary/secondary range of ~1.2-7 that we can
    # be pretty certain it's an EB.
    #
    d['psdepthratio'] = psdepthratio
    d['psdepthratioerr'] = psdepthratioerr


    # TODO: you might save combination of d, ebfitd, and the MLE fit dictionary...


    ##########################################

    plt.close('all')
    fig = plt.figure(figsize=figsize)

    # ax0: flux v time, top left
    # ax1: text on right
    # ax2: primary transit
    # ax3: occultation
    # ax4: odd
    # ax5: even
    ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (1, 1))
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax5 = plt.subplot2grid((3, 3), (2, 1))

    #
    # ax0: flux v time, top left
    #

    ax0.scatter(stime, sflux, c='black', alpha=0.9, zorder=2, s=50,
                rasterized=True, linewidths=0)
    ax0.set_xlabel('BJDTDB', fontsize='xx-large')
    ax0.xaxis.get_offset_text().set_fontsize('xx-large')

    dtr_method_used = dtrdict['dtr_method_used'].replace('best-','').upper()
    ax0.set_ylabel(f'DTR1 flux (PCA & {dtr_method_used})', fontsize='xx-large')

    if isinstance(obsd_midtimes, np.ndarray):
        ylim = ax0.get_ylim()
        ax0.set_ylim((min(ylim), max(ylim)))
        ax0.vlines(obsd_midtimes, min(ylim), max(ylim), color='orangered',
                   linestyle='--', zorder=1, lw=2, alpha=0.3)
        ax0.set_ylim((min(ylim), max(ylim)))

    #
    # ax1: text on right
    #
    if d['psdepthratioerr'] == 0:
        d['psdepthratioerr'] = np.nan
    if d['period'] == 0:
        d['period'] = np.nan

    txt = (
    """
    P = {period:.6f} day
    $t_0$ = {t0:.6f} BJD
    $R_p$ = {rp:s} $R_\oplus$ (TICCONT {ticcont:s} not needed)
    $R_p/R_\star$ = {rp_rs:.3f}
    $T_{{14}}/P$ = {tdur_by_period:.3f}
    $T_{{14}}$ = {duration:.2f} hr
    SNR = {snr:.1f}, SNRpink/tra = {snr_pink_per_transit:.1f}

    $\delta_{{odd}}$ vs $\delta_{{even}}$ = {odd_even_mismatch:.1f} $\sigma$
    $\delta_{{tra}}/\delta_{{occ}}$ = {psdepthratio:.2f} $\pm$ {psdepthratioerr:.2f}

    Star: DR2 {source_id}
    TIC {ticid:s} - ticdist {ticdist:.2f}"
    $R_\star$ = {rstar:s} $R_\odot$, $M_\star$ = {mstar:s} $M_\odot$
    Teff = {teff:s} K
    RA,dec [deg] = {ra:.3f} {dec:.3f}
    G = {phot_g_mean_mag:.1f}, Rp = {phot_rp_mean_mag:.1f}, Bp = {phot_bp_mean_mag:.1f}, T = {tessmag:.1f}
    pmRA = {pmra:.1f}, pmDEC = {pmdec:.1f}
    $\omega$ = {plx_mas:.2f} $\pm$ {plx_mas_err:.2f} mas
    $d_{{\mathrm{{geom}}}}$ = {dist_pc:.0f} pc
    AstExc: {AstExcNoiseSig:.1f} $\sigma$
    $R_\star$+$M_\star$->$T_{{b0}}$: {circduration:s} hr

    Cluster: {cluster:s}
    Reference: {reference:s}
    """
    )

    max_nchar = 30
    if 'reference' in d:
        reference_val = d['reference'][:max_nchar]
    elif 'reference_id' in d:
        reference_val = d['reference_id'][:max_nchar]
    else:
        reference_val = ''
    try:
        max_nchar = 20
        outstr = txt.format(
            ticid=str(d['TICID']),
            ticdist=float(d['TICDIST']),
            ticcont=d['TICCONT'],
            tessmag=d['TESSMAG'],
            period=d['period'],
            t0=d['t0'],
            rp=str(d['rp']),
            rp_rs=d['rp_rs'],
            tdur_by_period=d['duration']/d['period'],
            duration=d['duration']*24,
            circduration=d['circduration'],
            snr=d['snr'],
            snr_pink_per_transit=np.nanmean(d['snr_pink_per_transit']),
            odd_even_mismatch=d['odd_even_mismatch'],
            source_id=hdr['Gaia-ID'],
            rstar=str(d['rstar']),
            mstar=str(d['mstar']),
            teff=str(d['teff']),
            ra=float(d['ra']),
            dec=float(d['dec']),
            psdepthratio=(
                float(d['psdepthratio'])
                if
                float(d['psdepthratio'])/float(d['psdepthratioerr'])>1
                else
                np.nan),
            psdepthratioerr=(
                float(d['psdepthratioerr'])
                if
                float(d['psdepthratio'])/float(d['psdepthratioerr'])>1
                else
                np.nan),
            phot_g_mean_mag=d['phot_g_mean_mag'],
            phot_rp_mean_mag=d['phot_rp_mean_mag'],
            phot_bp_mean_mag=d['phot_bp_mean_mag'],
            pmra=float(d['pmra']),
            pmdec=float(d['pmdec']),
            plx_mas=float(supprow['Parallax[mas][6]']),
            plx_mas_err=float(supprow['Parallax_error[mas][7]']),
            dist_pc=float(hdr['TICGDIST']),
            AstExcNoiseSig=d['AstExcNoiseSig'],
            cluster='N/A' if pd.isnull(d['cluster']) else d['cluster'][:max_nchar],
            reference=reference_val
        )
    except Exception as e:
        outstr = 'ERR! transitcheckdetails: got bug {}'.format(e)
        LOGERROR(outstr)

    txt_x, txt_y = 0.01, 0.99
    #ax1.text(txt_x, txt_y, textwrap.dedent(outstr),
    #         ha='left', va='top', fontsize=40, zorder=2,
    #         transform=ax1.transAxes)
    ax1.set_axis_off()

    #
    # ax2: primary transit
    #
    phasebin = 6e-3
    minbinelems=2
    tdur_by_period=d['duration']/d['period']
    plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
    try:
        _make_phased_magseries_plot(ax2, 0, stime, sflux, np.ones_like(sflux)/1e4,
                                    d['period'], d['t0'], True, True, phasebin,
                                    minbinelems, plotxlim, 'tls',
                                    xliminsetmode=False, magsarefluxes=True,
                                    phasems=6.0, phasebinms=12.0, verbose=True,
                                    lowerleftstr='primary',
                                    lowerleftfontsize='xx-large')

        model_time = tlsp['tlsresult']['model_lightcurve_time']
        model_y = tlsp['tlsresult']['model_lightcurve_model']

        phasedlc = phase_magseries(model_time, model_y, d['period'], d['t0'],
                                   wrap=True, sort=True)
        plotphase = phasedlc['phase']
        plotmags = phasedlc['mags']
        ax2.plot(
            plotphase, plotmags, zorder=0, color='gray'
        )

    except (TypeError, ValueError) as e:
        LOGINFO('ERR! phased magseries ax2 failed {}'.format(repr(e)))
        pass

    #
    # ax3: occultation
    #
    plotxlim=(-2.0*tdur_by_period+0.5,2.0*tdur_by_period+0.5)
    try:
        _make_phased_magseries_plot(ax3, 0, stime, sflux, np.ones_like(sflux)/1e4,
                                    d['period'], d['t0'], True, True, phasebin,
                                    minbinelems, plotxlim, 'tls',
                                    xliminsetmode=False, magsarefluxes=True,
                                    phasems=6.0, phasebinms=12.0, verbose=True,
                                    lowerleftstr='secondary',
                                    lowerleftfontsize='xx-large')
        ax3.plot(
            plotphase, plotmags, zorder=0, color='gray'
        )
    except (TypeError, ValueError) as e:
        LOGINFO('ERR! phased magseries ax3 failed {}'.format(repr(e)))
        pass

    #
    # ax4: odd
    #
    if not isinstance(obsd_midtimes, np.ndarray):
        raise NotImplementedError

    even_midtimes = obsd_midtimes[::2]
    odd_midtimes = obsd_midtimes[1::2]

    delta_t = 0.245*d['period']
    even_windows = np.array((even_midtimes - delta_t, even_midtimes+delta_t))

    even_mask = np.zeros_like(stime).astype(bool)
    for even_window in even_windows.T:
        even_mask |= np.array(
            (stime > np.min(even_window)) & (stime < np.max(even_window))
        )
    odd_mask = ~even_mask

    plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
    try:
        _make_phased_magseries_plot(ax4, 0, stime[odd_mask], sflux[odd_mask],
                                    np.ones_like(sflux[odd_mask])/1e4,
                                    d['period'], d['t0'], True, True, phasebin,
                                    minbinelems, plotxlim, 'tls',
                                    xliminsetmode=False, magsarefluxes=True,
                                    phasems=6.0, phasebinms=12.0, verbose=True,
                                    lowerleftstr='odd',
                                    lowerleftfontsize='xx-large')
        ax4.plot(
            plotphase, plotmags, zorder=0, color='gray'
        )
    except (TypeError, ValueError) as e:
        LOGINFO('ERR! phased magseries ax4 failed {}'.format(repr(e)))
        pass

    #
    # ax5: even
    #
    plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
    try:
        _make_phased_magseries_plot(ax5, 0, stime[even_mask], sflux[even_mask],
                                    np.ones_like(sflux[even_mask])/1e4,
                                    d['period'], d['t0'], True, True, phasebin,
                                    minbinelems, plotxlim, 'tls',
                                    xliminsetmode=False, magsarefluxes=True,
                                    phasems=6.0, phasebinms=12.0, verbose=True,
                                    lowerleftstr='even',
                                    lowerleftfontsize='xx-large')
        ax5.plot(
            plotphase, plotmags, zorder=0, color='gray'
        )
    except (TypeError, ValueError) as e:
        LOGINFO('ERR! phased magseries ax5 failed {}'.format(repr(e)))
        pass

    outstr = textwrap.dedent(outstr).lstrip('\n')
    ax1.text(0, 0.5, outstr,
             ha='left', va='center', fontsize=32, zorder=2,
             transform=ax1.transAxes)

    for ax in [ax0,ax2,ax3,ax4,ax5]:
        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
    for ax in [ax2,ax3,ax4,ax5]:
        ax.set_title('')
        ax.set_xlabel('phase', fontsize='xx-large')
        ax.set_ylabel('flux', fontsize='xx-large')

    ylim = ax2.get_ylim()
    for ax in [ax2,ax3,ax4,ax5]:
        # consistent ylims across each subplot
        ax.set_ylim(ylim)

    fig.tight_layout(h_pad=0, w_pad=0.5)

    if returnfig:
        return fig, d


def cluster_membership_check(hdr, supprow, infodict, suppfulldf, mdf,
                             k13_notes_df, figsize=(30,20)):

    raise DeprecationWarning('This plot is deprecated.')
    raise ValueError

    getfile = '../data/cluster_data/Kharchenko_2013_MWSC.vot'
    vot = parse(getfile)
    tab = vot.get_first_table().to_table()
    k13 = tab.to_pandas()
    del tab
    k13['Name'] = k13['Name'].str.decode('utf-8')
    k13['Type'] = k13['Type'].str.decode('utf-8')
    k13['MWSC'] = k13['MWSC'].str.decode('utf-8')

    #
    # note: assumption is that
    # /cdips/catalogbuild/construct_unique_cluster_name_column.py has been run.
    # collect all relevant name-parsing info.
    #
    have_name_match = False
    cluster = str(supprow['cluster'].iloc[0])
    clustersplt = cluster.split(',')

    if not pd.isnull(mdf['k13_name_match'].iloc[0]):
        have_name_match = True
        name_match = mdf['k13_name_match'].iloc[0]

    is_known_asterism = False
    if not pd.isnull(mdf['is_known_asterism'].iloc[0]):
        is_known_asterism = bool(mdf['is_known_asterism'].iloc[0])

    # In [3]: np.unique(cddf['why_not_in_k13'][~pd.isnull(cddf['why_not_in_k13'])])
    # Out[3]:
    # array(['K13index: duplicated/coincides with other cluster',
    #        'K13index: possibly this is a cluster, but parameters are not determined',
    #        'K13index: this is not a cluster',
    #        'Majaess IR cluster match missing in K13', 'is_bell_mg',
    #        'is_gagne_mg', 'is_gaia_member', 'is_kraus_mg', 'is_oh_mg',
    #        'is_rizzuto_mg', 'known missing from K13'], dtype=object)
    why_not_in_k13 = ''
    if not pd.isnull(mdf['why_not_in_k13'].iloc[0]):
        why_not_in_k13 = str(mdf['why_not_in_k13'].iloc[0])
        if 'K13index' in why_not_in_k13:
            why_not_in_k13 = 'K13index extra info...'

    if 'reference' in supprow:
        if ('Zari_2018_UMS' in supprow['reference'].iloc[0]
            or 'Zari_2018_PMS' in supprow['reference'].iloc[0]
        ):

            why_not_in_k13 = str(supprow['reference'].iloc[0])
    elif 'reference_id' in supprow:
        if ('Zari_2018_UMS' in supprow['reference_id'].iloc[0]
            or 'Zari_2018_PMS' in supprow['reference_id'].iloc[0]
        ):

            why_not_in_k13 = str(supprow['reference_id'].iloc[0])

    have_cluster_parameters = False
    if have_name_match:
        _k13 = k13.loc[k13['Name'] == name_match]
        have_cluster_parameters = True
        if len(_k13) == 0:
            have_cluster_parameters = False
    elif is_known_asterism or ~pd.isnull(why_not_in_k13):
        pass
    else:
        errmsg = 'ERR! didnt get name match for {}'.format(hdr['Gaia-ID'])
        raise AssertionError(errmsg)

    if have_name_match and 'K13' in why_not_in_k13:
        # no parameters, so by definition here no true name match
        have_cluster_parameters = False

    ##########################################

    plt.close('all')
    fig = plt.figure(figsize=figsize)

    # ax0: distance pdf (or mebe just plx pdf?)
    # ax1: proper motion
    # ax2: big text
    # ax3: HR diagram
    # ax4: spatial positions
    ax0 = plt.subplot2grid((2, 3), (0, 0))
    ax1 = plt.subplot2grid((2, 3), (0, 1))
    ax2 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    ax4 = plt.subplot2grid((2, 3), (1, 1))

    #
    # ax0: parallax probabilities. (boxplot).
    #
    if have_name_match:

        if have_cluster_parameters:
            k13_plx_mas = (1/float(_k13['d'].iloc[0]))*1e3  # "truth"
        else:
            k13_plx_mas = np.nan

        dr2_plx = supprow['Parallax[mas][6]'].iloc[0]
        dr2_plx_err = supprow['Parallax_error[mas][7]'].iloc[0]


        # ax0.errorbar(0, k13_plx_mas, yerr=0,
        #              xerr=0, fmt='o', ecolor='C0', capthick=2,
        #              color='C0', label='K13 cluster', zorder=3)


        dr2_samples = np.random.normal(loc=dr2_plx, scale=dr2_plx_err, size=300)

        ax0.scatter(
            0.333*np.ones_like(dr2_samples) +
                    np.random.normal(0,0.03,len(dr2_samples)),
            dr2_samples,
            c='C1', label='DR2 target star', zorder=3
        )
        # ax0.errorbar(0.5, dr2_plx, yerr=dr2_plx_err,
        #              xerr=0, fmt='o', ecolor='C1', capthick=2,
        #              color='C1', label='DR2 target star', zorder=3)
        # ax0.boxplot(dr2_samples, showfliers=False, whis=[5, 95], zorder=3)

        ax0.axhline(k13_plx_mas, lw=3, alpha=0.3, color='C0', zorder=2,
                    label='K13 {}'.format(name_match))

        cluster = str(supprow['cluster'].iloc[0])
        if have_name_match:
            cluster_df = suppfulldf.loc[suppfulldf['cluster'] == name_match]

        ax0.scatter(
            0.666*np.ones_like(cluster_df['Parallax[mas][6]'])
            + np.random.normal(0,0.03,len(cluster_df)),
            cluster_df['Parallax[mas][6]'],
            c='k', alpha=0.9, zorder=2, s=30, rasterized=True, linewidths=0,
            label='K13 {} Gaia xmatches'.format(name_match)
        )

        ax0.legend(loc='best')
        ax0.set_xticklabels('')
        ax0.set_xlabel('[horizontally separated for readability]',
                       fontsize='xx-large')
        ax0.set_ylabel('star parallax [mas]', fontsize='xx-large')

    #
    # ax1: proper motion. just use error bars in 2d.
    #
    if have_name_match:
        if have_cluster_parameters:
            ax1.errorbar(_k13['pmRA'], _k13['pmDE'], yerr=_k13['e_pm'],
                         xerr=_k13['e_pm'], fmt='o', ecolor='C0', capthick=2,
                         color='C0', label='K13 {}'.format(name_match),
                         zorder=3)

        ax1.errorbar(supprow['PM_RA[mas/yr][8]'].iloc[0],
                     supprow['PM_Dec[mas/year][9]'].iloc[0],
                     xerr=supprow['PMRA_error[mas/yr][10]'].iloc[0],
                     yerr=supprow['PMDec_error[mas/yr][11]'].iloc[0], fmt='o',
                     ecolor='C1', capthick=2, color='C1', label='DR2 target star',
                     ms=15, zorder=4)

        cluster = str(supprow['cluster'].iloc[0])
        if have_name_match:
            cluster_df = suppfulldf.loc[suppfulldf['cluster'] == name_match]
        ax1.scatter(
            cluster_df['PM_RA[mas/yr][8]'],
            cluster_df['PM_Dec[mas/year][9]'],
            c='k', alpha=0.9, zorder=2, s=30, rasterized=True, linewidths=0,
            label='K13 {} Gaia xmatches'.format(name_match)
        )

        ax1.legend(loc='best')

        ax1.set_xlabel(r'pmRA, $\mu_{{\alpha}} \cos\delta$ [mas/yr]',
                       fontsize='xx-large')
        ax1.set_ylabel('pmDEC, $\mu_{{\delta}}$ [mas/yr]',
                       fontsize='xx-large')

    #
    # ax2: big text
    #

    if have_name_match and have_cluster_parameters:
        mwscid = str(_k13['MWSC'].iloc[0])
        n1sr2 = float(_k13['N1sr2'])
        k13type = str(_k13['Type'].iloc[0])
        if k13type == '':
            k13type = 'oc'
        k13dist = float(_k13['d'])
    elif is_known_asterism:
        mwscid = 'N/A'
        name_match = cluster
        n1sr2 = np.nan
        k13type = 'KNOWN ASTERISM'
        k13dist = np.nan
        k13_plx_mas = np.nan
    elif ~pd.isnull(why_not_in_k13):
        mwscid = 'N/A'
        name_match = 'N/A'
        n1sr2 = np.nan
        k13type = why_not_in_k13
        k13dist = np.nan
        k13_plx_mas = np.nan

    time_key = 'logt' if 'logt' in list(supprow.columns) else 'k13_logt'
    logt = str(supprow[time_key].iloc[0])
    if time_key == 'logt':
        logt_provenance = str(supprow['logt_provenance'].iloc[0])
    else:
        logt_provenance = 'K13'

    if 'comment' in list(supprow.columns):
        comment = str(supprow['comment'].iloc[0])
    else:
        comment = 'N/A'

    d = infodict

    txt = (
    """
    Cluster: {cluster:s}
    Reference: {reference:s}
    Starname: {ext_catalog_name:s}
    xmatchdist: {xmatchdist:s}
    logt: {logt:s}, prov: {logt_provenance:s}

    K13 match: MWSC {mwscid:s}, {name_match:s}
    N1sr2: {n1sr2:.0f}
    type = {k13type:s}, $d_{{K13}}$ = {k13dist:.0f} pc
    Expect $\omega_{{K13}}$ = {omegak13:.2f} mas
    Got $\omega_{{DR2}}$ = {plx_mas:.2f} $\pm$ {plx_mas_err:.2f} mas

    Star: DR2 {source_id}
    $R_\star$ = {rstar:s} $R_\odot$, $M_\star$ = {mstar:s} $M_\odot$
    Teff = {teff:s} K
    RA = {ra:.3f}, DEC = {dec:.3f}
    G = {phot_g_mean_mag:.1f}, Rp = {phot_rp_mean_mag:.1f}, Bp = {phot_bp_mean_mag:.1f}
    pmRA = {pmra:.1f}, pmDEC = {pmdec:.1f}
    $\omega$ = {plx_mas:.2f} $\pm$ {plx_mas_err:.2f} mas
    d = 1/$\omega_{{as}}$ = {dist_pc:.0f} pc
    """
    )
    if 'reference' in d:
        reference_val = d['reference'][:max_nchar]
    elif 'reference_id' in d:
        reference_val = d['reference_id'][:max_nchar]
    else:
        reference_val = ''
    if 'ext_catalog_name' in d:
        ext_catalog_name_val=d['ext_catalog_name'][:max_nchar]
    else:
        ext_catalog_name_val=''
    try:
        max_nchar = 20
        outstr = txt.format(
            mwscid=mwscid,
            name_match=name_match,
            n1sr2=n1sr2,
            k13type=k13type[:max_nchar],
            k13dist=k13dist,
            omegak13=k13_plx_mas,
            source_id=hdr['Gaia-ID'],
            rstar=str(d['rstar']),
            mstar=str(d['mstar']),
            teff=str(d['teff']),
            ra=float(d['ra']),
            dec=float(d['dec']),
            phot_g_mean_mag=d['phot_g_mean_mag'],
            phot_rp_mean_mag=d['phot_rp_mean_mag'],
            phot_bp_mean_mag=d['phot_bp_mean_mag'],
            pmra=float(d['pmra']),
            pmdec=float(d['pmdec']),
            plx_mas=float(supprow['Parallax[mas][6]']),
            plx_mas_err=float(supprow['Parallax_error[mas][7]']),
            dist_pc=1/(1e-3 * float(hdr['Parallax[mas]'])),
            cluster='N/A' if pd.isnull(d['cluster']) else d['cluster'][:max_nchar],
            reference=reference_val,
            ext_catalog_name=ext_catalog_name_val,
            xmatchdist=','.join(
                ['{:.1e}"'.format(3600*float(l)) for l in str(d['dist']).split(',')]
            )[:max_nchar],
            logt=logt,
            logt_provenance=logt_provenance
        )
    except Exception as e:
        outstr = 'clusterdetails: got bug {}'.format(e)
        LOGINFO(outstr)

    outdict = {
        'k13dist':k13dist,
        'omegak13':k13_plx_mas,
        'plx_mas':float(supprow['Parallax[mas][6]']),
        'plx_mas_err':float(supprow['Parallax_error[mas][7]']),
        'pmra':float(d['pmra']),
        'pmdec':float(d['pmdec']),
        'mwscid':mwscid,
        'n1sr2':n1sr2,
        'k13type':k13type
    }

    if len(comment)>=1:
        comment = _insert_newlines(comment, every=30)
        outstr = textwrap.dedent(outstr) + '\nNote: '+comment
    else:
        outstr = textwrap.dedent(outstr)

    txt_x, txt_y = 0.01, 0.99
    ax2.text(txt_x, txt_y, outstr,
             ha='left', va='top', fontsize=36, zorder=2,
             transform=ax2.transAxes)
    ax2.set_axis_off()

    #
    # ax3: HR diagram
    #
    if have_name_match:

        cluster = str(supprow['cluster'].iloc[0])

        if have_name_match:
            cluster_df = suppfulldf.loc[suppfulldf['cluster'] == name_match]

        dfs = [cluster_df, supprow]
        zorders = [1,2]
        sizes = [30,150]
        colors = ['k','C1']
        labels = ['K13 '+cluster+' members','target star']

        for df,zorder,color,s,l in zip(dfs,zorders,colors,sizes,labels):

            _yval = nparr(
                df['phot_g_mean_mag[20]']
                #+ 5*np.log10(df['Parallax[mas][6]']/1e3) # NOTE: adds noise
                #+ 5
                #- df['a_g_val[38]']   # NOTE: too many nans
            )
            _xval = nparr(
                df['phot_bp_mean_mag[25]']
                - df['phot_rp_mean_mag[30]']
                #- df['e_bp_min_rp_val[41]']
            )

            ax3.scatter(_xval, _yval, c=color, alpha=1., zorder=zorder, s=s,
                        rasterized=True, linewidths=0, label=l)

        ylim = ax3.get_ylim()
        ax3.set_ylim((max(ylim),min(ylim)))
        ax3.legend(loc='best')
        ax3.set_ylabel('G', fontsize='xx-large') # - $A_G$
        #ax3.set_ylabel('G + 5$\log_{{10}}\omega_{{\mathrm{{as}}}}$ + 5', fontsize='large') # - $A_G$
        ax3.set_xlabel('$G_{\mathrm{BP}} - G_{\mathrm{RP}}$', fontsize='xx-large') #  - E(Bp-Rp)

    #
    # ax4: spatial positions
    #
    if have_name_match:

        dfs = [cluster_df, supprow]
        zorders = [1,2]
        sizes = [20,150]
        colors = ['k','C1']
        labels = ['K13 '+cluster+' members','target star']

        for df,zorder,color,s,l in zip(dfs,zorders,colors,sizes, labels):

            _xval = (
                df['RA[deg][2]']
            )
            _yval = (
                df['Dec[deg][3]']
            )

            ax4.scatter(_xval, _yval, c=color, alpha=1., zorder=zorder, s=s,
                        rasterized=True, linewidths=0, label=l)

        ax4.legend(loc='best')
        ax4.set_xlabel(r'RA, $\alpha$ [deg]', fontsize='xx-large')
        ax4.set_ylabel('Dec, $\delta$ [deg]', fontsize='xx-large')

    ##########################################

    for ax in [ax0,ax1,ax3,ax4]:
        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')

    fig.tight_layout(h_pad=0.5)
    return fig, outdict


def centroid_plots(c_obj, cd, hdr, _pfdf, toidf, figsize=(30,20),
                   Tmag_cutoff=16, findercachedir='~/.astrobase/stamp-cache'):
    """
    cd = {
        'm_oot_flux':m_oot_flux, # mean OOT image
        'm_oot_flux_err':m_oot_flux_err, # mean OOT image uncert
        'm_intra_flux':m_intra_flux, # mean in transit image
        'm_intra_flux_err':m_intra_flux_err,  # mean in transit image uncert
        'm_oot_minus_intra_flux':m_oot_minus_intra_flux, # mean OOT - mean intra
        'm_oot_minus_intra_flux_err':m_oot_minus_intra_flux_err,
        'm_oot_minus_intra_snr':m_oot_minus_intra_snr,
        'ctds_intra':ctds_intra, # centroids of all transits
        'ctds_oot_minus_intra':ctds_oot_minus_intra,
        'ctds_oot':ctds_oot, # centroids of all ootransits
        'm_ctd_intra':m_ctd_intra, # centroid of mean intransit image
        'm_ctd_oot':m_ctd_oot,
        'intra_imgs_flux':intra_imgs_flux,
        'oot_imgs_flux':oot_imgs_flux,
        'intra_imgs_flux_err':intra_imgs_flux_err,
        'oot_imgs_flux_err':oot_imgs_flux_err
    }
    """

    #
    # wcs information parsing
    # follow Clara Brasseur's https://github.com/ceb8/tessworkshop_wcs_hack
    #
    radius = 2.0*u.arcminute

    nbhr_stars = Catalogs.query_region(
        "{} {}".format(float(c_obj.ra.value), float(c_obj.dec.value)),
        catalog="TIC",
        radius=radius
    )

    cutout_wcs = cd['cutout_wcs']
    try:
        px,py = cutout_wcs.all_world2pix(
            nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['ra'],
            nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['dec'],
            0
        )
    except Exception as e:
        LOGINFO('ERR! wcs all_world2pix got {}'.format(repr(e)))
        return None, None

    ticids = nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['ID']
    tmags = nbhr_stars[nbhr_stars['Tmag'] < Tmag_cutoff]['Tmag']

    sel = (px > 0) & (px < 9.5) & (py > 0) & (py < 9.5)
    px,py = px[sel], py[sel]
    ticids, tmags = ticids[sel], tmags[sel]

    ra, dec = float(c_obj.ra.value), float(c_obj.dec.value)
    target_x, target_y = cutout_wcs.all_world2pix(ra,dec,0)

    # geometry: there are TWO coordinate axes. (x,y) and (ra,dec). To get their
    # relative orientations, the WCS and ignoring curvature will usually work.
    shiftra_x, shiftra_y = cutout_wcs.all_world2pix(ra+1e-4,dec,0)
    shiftdec_x, shiftdec_y = cutout_wcs.all_world2pix(ra,dec+1e-4,0)

    ##########################################

    import matplotlib as mpl
    mpl.rcParams['axes.titlesize'] = 'xx-large'
    mpl.rcParams['axes.labelsize'] = 'xx-large'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.labelsize'] = 'xx-large'
    mpl.rcParams['ytick.labelsize'] = 'xx-large'

    plt.close('all')
    fig = plt.figure(figsize=figsize)

    # ax0: OOT
    # ax1: intra
    # ax2: DSS
    # ax3: OOT-intra
    # ax4: OOT-intra SNR
    # ax5 (and 9): text

    ax0 = plt.subplot2grid((3, 3), (0, 0), projection=cutout_wcs)
    ax1 = plt.subplot2grid((3, 3), (0, 1), projection=cutout_wcs)
    ax2 = plt.subplot2grid((3, 3), (0, 2), projection=cutout_wcs)
    ax3 = plt.subplot2grid((3, 3), (1, 0), projection=cutout_wcs)
    ax4 = plt.subplot2grid((3, 3), (1, 1), projection=cutout_wcs)
    ax5 = plt.subplot2grid((3, 3), (1, 2), colspan=2)

    ##########################################

    #
    # ax0: OOT
    #
    vmin = np.min([np.min(cd['m_oot_flux']), np.min(cd['m_intra_flux'])])
    vmax = np.max([np.max(cd['m_oot_flux']), np.max(cd['m_intra_flux'])])

    cset0 = ax0.imshow(cd['m_oot_flux'], cmap='YlGnBu_r', origin='lower',
                       zorder=1, vmin=vmin, vmax=vmax)

    ax0.scatter(cd['ctds_oot'][:,0], cd['ctds_oot'][:,1], marker='o',
                linewidths=0, rasterized=True, c='fuchsia', alpha=0.9,
                zorder=3, s=60)

    ax0.scatter(px, py, marker='x', c='r', s=15, rasterized=True, zorder=2,
                linewidths=1)
    ax0.plot(target_x, target_y, mew=0.5, zorder=5, markerfacecolor='yellow',
             markersize=25, marker='*', color='k', lw=0)

    ax0.set_title('OOT (cyan o: centroid for each OOT window)',
                  fontsize='xx-large')
    ax0.set_xlabel(' ')
    ax0.set_ylabel(' ')

    cb0 = fig.colorbar(cset0, ax=ax0, extend='neither', fraction=0.046, pad=0.04)

    #
    # ax1: intra
    #

    cset1 = ax1.imshow(cd['m_intra_flux'], cmap='YlGnBu_r', origin='lower',
                       zorder=1, vmin=vmin, vmax=vmax)

    ax1.scatter(cd['ctds_intra'][:,0], cd['ctds_intra'][:,1], marker='o',
                linewidths=0, rasterized=True, c='fuchsia', alpha=0.9,
                zorder=3, s=60)

    ax1.scatter(px, py, marker='x', c='r', s=15, rasterized=True, zorder=2,
                linewidths=1)
    ax1.plot(target_x, target_y, mew=0.5, zorder=5, markerfacecolor='yellow',
             markersize=25, marker='*', color='k', lw=0)

    ax1.set_title('in transit  (cyan o: centroid for each transit)',
                  fontsize='xx-large')

    ax1.set_xlabel(' ')
    ax1.set_ylabel(' ')

    cb1 = fig.colorbar(cset1, ax=ax1, extend='neither', fraction=0.046, pad=0.04)

    #
    # ax2: positions & mags of nieghbor stars
    # Offsets of the difference image centroids relative the out-of-transit
    # centroids for each sector in which the star was observed. Nearby objects
    # are also marked. The 3 radius of uncertainty is displayed for the mean
    # (over sectors) centroid offset.
    #

    scalefactor=250
    sizes = scalefactor * (1/tmags)

    ax2.scatter(px, py, marker='o', edgecolors='k', s=sizes, rasterized=True,
                zorder=2, linewidths=1, facecolors='none')

    try:
        ax2.scatter(target_x, target_y, marker='o', edgecolors='C0',
                    facecolors='C0', s=sizes[0], rasterized=True, zorder=3,
                    linewidths=1)
    except IndexError:
        pass

    for ix, _px, _py, ticid, tmag in zip(np.arange(len(px)),
                                         px,py,ticids,tmags):
        txtstr = '{:d}'.format(ix)
        if ix==0:
            ax2.text(_px, _py, txtstr, ha='center', va='bottom', fontsize=18,
                     zorder=4, color='C0')
        else:
            ax2.text(_px, _py, txtstr, ha='center', va='bottom', fontsize=18,
                     zorder=4, color='k')

    # white background, for size scale
    whitecmap = ListedColormap(np.zeros((256,4)))

    cset2 = ax2.imshow(np.zeros_like(cd['m_oot_minus_intra_flux']),
                       cmap=whitecmap, origin='lower', zorder=1)

    ax2.set_title('neighborhood info. (starid,Tmag)', fontsize='xx-large')
    cb2 = fig.colorbar(cset2, ax=ax2, extend='neither', fraction=0.046,
                       pad=0.04, drawedges=False)

    cb2.set_alpha(0)
    cb2.set_ticks([])
    cb2.set_ticklabels([])
    cb2.ax.axis('off')
    cb2.ax.xaxis.set_visible(False)
    cb2.ax.yaxis.set_visible(False)
    cb2.outline.set_linewidth(0)

    ax2.set_xlabel(' ')
    ax2.set_ylabel(' ')

    #
    # ax3: oot - intra
    # fit 2d gaussian to the inner 8x8 pixels to find centroid
    #
    cen_x, cen_y = (cd['ctds_oot_minus_intra'][:,0],
                    cd['ctds_oot_minus_intra'][:,1])

    from photutils.centroids import centroid_2dg

    img = cd['m_oot_minus_intra_flux']
    mask = np.ones((10,10))
    mask[1:9,1:9] -= 1
    try:
        x_ctd, y_ctd = centroid_2dg(img, mask=mask.astype(bool))
    except ValueError as e:
        LOGINFO('ERR! {}'.format(repr(e)))
        x_ctd = np.nan
        y_ctd = np.nan

    if x_ctd > 10 or y_ctd > 10:
        # Centroiding failed. Default to target position to avoid poor sorting.
        x_ctd = target_x
        y_ctd = target_y

    cset3 = ax3.imshow(cd['m_oot_minus_intra_flux'], cmap='YlGnBu_r',
                       origin='lower', zorder=1)

    ax3.plot(x_ctd, y_ctd, mew=0.5,
             zorder=3, markerfacecolor='white', markersize=25, marker='*',
             color='k', lw=0)

    ax3.scatter(px, py, marker='x', c='r', s=20, rasterized=True, zorder=4,
                linewidths=1)
    ax3.plot(target_x, target_y, mew=0.5, zorder=5, markerfacecolor='yellow',
             markersize=25, marker='*', color='k', lw=0)

    ax3.set_title('OOT - in. (cyan *: 2d gaussian fit)', fontsize='xx-large')

    catalog_to_gaussian_sep_px = np.sqrt((target_x - x_ctd)**2 +
                                         (target_y - y_ctd)**2)

    cb3 = fig.colorbar(cset3, ax=ax3, extend='neither', fraction=0.046, pad=0.04)
    ax3.set_xlabel(' ')
    ax3.set_ylabel(' ')


    #
    # ax4 : OOT-intra SNR
    #

    cset4 = ax4.imshow(cd['m_oot_minus_intra_snr'], cmap='YlGnBu_r',
                       origin='lower', zorder= 1)
    ax4.set_title('(OOT - in)/noise', fontsize='xx-large')

    ax4.scatter(px, py, marker='x', c='r', s=15, rasterized=True, zorder=2,
                linewidths=1)
    ax4.plot(target_x, target_y, mew=0.5, zorder=5, markerfacecolor='yellow',
             markersize=25, marker='*', color='k', lw=0)

    cb4 = fig.colorbar(cset4, ax=ax4, extend='neither', fraction=0.046, pad=0.04)
    ax4.set_xlabel(' ')
    ax4.set_ylabel(' ')


    #
    # ITNERMEDIATE SINCE TESS IMAGES NOW PLOTTED
    #
    for ax in [ax0,ax1,ax2,ax3,ax4]:
        ax.grid(ls='--', alpha=0.5)
        if shiftra_x - target_x > 0:
            # want RA to increase to the left (almost E)
            ax.invert_xaxis()
        if shiftdec_y - target_y < 0:
            # want DEC to increase up (almost N)
            ax.invert_yaxis()

    #
    # ax5 : text. neighbors from catalog, matching ephemerides, and any TOI
    # matches.
    #

    # calculate eta_ctd: catalog position - centroid of mean difference
    # image.
    _coord = cutout_wcs.all_pix2world(
        cd['ctd_m_oot_minus_m_intra'][0],
        cd['ctd_m_oot_minus_m_intra'][1],
        0
    )
    _coord = SkyCoord(_coord[0], _coord[1], unit=(u.deg), frame='icrs')
    sep = float(_coord.separation(c_obj).to(u.arcsec).value)

    catalog_to_gaussian_sep_arcsec = catalog_to_gaussian_sep_px*21

    # determining the error is tricky! the question is "with what precision
    # have you measured the center of light in the OOT-intra image?" is worked
    # out in some detail by Bryson+2013 https://arxiv.org/pdf/1303.0052.pdf

    txt = (
    """
    DR2 {source_id}
    ctlg - <OOT-intra>: {sep:.1f}"
    ctlg - gauss(OOT-intra): {catalog_to_gaussian_sep_arcsec:.1f}"
    """
    )
    try:
        outstr = txt.format(
            source_id=hdr['Gaia-ID'],
            sep=sep,
            catalog_to_gaussian_sep_arcsec=catalog_to_gaussian_sep_arcsec
        )
    except Exception as e:
        outstr = 'centroid analysis: got bug {}'.format(e)
        LOGINFO(outstr)

    # TEMPLATE: outstr is like "\nSome stuff\nMore stuff\n"
    outstr = textwrap.dedent(outstr)

    # check TOI list for spatial matches
    toicoords = SkyCoord(nparr(toidf['TIC Right Ascension']),
                         nparr(toidf['TIC Declination']),
                         unit=(u.deg), frame='icrs')
    toiseps = toicoords.separation(c_obj).to(u.arcsec).value
    spatial_cutoff = 126 # arcseconds ~= 6 pixels

    if len(toiseps[toiseps < spatial_cutoff]) >= 1:

        seldf = toidf[toiseps < spatial_cutoff]
        selcols = ['TIC', 'Full TOI ID', 'TOI Disposition', 'Public Comment']
        outstr += '\nGot spatial TOI list match!'

        if int(seldf[selcols]['TIC']) == int(ticids[0]):
            outstr += '\nTICIDs match too'
        else:
            outstr += '\nWRN: TICIDs do not match!'

        outstr += '\n{}\n'.format(
            seldf[selcols].to_string(index=False)
        )

    for ix, _px, _py, ticid, tmag in zip(np.arange(len(px)),
                                         px,py,ticids,tmags):
        if ix == 0:
            outstr += (
                '\nTarget star (0) and {} nearest nbhrs\nTICID (Tmag)\n'.
                format( np.min([15, len(px)-1]) ) )
        if ix >= 16:
            continue
        outstr += '{}: {} ({:.1f})\n'.format(ix, ticid, tmag)

    if isinstance(_pfdf, pd.DataFrame):
        outstr += '\n{} ephem matches in CDIPS LCs\n'.format(len(_pfdf))
        outstr += 'DR2, sepn [arcsec], G_Rp\n'
        # GaiaID, sep, G_Rp mag
        iy = 0
        _pfdf = _pfdf.sort_values(by='seps_px')
        for i,s,rp in zip(_pfdf['source_id'], _pfdf['seps_px'],
                          _pfdf['phot_rp_mean_mag']):

            if iy >= 16:
                continue
            outstr += '{}: {:.1f}", {:.1f}\n'.format(i,s,rp)
            iy += 1

    txt_x, txt_y = 0.03, 0.98
    #ax5.text(txt_x, txt_y, outstr.rstrip('\n'), ha='left', va='top',
    #         fontsize=22, zorder=2, transform=ax5.transAxes)
    ax5.set_axis_off()

    #
    # ax6: DSS linear (rotated to TESS WCS)
    #
    ra = c_obj.ra.value
    dec = c_obj.dec.value
    sizepix = 220
    try:
        dss, dss_hdr = skyview_stamp(ra, dec, survey='DSS2 Red',
                                     scaling='Linear', convolvewith=None,
                                     sizepix=sizepix, flip=False,
                                     cachedir='~/.astrobase/stamp-cache',
                                     verbose=True, savewcsheader=True)
    except (OSError, IndexError, TypeError) as e:
        LOGERROR('downloaded FITS appears to be corrupt, retrying...')
        try:
            dss, dss_hdr = skyview_stamp(ra, dec, survey='DSS2 Red',
                                         scaling='Linear', convolvewith=None,
                                         sizepix=sizepix, flip=False,
                                         cachedir='~/.astrobase/stamp-cache',
                                         verbose=True, savewcsheader=True,
                                         forcefetch=True)

        except Exception as e:
            LOGERROR('failed to get DSS stamp ra {} dec {}, error was {}'.
                     format(ra, dec, repr(e)))
            return None, None

    # image 1: TESS mean OOT. (data: cd['m_oot_flux'], wcs: cutout_wcs)
    # image 2: DSS linear. (data: dss, hdr: dss_hdr)
    ax6 = plt.subplot2grid((3, 3), (2, 0), projection=WCS(dss_hdr))
    ax7 = plt.subplot2grid((3, 3), (2, 1), projection=WCS(dss_hdr))

    cset6 = ax6.imshow(dss, origin='lower', cmap=plt.cm.gray_r)

    ax6.grid(ls='--', alpha=0.5)
    ax6.set_title('DSS2 Red linear', fontsize='xx-large')
    cb6 = fig.colorbar(cset6, ax=ax6, extend='neither', fraction=0.046,
                       pad=0.04)
    ax6.set_xlabel(' ')
    ax6.set_ylabel(' ')

    # DSS is ~1 arcsecond per pixel. overplot apertures on axes 6,7
    for ix, radius_px in enumerate([21,21*1.5,21*2.25]):
        circle = plt.Circle((sizepix/2, sizepix/2), radius_px,
                            color='C{}'.format(ix), fill=False, zorder=5+ix)
        ax6.add_artist(circle)

    #
    # ax7: DSS log (rotated to TESS WCS)
    #
    interval = vis.PercentileInterval(99.99)
    vmin,vmax = interval.get_limits(dss)
    norm = vis.ImageNormalize(
        vmin=vmin, vmax=vmax, stretch=vis.LogStretch(1000))

    cset7 = ax7.imshow(dss, origin='lower', cmap=plt.cm.gray_r, norm=norm)

    ax7.grid(ls='--', alpha=0.5)
    ax7.set_title('DSS2 Red logstretch', fontsize='xx-large')
    cb7 = fig.colorbar(cset7, ax=ax7, extend='neither', fraction=0.046,
                       pad=0.04)

    # DSS is ~1 arcsecond per pixel. overplot apertures on axes 6,7
    for ix, radius_px in enumerate([21,21*1.5,21*2.25]):
        circle = plt.Circle((sizepix/2, sizepix/2), radius_px,
                            color='C{}'.format(ix), fill=False, zorder=5+ix)
        ax7.add_artist(circle)

    ax7.set_xlabel(' ')
    ax7.set_ylabel(' ')

    ##########################################
    fig.tight_layout(pad=3.0, h_pad=3.5)

    ax5.text(txt_x, txt_y, outstr, ha='left', va='top',
             fontsize=22, zorder=2, transform=ax5.transAxes)

    return fig, catalog_to_gaussian_sep_arcsec


def plot_group_neighborhood(
        targetname, groupname, referencename, group_df_dr2, target_df, nbhd_df,
        pmdec_min=None,
        pmdec_max=None,
        pmra_min=None,
        pmra_max=None,
        plx_ylim=None,
        return_figure=True,
        figsize=(30,30),
        source_id=None,
        show_rvs=False,
        cdips_cat_vnum=0.6,
    ):
    """
    source_id: optional, but recommended!
    show_rvs: usually not worthwhile
    """

    l = f'{referencename}\n{groupname}'

    nrows, ncols = 3, 3
    if show_rvs:
        nrows, ncols = 4, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    # both with and without overplot:
    # * parallax vs pm dec
    # * parallax vs pm ra
    # * proper motions
    for row_ix in range(2):

        extra_overplot = False if row_ix == 0 else True

        ######################
        # parallax vs pm dec #
        ######################
        ax = axs[row_ix,0]

        ax.scatter(
            nbhd_df['pmdec'], nbhd_df['parallax'], c='gray', alpha=0.9,
            zorder=2, s=15, rasterized=True, linewidths=0, label='nbhd stars'
        )
        if extra_overplot:
            ax.scatter(
                group_df_dr2['pmdec'], group_df_dr2['parallax'], c='k',
                alpha=0.9, zorder=3, s=20, rasterized=True, linewidths=0,
                label=l
            )

        ax.plot(
            target_df['pmdec'],
            target_df['parallax'],
            alpha=1, mew=0.5, zorder=8, label=targetname,
            markerfacecolor='yellow', markersize=22, marker='*', color='black',
            lw=0
        )

        ax.legend(loc='best', fontsize='x-large')
        ax.set_xlabel(r'pmDEC, $\mu_{{\delta}}$ [mas/yr]', fontsize='xx-large')
        ax.set_ylabel('star parallax [mas]', fontsize='xx-large')

        ax.set_xlim([pmdec_min, pmdec_max])

        ######################
        # parallax vs pm ra #
        ######################
        ax = axs[row_ix,1]

        ax.scatter(
            nbhd_df['pmra'], nbhd_df['parallax'], c='gray', alpha=0.9,
            zorder=2, s=15, rasterized=True, linewidths=0, label='nbhd stars'
        )
        if extra_overplot:
            ax.scatter(
                group_df_dr2['pmra'], group_df_dr2['parallax'], c='k',
                alpha=0.9, zorder=3, s=20, rasterized=True, linewidths=0,
                label=l
            )
        ax.plot(
            target_df['pmra'], target_df['parallax'], alpha=1, mew=0.5,
            zorder=8, label=targetname, markerfacecolor='yellow',
            markersize=22, marker='*', color='black', lw=0
        )

        ax.legend(loc='best', fontsize='x-large')
        ax.set_xlabel(r'pmRA, $\mu_{{\alpha}} \cos\delta$ [mas/yr]',
                      fontsize='xx-large')
        ax.set_ylabel('star parallax [mas]', fontsize='xx-large')

        ax.set_xlim([pmra_min, pmra_max])

        ##################
        # proper motions #
        ##################
        ax = axs[row_ix,2]

        ax.scatter(
            nbhd_df['pmra'], nbhd_df['pmdec'], c='gray', alpha=0.9, zorder=2,
            s=15, rasterized=True, linewidths=0, label='nbhd stars'
        )
        if extra_overplot:
            ax.scatter(
                group_df_dr2['pmra'], group_df_dr2['pmdec'], c='k', alpha=0.9,
                zorder=3, s=20, rasterized=True, linewidths=0, label=l
            )
        ax.plot(
            target_df['pmra'], target_df['pmdec'], alpha=1, mew=0.5, zorder=8,
            label=targetname, markerfacecolor='yellow', markersize=22,
            marker='*', color='black', lw=0
        )

        ax.legend(loc='best', fontsize='x-large')

        ax.set_xlabel(r'pmRA, $\mu_{{\alpha}} \cos\delta$ [mas/yr]',
                      fontsize='xx-large')
        ax.set_ylabel(r'pmDEC, $\mu_{{\delta}}$ [mas/yr]',
                      fontsize='xx-large')

        ax.set_xlim([pmra_min, pmra_max])
        ax.set_ylim([pmdec_min, pmdec_max])

    #######
    # CMD #
    #######
    ax = axs[2,0]

    ax.scatter(
        nbhd_df['phot_bp_mean_mag']-nbhd_df['phot_rp_mean_mag'], nbhd_df['phot_g_mean_mag'],
        c='gray', alpha=1., zorder=2, s=15, rasterized=True, linewidths=0,
        label='nbhd stars'
    )
    ax.scatter(
        group_df_dr2['phot_bp_mean_mag']-group_df_dr2['phot_rp_mean_mag'],
        group_df_dr2['phot_g_mean_mag'],
        c='k', alpha=1., zorder=3, s=15, rasterized=True, linewidths=0, label=l
    )
    ax.plot(
        target_df['phot_bp_mean_mag']-target_df['phot_rp_mean_mag'],
        target_df['phot_g_mean_mag'],
        alpha=1, mew=0.5, zorder=8, label=targetname, markerfacecolor='yellow',
        markersize=22, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', fontsize='x-large')
    ax.set_ylabel('G', fontsize='xx-large')
    ax.set_xlabel('$G_{\mathrm{BP}} - G_{\mathrm{RP}}$', fontsize='xx-large')

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))
    xlim = ax.get_xlim()
    ax.set_xlim((min(xlim),max(xlim)))

    ##############
    # HR diagram #
    ##############
    ax = axs[2,1]

    nbhd_yval = np.array([nbhd_df['phot_g_mean_mag'] +
                          5*np.log10(nbhd_df['parallax']/1e3) + 5])
    ax.scatter(
        nbhd_df['phot_bp_mean_mag']-nbhd_df['phot_rp_mean_mag'], nbhd_yval,
        c='gray', alpha=1., zorder=2, s=15, rasterized=True, linewidths=0,
        label='nbhd stars'
    )
    yval = group_df_dr2['phot_g_mean_mag'] + 5*np.log10(group_df_dr2['parallax']/1e3) + 5
    ax.scatter(
        group_df_dr2['phot_bp_mean_mag']-group_df_dr2['phot_rp_mean_mag'],
        yval,
        c='k', alpha=1., zorder=3, s=15, rasterized=True, linewidths=0, label=l
    )
    target_yval = np.array([target_df['phot_g_mean_mag'] +
                            5*np.log10(target_df['parallax']/1e3) + 5])
    ax.plot(
        target_df['phot_bp_mean_mag']-target_df['phot_rp_mean_mag'],
        target_yval,
        alpha=1, mew=0.5, zorder=8, label=targetname, markerfacecolor='yellow',
        markersize=22, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', fontsize='x-large')
    ax.set_ylabel('$G + 5\log_{10}(\omega_{\mathrm{as}}) + 5$', fontsize='xx-large')
    ax.set_xlabel('$G_{\mathrm{BP}} - G_{\mathrm{RP}}$', fontsize='xx-large')

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))
    xlim = ax.get_xlim()
    ax.set_ylim((min(xlim),max(xlim)))

    # set M_omega y limits
    min_y = np.nanmin(np.array([np.nanpercentile(nbhd_yval, 2),
                                float(target_yval)]))
    max_y = np.nanmax(np.array([np.nanpercentile(nbhd_yval, 98),
                                float(target_yval)]))
    edge_y = 0.01*(max_y - min_y)
    momega_ylim = [max_y+edge_y, min_y-edge_y]
    ax.set_ylim(momega_ylim)

    #############
    # positions #
    #############
    ax = axs[2,2]

    ax.scatter(
        nbhd_df['ra'], nbhd_df['dec'], c='gray', alpha=1., zorder=2, s=15,
        rasterized=True, linewidths=0, label='nbhd stars'
    )
    ax.scatter(
        group_df_dr2['ra'], group_df_dr2['dec'], c='k', alpha=1., zorder=3,
        s=15, rasterized=True, linewidths=0, label=l
    )
    ax.plot(
        target_df['ra'], target_df['dec'], alpha=1, mew=0.5, zorder=8,
        label=targetname, markerfacecolor='yellow', markersize=22, marker='*',
        color='black', lw=0
    )

    ax.legend(loc='best', fontsize='x-large')
    ax.set_xlabel(r'RA, $\alpha$ [deg]', fontsize='xx-large')
    ax.set_ylabel('Dec, $\delta$ [deg]', fontsize='xx-large')

    if show_rvs:
        ######################
        # parallax vs DR2 rv #
        ######################
        for col_ix in range(2):

            extra_overplot = False if col_ix == 0 else True

            ax = axs[3,col_ix]

            ax.scatter(
                nbhd_df['radial_velocity'], nbhd_df['parallax'], c='gray',
                alpha=0.9, zorder=2, s=15, rasterized=True, linewidths=0,
                label='nbhd stars'
            )
            ax.plot(
                target_df['radial_velocity'], target_df['parallax'], alpha=1,
                mew=0.5, zorder=8, label=targetname, markerfacecolor='yellow',
                markersize=22, marker='*', color='black', lw=0
            )
            if extra_overplot:
                rvl = (
                    r' ($\langle$RV$\rangle$='+
                    '{:.1f})'.format(np.nanmean(group_df_dr2.radial_velocity))
                )
                ax.scatter(
                    group_df_dr2['radial_velocity'], group_df_dr2['parallax'],
                    c='k', alpha=0.9, zorder=3, s=30, rasterized=True,
                    linewidths=0, label=rvl
                )

            ax.legend(loc='best', fontsize='x-large')
            ax.set_xlabel('DR2 RV (if available) [km/s]', fontsize='xx-large')
            ax.set_ylabel('star parallax [mas]', fontsize='xx-large')

        ########
        # text #
        ########

        ax = axs[3,2]

        if isinstance(source_id, int) or isinstance(source_id, str):
            row = get_cdips_pub_catalog_entry(source_id, ver=cdips_cat_vnum)
            if 'reference' in row:
                references = row['reference'].iloc[0]
            elif 'reference_id' in row:
                references = row['reference_id'].iloc[0]
            else:
                references = 'N/A'
            clusters = row['cluster'].iloc[0]
            if 'ext_catalog_name' in row:
                ext_catalog_names = row['ext_catalog_name'].iloc[0]
            else:
                ext_catalog_names = 'N/A'
        else:
            references = 'N/A'
            clusters = 'N/A'
            ext_catalog_names = 'N/A'

        max_nchar = 30
        outstr = (
            'Target: {targetname}'
            '\nGAIADR2: {source_id}'
            '\nGroup: {groupname}'
            '\nRefs: {references}'
            '\nNames: {clusters}'
        ).format(
            targetname=targetname,
            source_id=source_id,
            groupname=groupname,
            references=references[:max_nchar],
            clusters=clusters[:max_nchar],
            ext_catalog_names=ext_catalog_names[:max_nchar]
        )

        txt_x, txt_y = 0.03, 0.98

        ax.set_axis_off()
        ax.text(txt_x, txt_y, outstr, ha='left', va='top', fontsize=22, zorder=2,
                transform=ax.transAxes)


    ############
    # fix axes #
    ############
    for row_ix in range(nrows):
        for col_ix in range(ncols):
            ax = axs[row_ix, col_ix]
            ax.get_yaxis().set_tick_params(which='both', direction='in',
                                           labelsize='xx-large')
            ax.get_xaxis().set_tick_params(which='both', direction='in',
                                           labelsize='xx-large')

    # set parallax y limits
    min_plx = np.nanmin(np.array([np.nanpercentile(nbhd_df['parallax'], 2),
                                  target_df['parallax'].iloc[0]]))
    max_plx = np.nanmax(np.array([np.nanpercentile(nbhd_df['parallax'], 98),
                                  target_df['parallax'].iloc[0]]))
    edge_plx = 0.01*(max_plx - min_plx)
    plx_ylim = [min_plx-edge_plx, max_plx+edge_plx]

    if show_rvs:
        for ax in [
            axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[3,0], axs[3,1]
        ]:
            ax.set_ylim(plx_ylim)

        # set RV x limits
        min_rv = np.nanmin(np.array([np.nanpercentile(nbhd_df['radial_velocity'], 2),
                                     target_df['parallax'].iloc[0]]))
        max_rv = np.nanmax(np.array([np.nanpercentile(nbhd_df['radial_velocity'], 98),
                                     target_df['parallax'].iloc[0]]))
        edge_rv = 0.01*(max_rv - min_rv)
        rv_ylim = [min_rv - edge_rv, max_rv + edge_rv]
        for ax in [
            axs[3,0], axs[3,1]
        ]:
            ax.set_xlim(rv_ylim)

    else:
        for ax in [
            axs[0,0], axs[0,1], axs[1,0], axs[1,1]
        ]:
            ax.set_ylim(plx_ylim)


    ########
    # save #
    ########
    fig.tight_layout()

    if return_figure:
        return fig

    else:
        outpath = (
            '../results/neighborhoods/{}_{}_neighborhood.png'.
            format(targetname, groupname)
        )

        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        LOGINFO('made {}'.format(outpath))


def plot_neighborhood_only(
        targetname, groupname, target_df, nbhd_df,
        pmdec_min=None, pmdec_max=None,
        pmra_min=None, pmra_max=None,
        source_id=None, figsize=(30,20),
        plx_ylim=None,
        return_figure=True
    ):
    """
    source_id: optional, but recommended!
    """

    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    # both with and without overplot:
    # * parallax vs pm dec
    # * parallax vs pm ra
    # * proper motions
    for row_ix in range(1):

        ######################
        # parallax vs pm dec #
        ######################
        ax = axs[row_ix,0]

        ax.scatter(
            nbhd_df['pmdec'], nbhd_df['parallax'], c='gray', alpha=0.9,
            zorder=2, s=15, rasterized=True, linewidths=0, label='nbhd stars'
        )

        ax.plot(
            target_df['pmdec'],
            target_df['parallax'],
            alpha=1, mew=0.5, zorder=8, label=targetname,
            markerfacecolor='yellow', markersize=22, marker='*', color='black',
            lw=0
        )

        ax.legend(loc='best', fontsize='x-large')
        ax.set_xlabel(r'pmDEC, $\mu_{{\delta}}$ [mas/yr]', fontsize='xx-large')
        ax.set_ylabel('star parallax [mas]', fontsize='xx-large')

        ax.set_xlim([pmdec_min, pmdec_max])

        ######################
        # parallax vs pm ra #
        ######################
        ax = axs[row_ix,1]

        ax.scatter(
            nbhd_df['pmra'], nbhd_df['parallax'], c='gray', alpha=0.9,
            zorder=2, s=15, rasterized=True, linewidths=0, label='nbhd stars'
        )
        ax.plot(
            target_df['pmra'], target_df['parallax'], alpha=1, mew=0.5,
            zorder=8, label=targetname, markerfacecolor='yellow',
            markersize=22, marker='*', color='black', lw=0
        )

        ax.legend(loc='best', fontsize='x-large')
        ax.set_xlabel(r'pmRA, $\mu_{{\alpha}} \cos\delta$ [mas/yr]',
                      fontsize='xx-large')
        ax.set_ylabel('star parallax [mas]', fontsize='xx-large')

        ax.set_xlim([pmra_min, pmra_max])

        ##################
        # proper motions #
        ##################
        ax = axs[row_ix,2]

        ax.scatter(
            nbhd_df['pmra'], nbhd_df['pmdec'], c='gray', alpha=0.9, zorder=2,
            s=15, rasterized=True, linewidths=0, label='nbhd stars'
        )
        ax.plot(
            target_df['pmra'], target_df['pmdec'], alpha=1, mew=0.5, zorder=8,
            label=targetname, markerfacecolor='yellow', markersize=22,
            marker='*', color='black', lw=0
        )

        ax.legend(loc='best', fontsize='x-large')

        ax.set_xlabel(r'pmRA, $\mu_{{\alpha}} \cos\delta$ [mas/yr]',
                      fontsize='xx-large')
        ax.set_ylabel(r'pmDEC, $\mu_{{\delta}}$ [mas/yr]',
                      fontsize='xx-large')

        ax.set_xlim([pmra_min, pmra_max])
        ax.set_ylim([pmdec_min, pmdec_max])

    #######
    # CMD #
    #######
    ax = axs[1,0]

    ax.scatter(
        nbhd_df['phot_bp_mean_mag']-nbhd_df['phot_rp_mean_mag'], nbhd_df['phot_g_mean_mag'],
        c='gray', alpha=1., zorder=2, s=15, rasterized=True, linewidths=0,
        label='nbhd stars'
    )
    ax.plot(
        target_df['phot_bp_mean_mag']-target_df['phot_rp_mean_mag'],
        target_df['phot_g_mean_mag'],
        alpha=1, mew=0.5, zorder=8, label=targetname, markerfacecolor='yellow',
        markersize=22, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', fontsize='x-large')
    ax.set_ylabel('$G$', fontsize='xx-large')
    ax.set_xlabel('$G_{\mathrm{BP}} - G_{\mathrm{RP}}$', fontsize='xx-large')

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))
    xlim = ax.get_xlim()
    ax.set_xlim((min(xlim),max(xlim)))

    ##############
    # HR diagram #
    ##############
    ax = axs[1,1]

    nbhd_yval = np.array([nbhd_df['phot_g_mean_mag'] +
                          5*np.log10(nbhd_df['parallax']/1e3) + 5])
    ax.scatter(
        nbhd_df['phot_bp_mean_mag']-nbhd_df['phot_rp_mean_mag'], nbhd_yval,
        c='gray', alpha=1., zorder=2, s=15, rasterized=True, linewidths=0,
        label='nbhd stars'
    )
    target_yval = np.array([target_df['phot_g_mean_mag'] +
                            5*np.log10(target_df['parallax']/1e3) + 5])
    ax.plot(
        target_df['phot_bp_mean_mag']-target_df['phot_rp_mean_mag'],
        target_yval,
        alpha=1, mew=0.5, zorder=8, label=targetname, markerfacecolor='yellow',
        markersize=22, marker='*', color='black', lw=0
    )

    ax.legend(loc='best', fontsize='x-large')
    ax.set_ylabel('$G + 5\log_{10}(\omega_{\mathrm{as}}) + 5$', fontsize='xx-large')
    ax.set_xlabel('$G_{\mathrm{BP}} - G_{\mathrm{RP}}$', fontsize='xx-large')

    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))

    # set M_omega y limits
    min_y = np.nanmin(np.array([np.nanpercentile(nbhd_yval, 2), target_yval]))
    max_y = np.nanmax(np.array([np.nanpercentile(nbhd_yval, 98), target_yval]))
    edge_y = 0.01*(max_y - min_y)
    momega_ylim = [max_y+edge_y, min_y-edge_y]
    ax.set_ylim(momega_ylim)

    xlim = ax.get_xlim()
    ax.set_xlim((min(xlim),max(xlim)))

    #############
    # positions #
    #############
    ax = axs[1,2]

    ax.scatter(
        nbhd_df['ra'], nbhd_df['dec'], c='gray', alpha=1., zorder=2, s=15,
        rasterized=True, linewidths=0, label='nbhd stars'
    )
    ax.plot(
        target_df['ra'], target_df['dec'], alpha=1, mew=0.5, zorder=8,
        label=targetname, markerfacecolor='yellow', markersize=22, marker='*',
        color='black', lw=0
    )

    ax.legend(loc='best', fontsize='x-large')
    ax.set_xlabel(r'RA, $\alpha$ [deg]', fontsize='xx-large')
    ax.set_ylabel('Dec, $\delta$ [deg]', fontsize='xx-large')

    ############
    # fix axes #
    ############
    for row_ix in range(nrows):
        for col_ix in range(ncols):
            ax = axs[row_ix, col_ix]
            ax.get_yaxis().set_tick_params(which='both', direction='in',
                                           labelsize='xx-large')
            ax.get_xaxis().set_tick_params(which='both', direction='in',
                                           labelsize='xx-large')

    # set parallax y limits
    min_plx = np.nanmin(np.array([np.nanpercentile(nbhd_df['parallax'], 2),
                                  target_df['parallax'].iloc[0]]))
    max_plx = np.nanmax(np.array([np.nanpercentile(nbhd_df['parallax'], 98),
                                  target_df['parallax'].iloc[0]]))
    edge_plx = 0.01*(max_plx - min_plx)
    plx_ylim = [min_plx-edge_plx, max_plx+edge_plx]

    for ax in [
        axs[0,0], axs[0,1]
    ]:
        ax.set_ylim(plx_ylim)


    ########
    # save #
    ########
    fig.tight_layout()

    if return_figure:
        return fig

    else:
        outpath = (
            '../results/neighborhoods/{}_{}_neighborhood.png'.
            format(targetname, groupname)
        )

        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        LOGINFO('made {}'.format(outpath))
