"""
make_allvar_report
    allvar_periodogram_checkplot
    allvar_plot_timeseries_vecs
    plot_rotationcheck
"""
from glob import glob
import os, pickle, shutil, multiprocessing

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from numpy import array as nparr
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.table import Table

from astroquery.vizier import Vizier
from astroquery.mast import Catalogs

from astrobase import periodbase, checkplot
from astrobase.plotbase import skyview_stamp

from cdips.plotting import vetting_pdf as vp
from cdips.paths import DATADIR

from cdips.vetting import (
    centroid_analysis as cdva,
    initialize_neighborhood_information as ini
)

nworkers = multiprocessing.cpu_count()

APSIZEDICT = {
    1: 1,
    2: 1.5,
    3: 2.25
}


def make_allvar_report(allvardict, plotdir):
    """
        allvardict = {
            'source_id': source_id,
            'ap': ap,
            'TMID_BJD': time,
            f'PCA{ap}': flux,
            f'IRE{ap}': fluxerr,
            'STIME': s_time,
            f'SPCA{ap}': s_flux
            f'SPCAE{ap}': s_flux
            'dtr_infos': dtr_infos
        }

    Each dtr_infos tuple entry contains
        primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs
    """

    source_id = allvardict['source_id']
    outpath = os.path.join(plotdir, f'{source_id}_allvar_report.pdf')

    with PdfPages(outpath) as pdf:

        ##########
        # page 1
        ##########
        fig, lsp, spdm, objectinfo = allvar_periodogram_checkplot(
            allvardict
        )

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        if pd.isnull(lsp):
            return

        ##########
        # page 2
        ##########
        fig = allvar_plot_timeseries_vecs(
            allvardict
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        ##########
        # page 3
        ##########
        fig, n_dict = plot_rotationcheck(
            allvardict, lsp, objectinfo
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        outd = {
            'lsp': lsp,
            'spdm': spdm,
            'objectinfo': objectinfo,
            'n_dict': n_dict
        }

    return outd


def allvar_periodogram_checkplot(a):

    ap = int(a['ap'])
    time, flux, err = a['STIME'], a[f'SPCA{ap}'], a[f'SPCAE{ap}']

    spdm = periodbase.stellingwerf_pdm(
        time, flux, err, magsarefluxes=True, startp=0.05, endp=30,
        sigclip=5.0, nworkers=nworkers
    )

    lsp = periodbase.pgen_lsp(
        time, flux, err, magsarefluxes=True, startp=0.05, endp=30,
        autofreq=True, sigclip=5.0, nworkers=nworkers
    )

    objectinfo = {}
    keys = [
        'objectid','ra','decl','pmra','pmdecl','teff','gmag'
    ]
    hdrkeys = [
        'Gaia-ID', 'RA_OBJ', 'DEC_OBJ', 'PM_RA[mas/yr]', 'PM_Dec[mas/year]',
        'teff_val', 'phot_g_mean_mag'
    ]
    hdr = a['dtr_infos'][0][0]
    for k,hk in zip(keys,hdrkeys):
        if hk in hdr:
            objectinfo[k] = hdr[hk]
        else:
            objectinfo[k] = np.nan

    import matplotlib as mpl
    mpl.rcParams['axes.titlesize'] = 'xx-large'
    mpl.rcParams['axes.labelsize'] = 'xx-large'

    fig = checkplot.twolsp_checkplot_png(
        lsp, spdm, time, flux, err, magsarefluxes=True, objectinfo=objectinfo,
        varepoch='min', sigclip=None, plotdpi=100, phasebin=3e-2, phasems=6.0,
        phasebinms=14.0, unphasedms=6.0, figsize=(30,24), returnfigure=True,
        circleoverlay=APSIZEDICT[ap]*21, yticksize=20, trimylim=True
    )

    axs = fig.get_axes()

    for ax in axs:
        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')

    return fig, lsp, spdm, objectinfo


def plot_rotationcheck(a, lsp, objectinfo):

    ap = int(a['ap'])
    APSIZE = APSIZEDICT[ap]
    phdr = a['dtr_infos'][0][0]
    TESSMAG = phdr['TESSMAG']

    time, flux, err = a['STIME'], a[f'SPCA{ap}'], a[f'SPCAE{ap}']
    rawtime, rawmag = a['TMID_BJD'], a['vec_dict'][f'IRM{ap}']

    ra, dec = phdr['RA_OBJ'], phdr['DEC_OBJ']
    dss, dss_hdr, sizepix = _get_dss(ra, dec)

    # Count how many stars inside the aperture are brighter.
    radius = APSIZE*21.0*u.arcsec
    nbhr_stars = Catalogs.query_region(
        "{} {}".format(float(ra), float(dec)),
        catalog="TIC",
        radius=radius
    )
    n_in_ap_equal = len(nbhr_stars[nbhr_stars['Tmag'] < TESSMAG])
    n_in_ap_close = len(nbhr_stars[nbhr_stars['Tmag'] < (TESSMAG+1.25)])
    n_in_ap_faint = len(nbhr_stars[nbhr_stars['Tmag'] < (TESSMAG+2.5)])
    n_dict = {
        'equal': n_in_ap_equal,
        'close': n_in_ap_close,
        'faint': n_in_ap_faint
    }


    #
    # make the plot!
    #

    figsize=(30,10)
    plt.close('all')

    fig = plt.figure(figsize=figsize)
    ax0 = plt.subplot2grid((1, 3), (0, 0))
    ax1 = plt.subplot2grid((1, 3), (0, 1))
    ax2 = plt.subplot2grid((1, 3), (0, 2), projection=WCS(dss_hdr))
    # fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    #
    # First: periodogram
    #
    ls_period_0 = lsp['bestperiod']
    period = lsp['periods']
    power = lsp['lspvals']

    ax0.plot(period, power)
    ax0.axvline(ls_period_0, alpha=0.4, lw=1, color='C0', ls='-')
    ax0.axvline(2*ls_period_0, alpha=0.4, lw=1, color='C0', ls='--')
    ax0.axvline(0.5*ls_period_0, alpha=0.4, lw=1, color='C0', ls='--')
    ax0.set_title(f'P = {ls_period_0:.3f} d')
    ax0.set_ylabel('LS Power')
    ax0.set_xlabel('Period [days]')
    ax0.set_xscale('log')

    #
    # Next: Prot vs (extinction corrected) Bp-Rp color.
    #
    E_BpmRp = a['E_BpmRp']
    if E_BpmRp is None:
        include_extinction = False
    else:
        include_extinction = True

    BpmRp = phdr['phot_bp_mean_mag'] - phdr['phot_rp_mean_mag']
    if include_extinction:
        BpmRp -= E_BpmRp

    rotdir = os.path.join(DATADIR)

    classes = ['pleiades', 'praesepe']
    colors = ['k', 'gray']
    zorders = [3, 2]
    markers = ['o', 'x']
    lws = [0, 0.]
    mews= [0.5, 0.5]
    ss = [3.0, 6]
    labels = ['Pleaides', 'Praesepe']

    for _cls, _col, z, m, l, lw, s, mew in zip(
        classes, colors, zorders, markers, labels, lws, ss, mews
    ):

        t = Table.read(
            os.path.join(rotdir, 'Curtis_2020_apjabbf58t5_mrt.txt'),
            format='cds'
        )
        if _cls == 'pleiades':
            df = t[t['Cluster'] == 'Pleiades'].to_pandas()
        elif _cls == 'praesepe':
            df = t[t['Cluster'] == 'Praesepe'].to_pandas()
        else:
            raise NotImplementedError

        xval = df['(BP-RP)0']

        ax1.plot(
            xval, df['Prot'], c=_col, alpha=1, zorder=z,
            markersize=s, rasterized=False, lw=lw, label=l, marker=m, mew=mew,
            mfc=_col
        )

    ax1.plot(
        BpmRp, ls_period_0,
        alpha=1, mew=0.5, zorder=8, label='Target', markerfacecolor='yellow',
        markersize=18, marker='*', color='black', lw=0
    )

    ax1.legend(loc='best', handletextpad=0.1, framealpha=0.7)
    ax1.set_ylabel('P$_\mathrm{rot}$ [days]')
    ax1.set_xlabel('($G_{\mathrm{BP}}-G_{\mathrm{RP}}$)$_0$ [mag]')
    ax1.set_ylim((0,14))

    #
    # Finally: blending check. DSS finder.
    #

    # standard tick formatting fails for these images.
    import matplotlib as mpl
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    cset2 = ax2.imshow(dss, origin='lower', cmap=plt.cm.gray_r)

    ax2.grid(ls='--', alpha=0.5)
    ax2.set_title('DSS2 Red', fontsize='xx-large')
    showcolorbar = False
    if showcolorbar:
        cb = fig.colorbar(cset2, ax=ax2, extend='neither', fraction=0.046,
                          pad=0.04)

    # DSS is ~1 arcsecond per pixel. overplot aperture that was used.
    px_to_arcsec = 21
    circle = plt.Circle((sizepix/2, sizepix/2), APSIZE*px_to_arcsec, color=f'C0',
                        fill=False, zorder=5)
    ax2.add_artist(circle)

    ax2.set_xlabel(r'$\alpha_{2000}$')
    ax2.set_ylabel(r'$\delta_{2000}$')

    #
    # clean figure
    #

    for ax in [ax0, ax1, ax2]:
        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')

    fig.tight_layout(w_pad=0.5, h_pad=0.5)

    return fig, n_dict


def _get_dss(ra, dec):

    ###########
    # get DSS #
    ###########
    sizepix = 220
    try:
        dss, dss_hdr = skyview_stamp(ra, dec, survey='DSS2 Red',
                                     scaling='Linear', convolvewith=None,
                                     sizepix=sizepix, flip=False,
                                     cachedir='~/.astrobase/stamp-cache',
                                     verbose=True, savewcsheader=True)
    except (OSError, IndexError, TypeError) as e:
        print('downloaded FITS appears to be corrupt, retrying...')
        try:
            dss, dss_hdr = skyview_stamp(ra, dec, survey='DSS2 Red',
                                         scaling='Linear', convolvewith=None,
                                         sizepix=sizepix, flip=False,
                                         cachedir='~/.astrobase/stamp-cache',
                                         verbose=True, savewcsheader=True,
                                         forcefetch=True)

        except Exception as e:
            print('failed to get DSS stamp ra {} dec {}, error was {}'.
                  format(ra, dec, repr(e)))
            return None, None, None

    return dss, dss_hdr, sizepix





def allvar_plot_timeseries_vecs(a):

    ap = int(a['ap'])
    time, flux, err = a['STIME'], a[f'SPCA{ap}'], a[f'SPCAE{ap}']
    rawtime, rawmag = a['TMID_BJD'], a['vec_dict'][f'IRM{ap}']

    #
    # make the plot!
    #

    figsize=(30,20)
    plt.close('all')
    nrows = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=figsize)

    axs[0].scatter(rawtime, rawmag, c='black', alpha=0.9, zorder=2, s=20,
                   rasterized=True, linewidths=0)

    axs[1].scatter(time, flux, c='black', alpha=0.9, zorder=2, s=20,
                   rasterized=True, linewidths=0)

    for i in range(0,7):
        yval = a['vec_dict'][f'CBV{i}']
        axs[2].scatter(rawtime, yval, c=f'C{i}', alpha=0.9, zorder=2, s=20,
                       rasterized=True, linewidths=0)

    for ax in axs:
        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')

    for ax in [axs[0]]:
        ax.set_ylim(ax.get_ylim()[::-1])

    fig.tight_layout(h_pad=0)

    axs[-1].set_xlabel('BJDTDB', fontsize='xx-large')
    axs[0].set_ylabel(f'IRM{ap}', fontsize='xx-large')
    axs[1].set_ylabel(f'SPCA{ap}', fontsize='xx-large')
    axs[2].set_ylabel(f'CBVs', fontsize='xx-large')

    return fig

