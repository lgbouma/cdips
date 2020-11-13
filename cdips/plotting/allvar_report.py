from glob import glob
import os, pickle, shutil, multiprocessing

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cdips.plotting import vetting_pdf as vp

from cdips.vetting import (
    centroid_analysis as cdva,
    initialize_neighborhood_information as ini
)

from numpy import array as nparr
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

nworkers = multiprocessing.cpu_count()

from astrobase import periodbase, checkplot

APSIZEDICT = {
    1: 1,
    2: 1.5,
    3: 2.25
}

def allvar_periodogram_checkplot(a):

    ap = int(a['ap'])
    time, flux, err = a['STIME'], a[f'SPCA{ap}'], a[f'SPCAE{ap}']

    spdm = periodbase.stellingwerf_pdm(
        time, flux, err, magsarefluxes=True, startp=0.05, endp=19,
        sigclip=10.0, nworkers=nworkers
    )

    lsp = periodbase.pgen_lsp(
        time, flux, err, magsarefluxes=True, startp=0.05, endp=19,
        autofreq=True, sigclip=10.0, nworkers=nworkers
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
        circleoverlay=APSIZEDICT[ap]*21, yticksize=20
    )

    axs = fig.get_axes()

    for ax in axs:
        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')

    return fig, lsp, spdm


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

    axs[0].scatter(rawtime, rawmag, c='black', alpha=0.9, zorder=2, s=50,
                   rasterized=True, linewidths=0)

    axs[1].scatter(time, flux, c='black', alpha=0.9, zorder=2, s=50,
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
        fig, lsp, spdm = allvar_periodogram_checkplot(
            allvardict
        )

        pdf.savefig(fig)
        plt.close()
        if pd.isnull(lsp):
            return

        ##########
        # page 2
        ##########

        fig = allvar_plot_timeseries_vecs(
            allvardict
        )
        pdf.savefig(fig)
        plt.close()


