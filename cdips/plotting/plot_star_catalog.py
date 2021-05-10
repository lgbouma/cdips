import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, subprocess, itertools, IPython
from datetime import datetime

from numpy import array as nparr

import astropy.coordinates as coord
from astropy import units as u, constants as c

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.plotting import savefig
pfrespath = '../../results/cdips_lc_periodfinding/sector-6/initial_period_finding_results_supplemented.csv'

def main():

    do_star_catalog_skymap = 1
    do_star_catalog_mag_histogram = 0
    do_star_catalog_cmd = 0
    do_star_catalog_hrd_scat = 0

    overplot_s6_results = 1

    ##########################################

    df = ccl.get_cdips_catalog(ver=0.4)
    if overplot_s6_results:
        pfdf = pd.read_csv(pfrespath)
    else:
        pfdf = None

    if do_star_catalog_skymap:
        star_catalog_skymap(df, pfdf, closesubset=True)
        #star_catalog_skymap(df)
    if do_star_catalog_mag_histogram:
        star_catalog_mag_histogram(df, 'phot_g_mean_mag')
        star_catalog_mag_histogram(df, 'phot_rp_mean_mag')
    if do_star_catalog_cmd:
        star_catalog_cmd(df)
    if do_star_catalog_hrd_scat:
        star_catalog_hrd_scat(df)

def star_catalog_skymap(df, pfdf, closesubset=False):

    if closesubset:
        df = df[df['parallax'] > 0]
        plx_as = df['parallax']/1000
        df = df[ 1/plx_as < 1000 ]

    ra = coord.Angle(nparr(df['ra'])*u.deg)
    ra = ra.wrap_at(180*u.deg)
    dec = coord.Angle(nparr(df['dec'])*u.deg)

    f = plt.figure(figsize=(4,3))
    ax = f.add_subplot(111, projection='mollweide')

    ax.scatter(ra.radian, dec.radian, rasterized=True, s=0.5, alpha=0.5,
               linewidths=0, zorder=5, c='C0', label='CDIPS stars')
    if isinstance(pfdf, pd.DataFrame):
        ra = coord.Angle(nparr(pfdf['ra_x'])*u.deg)
        ra = ra.wrap_at(180*u.deg)
        dec = coord.Angle(nparr(pfdf['dec_x'])*u.deg)
        ax.scatter(ra.radian, dec.radian, rasterized=True, s=0.5, alpha=0.5,
                   linewidths=0, zorder=5, c='C1', label='Sector 6')

    ax.set_xticklabels(['14h','16h','18h','20h','22h',
                        '0h','2h','4h','6h','8h','10h'])
    ax.tick_params(axis='both', which='major', labelsize='x-small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=-1)

    f.tight_layout()
    f.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)

    csstr = '_closesubset' if closesubset else ''
    opstr = '_overplotlcs' if isinstance(pfdf, pd.DataFrame) else ''

    outpath = '../../results/star_catalog_skymap{csstr}{opstr}.png'.format(
        csstr=csstr, opstr=opstr)
    savefig(f, outpath)


def star_catalog_mag_histogram(df, magstr, savpath=None):

    f,ax = plt.subplots(figsize=(4,3))

    bins = np.arange(np.floor(np.min(df[magstr])),
                     np.ceil(np.max(df[magstr]))+0.5,
                     1)
    ax.hist(df[magstr], bins=bins, cumulative=True, color='black', fill=False,
            linewidth=0.5)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    if magstr == 'phot_rp_mean_mag':
        ax.set_xlabel('Apparent mag [$G_\mathrm{{RP}}$]')
    if magstr == 'phot_bp_mean_mag':
        ax.set_xlabel('Apparent mag [$G_\mathrm{{BP}}$]')
    ax.set_ylabel('Cumulative number')
    ax.set_yscale('log')

    ax.set_xlim([2,16])

    f.tight_layout(pad=0.2)
    if not isinstance(savpath, str):
        savpath = '../../results/star_catalog_mag_histogram_{}.png'.format(magstr)
    savefig(f, savpath)


def star_catalog_cmd(df):

    f,ax = plt.subplots(figsize=(4,3))

    ax.scatter(df['phot_bp_mean_mag']-df['phot_rp_mean_mag'],
               df['phot_g_mean_mag'],
               rasterized=True, s=0.2, alpha=1, linewidths=0, zorder=5)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    ax.set_xlabel('$G_{\mathrm{BP}} - G_{\mathrm{RP}}$')
    ax.set_ylabel('$G$')
    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))

    f.tight_layout(pad=0.2)
    savefig(f, '../../results/star_catalog_cmd.png')


def star_catalog_hrd_scat(df):

    # 2d SCATTER CASE
    df = df[df['parallax'] > 0]
    plx_as = df['parallax']/1000
    df = df[ 1/plx_as < 1000 ]

    f,ax = plt.subplots(figsize=(4,3))

    color = df['phot_bp_mean_mag']-df['phot_rp_mean_mag']
    M_omega = df['phot_g_mean_mag'] + 5*np.log10(df['parallax']/1000) + 5

    ax.scatter(color,
               M_omega,
               rasterized=True, s=0.1, alpha=1, linewidths=0, zorder=5)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    ax.set_xlabel('$G_{\mathrm{BP}} - G_{\mathrm{RP}}$')
    ax.set_ylabel('$M_\omega = G + 5\log_{10}(\omega_{\mathrm{as}}) + 5$')
    ylim = ax.get_ylim()
    ax.set_ylim((max(ylim),min(ylim)))

    ax.set_xlim((-0.7, 4.3))

    txtstr= (
        '$\omega>0$, $1/\omega_{\mathrm{as}} < 1000$, '+'{} stars'.
        format(len(df))
    )
    ax.text(
        0.97, 0.97,
        txtstr,
        ha='right', va='top',
        fontsize='x-small',
        transform=ax.transAxes
    )

    f.tight_layout(pad=0.2)
    savefig(f, '../../results/star_catalog_hrd_scat.png')

if __name__=="__main__":
    main()
