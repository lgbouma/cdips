import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, subprocess, itertools
from datetime import datetime

from numpy import array as nparr

import astropy.coordinates as coord
from astropy import units as u, constants as c

catalogpath = '../data/cluster_data/OC_MG_FINAL_GaiaRp_lt_16.csv'

def main():
    do_star_catalog_skymap = 0
    do_star_catalog_mag_histogram = 0
    do_star_catalog_cmd = 1

    df = pd.read_csv(catalogpath, sep=';')

    if do_star_catalog_skymap:
        star_catalog_skymap(df)
    if do_star_catalog_mag_histogram:
        star_catalog_mag_histogram(df, 'phot_g_mean_mag')
        star_catalog_mag_histogram(df, 'phot_rp_mean_mag')
    if do_star_catalog_cmd:
        star_catalog_cmd(df)

def savefig(fig, figpath):
    fig.savefig(figpath, dpi=450, bbox_inches='tight')
    print('{}: made {}'.format(datetime.utcnow().isoformat(), figpath))

def star_catalog_skymap(df):

    ra = coord.Angle(df['ra']*u.deg)
    ra = ra.wrap_at(180*u.deg)
    dec = coord.Angle(df['dec']*u.deg)

    f = plt.figure(figsize=(4,3))
    ax = f.add_subplot(111, projection='mollweide')

    ax.scatter(ra.radian, dec.radian, rasterized=True, s=0.5, alpha=0.5,
               linewidths=0, zorder=5)

    ax.set_xticklabels(['14h','16h','18h','20h','22h',
                        '0h','2h','4h','6h','8h','10h'])
    ax.tick_params(axis='both', which='major', labelsize='x-small')
    ax.tick_params(axis='both', which='minor', labelsize='x-small')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=-1)

    f.tight_layout()
    f.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
    savefig(f, '../results/star_catalog_skymap.png')

def star_catalog_mag_histogram(df, magstr):

    f,ax = plt.subplots(figsize=(4,3))

    bins = np.arange(np.floor(np.min(df[magstr])),
                     np.ceil(np.max(df[magstr]))+0.5,
                     1)
    ax.hist(df[magstr], bins=bins, cumulative=True)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    ax.set_xlabel(magstr)
    ax.set_ylabel('cumulative number')
    ax.set_yscale('log')

    f.tight_layout(pad=0.2)
    savpath = '../results/star_catalog_mag_histogram_{}.png'.format(magstr)
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
    savefig(f, '../results/star_catalog_cmd.png')


if __name__=="__main__":
    main()
