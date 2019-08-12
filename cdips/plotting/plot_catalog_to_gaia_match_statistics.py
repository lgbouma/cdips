from glob import glob
import os

import matplotlib.pyplot as plt, numpy as np, pandas as pd

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.io.votable import from_table, writeto, parse

from cdips.plotting import savefig

def get_mwsc_gaia_xmatch_statistics():

    # first, get statistics for the MWSC to Gaia match.
    # since we only care about the non-globulars from MWSC, pull them out.
    mwscpath = '../data/cluster_data/Kharchenko_2013_MWSC.vot'
    tab = parse(mwscpath)
    t = tab.get_first_table().to_table()

    t = t[t['Type'] != 'g']

    datadir = '../data/cluster_data/MWSC_1sigma_members_Gaia_matched/'
    df_list, fail_list = [], []
    for ix, cname in enumerate(t['Name']):
        print("{}/{}: {}...".format(ix, len(t), cname))
        csvfile = glob(os.path.join(
            datadir,'[0-9][0-9][0-9][0-9]_{}_*.csv'.
            format(str(cname)))
        )
        if len(csvfile) != 1:
            print('WRN! should be one csv file per cluster')
            fail_list.append(cname)
            continue
        csvfile=csvfile[0]

        thisdf = pd.read_csv(csvfile)
        thisdf['cname'] = cname

        df_list.append(thisdf)

    print('-'*42)
    print('THE FOLLOWING FAILED')
    print(fail_list)
    print('-'*42)

    bigdf = pd.concat(df_list)

    return bigdf

def plot_catalog_to_gaia_match_statistics(bigdf, outpath, isD14=False):

    if not isD14:
        dist_arcsec = (np.array(bigdf['dist_deg'])*u.deg).to(u.arcsec)
    else:
        dist_arcsec = (np.array(bigdf['dist'])*u.deg).to(u.arcsec)

    f,axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

    axs[0].hist(dist_arcsec.value, bins=np.arange(0,5.5,0.5), color='black',
                fill=False, linewidth=0.5)

    axs[0].set_xlabel('Distance [arcsec]', fontsize='x-large')
    axs[0].set_ylabel('Number per bin', fontsize='x-large')
    axs[0].set_yscale('log')

    if isD14:
        axs[1].hist(bigdf['gaia_gmag']-bigdf['gmag_estimate'], color='black',
                    bins=np.arange(-2,2+0.5,0.5),
                    fill=False, linewidth=0.5)
    else:
        axs[1].hist(bigdf['gmag_match_minus_estimate'], color='black',
                    bins=np.arange(-2,2+0.5,0.5),
                    fill=False, linewidth=0.5)

    axs[1].set_xlabel('$G_{\mathrm{true}}$ - '
                      '$G_{\mathrm{pred}}$', fontsize='x-large')
    axs[1].set_ylabel('Number per bin', fontsize='x-large')
    axs[1].set_xlim([-2.1,2.1])
    axs[1].set_yscale('log')

    if isD14:
        axs[2].scatter(bigdf['gaia_gmag'],
                       bigdf['gaia_gmag']-bigdf['gmag_estimate'], s=5,
                       alpha=0.1, rasterized=True, linewidths=0, color='black')
    else:
        axs[2].scatter(bigdf['gmag_match'], bigdf['gmag_match_minus_estimate'],
                       s=5, alpha=0.1, rasterized=True, linewidths=0,
                       color='black')

    axs[2].set_xlabel('$G_{\mathrm{true}}$', fontsize='x-large')
    axs[2].set_ylabel('$G_{\mathrm{true}}$ - '
                      '$G_{\mathrm{pred}}$', fontsize='x-large')
    axs[2].set_xlim([4,18])
    axs[2].set_ylim([-2.1,2.1])

    for ax in axs:
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize('large')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize('large')

    f.tight_layout(pad=0.2, w_pad=0.5)
    savefig(f, outpath)


if __name__=="__main__":

    # Kharchenko+ 2013 catalog
    mwscconcatpath = '../data/cluster_data/MWSC_Gaia_matched_concatenated.csv'
    if not os.path.exists(mwscconcatpath):
        bigdf = get_mwsc_gaia_xmatch_statistics()
        bigdf.to_csv(mwscconcatpath, index=False)
    else:
        bigdf = pd.read_csv(mwscconcatpath)
    outpath = '../results/catalog_to_gaia_match_statistics_MWSC.png'

    plot_catalog_to_gaia_match_statistics(bigdf, outpath, isD14=False)

    # Dias 2014 catalog
    d14_df = pd.read_csv('../data/cluster_data/Dias14_seplt5arcsec_Gdifflt2.csv')
    outpath = '../results/catalog_to_gaia_match_statistics_Dias14.png'
    plot_catalog_to_gaia_match_statistics(d14_df, outpath, isD14=True)
