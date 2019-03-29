from glob import glob
import os

import matplotlib.pyplot as plt, numpy as np, pandas as pd

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.io.votable import from_table, writeto, parse

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

def plot_catalog_to_gaia_match_statistics(bigdf):

    dist_arcsec = (np.array(bigdf['dist_deg'])*u.deg).to(u.arcsec)

    f,axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

    axs[0].hist(dist_arcsec.value)
    axs[0].set_xlabel('distance [arcsec]')
    axs[0].set_yscale('log')

    axs[1].hist(bigdf['gmag_match_minus_estimate'])
    axs[1].set_xlabel('$\mathrm{G}_{\mathrm{true}}$ -'
                        '$\mathrm{G}_{\mathrm{pred}}$')
    axs[1].set_yscale('log')

    axs[2].scatter(bigdf['gmag_match'], bigdf['gmag_match_minus_estimate'],
                   s=5, alpha=0.1, rasterized=True, linewidths=0)
    axs[2].set_xlabel('$\mathrm{G}_{\mathrm{true}}$')
    axs[2].set_ylabel('$\mathrm{G}_{\mathrm{true}}$ -'
                        '$\mathrm{G}_{\mathrm{pred}}$')

    outpath = '../results/catalog_to_gaia_match_statististics.png'
    f.tight_layout()
    f.savefig(outpath, dpi=300)
    print('saved {}'.format(outpath))


if __name__=="__main__":

    mwscconcatpath = '../data/cluster_data/MWSC_Gaia_matched_concatenated.csv'
    if not os.path.exists(mwscconcatpath):
        bigdf = get_mwsc_gaia_xmatch_statistics()
        bigdf.to_csv(mwscconcatpath, index=False)
    else:
        bigdf = pd.read_csv(mwscconcatpath)

    plot_catalog_to_gaia_match_statistics(bigdf)
