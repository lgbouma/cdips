import numpy as np, matplotlib.pyplot as plt, pandas as pd
from glob import glob
import os

from numpy import array as nparr

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

from datetime import datetime

def plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='none'):

    tab = NasaExoplanetArchive.get_confirmed_planets_table(all_columns=True)

    f,ax = plt.subplots(figsize=(6,4))

    xval = nparr(tab['pl_orbper'])
    yval = nparr(tab['pl_rade'])

    age = nparr(tab['st_age'])
    age_perr = nparr(tab['st_ageerr1'])
    age_merr = nparr(tab['st_ageerr2'])

    age_err = np.maximum(age_perr, np.abs(age_merr))

    transits = nparr(tab['pl_tranflag']).astype(bool)

    sel = (
        np.isfinite(xval) & (xval > 0)
        &
        np.isfinite(yval) & (yval > 0)
        &
        transits
    )
    if whichcut in ['finiteage', 'sigmatau_by_tau_pt5', 'sigmatau_by_tau_pt2',
                   'age_lt_1Gyr', 'age_gt_10Gyr', 'age_gt_8Gyr',
                    'age_lt_500Myr', 'age_lt_100Myr']:
        sel &= np.isfinite(age)
        sel &= (age>0)
        sel &= np.isfinite(age_perr)
        sel &= (age_perr>0)
        sel &= np.isfinite(age_merr)
        sel &= (np.abs(age_merr)>0)
    if whichcut=='sigmatau_by_tau_pt5':
        sel &= (age_err/age < 0.5)
    if whichcut=='sigmatau_by_tau_pt2':
        sel &= (age_err/age < 0.2)
    if whichcut.endswith('Gyr'):
        cutval = int(whichcut.split('Gyr')[0].split('_')[-1])
        if '_gt_' in whichcut:
            sel &= age > cutval
        elif '_lt_' in whichcut:
            sel &= age < cutval
    if whichcut.endswith('Myr'):
        cutval_Myr = int(whichcut.split('Myr')[0].split('_')[-1])
        cutval = (cutval_Myr / 1e3)
        if '_gt_' in whichcut:
            sel &= age > cutval
        elif '_lt_' in whichcut:
            sel &= age < cutval

    ax.scatter(
        xval[sel],
        yval[sel],
        rasterized=True,
        alpha=0.8,
        zorder=3,
        c='k',
        lw=0,
        s=6
    )

    ax.set_xlabel('Orbital period [days]')
    ax.set_ylabel('Radius [$R_\oplus$]')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim((0.09, 3.1e3))
    ax.set_ylim((0.25, 25))

    datestr = datetime.today().isoformat().split('T')[0]

    if whichcut=='none':
        txtstr = (
            '{} transiting\n{}\nExoplanet Archive'.
            format(len(xval[sel]),datestr)
        )
    elif whichcut=='finiteage':
        txtstr = (
            '{} transiting, w/ "age"\n{}\nExoplanet Archive'.
            format(len(xval[sel]),datestr)
        )
    elif whichcut=='sigmatau_by_tau_pt5':
        selectionstr = r'$\sigma_\tau / \tau < 0.5$'
        txtstr = (
            '{} transiting, w/ {}\n{}\nExoplanet Archive'.
            format(len(xval[sel]),selectionstr,datestr)
        )
    elif whichcut=='sigmatau_by_tau_pt2':
        selectionstr = r'$\sigma_\tau / \tau < 0.2$'
        txtstr = (
            '{} transiting, w/ {}\n{}\nExoplanet Archive'.
            format(len(xval[sel]),selectionstr,datestr)
        )
    elif whichcut.endswith('Gyr'):
        if '_gt_' in whichcut:
            selectionstr = r'$\tau > {}$ Gyr'.format(cutval)
        elif '_lt_' in whichcut:
            selectionstr = r'$\tau < {}$ Gyr'.format(cutval)
        txtstr = (
            '{} transiting, w/ {}\n{}\nExoplanet Archive'.
            format(len(xval[sel]),selectionstr,datestr)
        )
    elif whichcut.endswith('Myr'):
        if '_gt_' in whichcut:
            selectionstr = r'$\tau > {}$ Myr'.format(cutval_Myr)
        elif '_lt_' in whichcut:
            selectionstr = r'$\tau < {}$ Myr'.format(cutval_Myr)
        txtstr = (
            '{} transiting, w/ {}\n{}\nExoplanet Archive'.
            format(len(xval[sel]),selectionstr,datestr)
        )


    ax.text(
        0.98, 0.02,
        txtstr,
        ha='right', va='bottom',
        fontsize='small',
        transform=ax.transAxes
    )

    savdir = '../results/exoplanet_archive_age_cut_plots/'
    savpath = os.path.join(
        savdir,'exoplanetarchive_Rp_vs_P_cut{}.png'.format(whichcut)
    )
    f.savefig(savpath, bbox_inches='tight', dpi=400)
    print('made {}'.format(savpath))

    return tab

if __name__ == "__main__":

    _ = plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='none')
    _ = plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='finiteage')
    _ = plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='sigmatau_by_tau_pt5')
    _ = plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='sigmatau_by_tau_pt2')

    _ = plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='age_lt_1Gyr')
    _ = plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='age_gt_8Gyr')
    _ = plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='age_gt_10Gyr')
    _ = plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='age_lt_500Myr')
    _ = plot_exoplanetarchive_Rp_vs_P_age_cuts(whichcut='age_lt_100Myr')


