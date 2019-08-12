from glob import glob
import datetime, os, pickle, shutil, subprocess
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from numpy import array as nparr
from datetime import datetime

from astropy.io import fits
from astropy.io.votable import from_table, writeto, parse
from astropy.coordinates import SkyCoord
from astropy import units as u

from astrobase import lcmath
from astrobase.lcmath import phase_magseries, phase_bin_magseries
from astrobase.varbase.trends import (
    smooth_magseries_ndimage_medfilt as medfilt,
    smooth_magseries_savgol as savgol )
from numpy.polynomial.legendre import Legendre

import imageutils as iu
from cdips.plotting import savefig

from cdips.lcproc import mask_orbit_edges as moe
from transitleastsquares import transitleastsquares
from astropy.stats import LombScargle
from astrobase import periodbase, checkplot

from matplotlib.ticker import FormatStrFormatter

OUTDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/paper_figures/'
CLUSTERDATADIR = '/home/lbouma/proj/cdips/data/cluster_data'
LCDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/'

def plot_quilt_s6_s7(overwrite=1):

    outpath = os.path.join(OUTDIR, 'quilt_s6_s7.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; skip'.format(outpath))
        return

    gaiaids = [
        3064530810048196352, # TOI 496 -- V-shaped giant PC, NGC 2548
        3080104185367102592, # TOI 625, a UMS HJ. T=9.6, so nice.
        3027361888196408832, # eccentric EB
        3125738423345810048, # 80Myr member semidetached EB with sick OOT
        3326715714242517248, # V684 Mon, 10 Myr old detached binary.
        3064487241899832832, # semidetached EB, NGC 2548
        3024952755135530496, # two-spotted pulsator, NGC 2184
        3209428644243836928, # M42 Dias, rapid rotating m dwarf, V468 Ori
        3214130293403123712  # zhuchang-class weirdo.
        #2949587413510194688, # double-hump rotator / pulsator??
        #5274902193830445312  # sick PMS star. pulsating? spots??
    ]

    lctypes = ['PCA2']*9
    pgtypes = ['tls']*3 + ['gls'] + ['tls']*2 + ['gls','gls','gls']
    peakindices = [0,0,0,
                   0,1,0,
                   0,0,0]

    lcpaths = []
    for g in gaiaids:
        globpaths = glob(
            '/home/lbouma/cdips_lcs/sector-?/cam?_ccd?/*{}*.fits'.format(g)
        )
        if len(globpaths) >= 1:
            # weird edge case
            if g == 2949587413510194688:
                lcpaths.append(globpaths[1])
                continue
            lcpaths.append(globpaths[0])
        else:
            raise AssertionError('error! got no lc matches')

    ylims = [
             None,(0.989,1.005), None,
             None,None,None,
             None,(0.96, 1.04),None
    ]
    yticks = [
             [0.98, 1.00],[0.99, 1.00], None,
             [0.90,1.00],[0.93,1.00],[0.98, 1.01],
             [0.98,1.01], [0.97, 1.03], [0.85, 1.15]
    ]

    alphas = np.ones_like(lcpaths).astype(int)*0.7

    inds = ['']*len(lcpaths)

    f, axs = plt.subplots(nrows=3,ncols=3,figsize=(7.5,4.5))
    axs = axs.flatten()

    ix = 0

    overwritecsv = 0 # option to save time

    for fpath, ax, a, ind, lctype, pgtype, peakindex in zip(
        lcpaths, axs, alphas, inds, lctypes, pgtypes, peakindices
    ):

        plot_phase(fpath, ax, ind, alpha=a, lctype=lctype,
                   periodogramtype=pgtype, peakindex=peakindex,
                   overwritecsv=overwritecsv)
        print('{}: {}'.format(ind, gaiaids[ix]))
        ix += 1

        if ix < 7:
            ax.set_xticklabels('')

    for ix, ax in enumerate(axs):

        if ylims[ix] is not None:
            ax.set_ylim(ylims[ix])

        if yticks[ix] is not None:
            ax.set_yticks(yticks[ix])

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize('small')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize('small')

    f.text(0.5,-0.035, 'Phase', ha='center', fontsize='medium')
    f.text(-0.03,0.5, 'Relative flux', va='center', rotation=90,
           fontsize='medium')

    f.tight_layout(h_pad=0.15, w_pad=0.5, pad=0.1)

    savefig(f, outpath)


def plot_phase(fpath, ax, ind, s=3, alpha=0.3, lctype='IRM2',
               periodogramtype=None, peakindex=0, plot_bin_phase=False,
               overwritecsv=1):

    outsavpath = os.path.join(
        OUTDIR, 'quilt_s6_s7_'+os.path.basename(fpath).replace('.fits','.csv')
    )

    if os.path.exists(outsavpath) and not overwritecsv:
        df = pd.read_csv(outsavpath)
        phase = nparr(df['phase'])
        phz_flux = nparr(df['phz_flux'])
        period = nparr(df['period'])[0]

    else:
        #
        # get data. fpath here is a fits LC file. apply the periodogram requested.
        #
        time = iu.get_data_keyword(fpath, 'TMID_BJD', ext=1)
        mag = iu.get_data_keyword(fpath, lctype, ext=1)

        f_x0 = 1e4
        m_x0 = 10
        flux = f_x0 * 10**( -0.4 * (mag - m_x0) )
        flux /= np.nanmedian(flux)

        time, flux = moe.mask_orbit_start_and_end(time, flux)

        # fit out long term trend (light detrending) with median filter of 5 days.
        if 'IRM' in lctype:
            ngroups, groups = lcmath.find_lc_timegroups(time, mingap=0.5)
            assert ngroups == 2

            windowsize = 48*5 + 1 # 5 days
            tg_smooth_flux = []
            for group in groups:

                # 
                # fit out arbitrary order legendre series
                # p(x) = c_0*L_0(x) + c_1*L_1(x) + c_2*L_2(x) + ... + c_n*L_n(x)
                #
                legendredeg = 2
                p = Legendre.fit(time[group], flux[group], legendredeg)
                coeffs = p.coef
                fit_flux = p(time[group])

                tg_smooth_flux.append( flux[group]/fit_flux )

            flux = np.concatenate(tg_smooth_flux)


        if periodogramtype=='tls':
            period_min, period_max = 0.5, 5
            tlsp = periodbase.tls_parallel_pfind(time, flux, 1e-3*flux,
                                                 magsarefluxes=True,
                                                 tls_rstar_min=0.1,
                                                 tls_rstar_max=10,
                                                 tls_mstar_min=0.1,
                                                 tls_mstar_max=5.0,
                                                 tls_oversample=8,
                                                 tls_mintransits=1,
                                                 tls_transit_template='default',
                                                 nbestpeaks=5, sigclip=None,
                                                 nworkers=52)

            period = tlsp['nbestperiods'][peakindex]
            t0 = tlsp['tlsresult']['T0']
            if peakindex==1:
                t0 += period/2

        elif periodogramtype=='gls':
            period_min, period_max = 0.1, 5
            ls = LombScargle(time, flux, flux*1e-3)
            freq, power = ls.autopower(minimum_frequency=1/period_max,
                                       maximum_frequency=1/period_min,
                                       samples_per_peak=20)
            period = 1/freq[np.argmax(power)]
            t0 = time[np.argmin(flux)]

        else:
            raise NotImplementedError(
                'got {}, not imlemented'.format(periodogramtype)
            )

        #
        # phase data
        #
        phzd = phase_magseries(time, flux, period, t0, wrap=True, sort=True)

        phase = phzd['phase']
        phz_flux = phzd['mags']

    #
    # plot data
    #
    ax.scatter(phase, phz_flux, c='k', alpha=alpha, zorder=3, s=s,
               rasterized=True, linewidths=0)

    ax.text(0.88,0.03,'{:.2f}d'.format(period),
            transform=ax.transAxes, ha='right', va='bottom')

    ax.text(0.04,0.06,'{}'.format(ind),
            transform=ax.transAxes, ha='left', va='bottom')

    if overwritecsv:
        outdf = pd.DataFrame({'phase':phase, 'phz_flux':phz_flux,
                              'period':np.ones_like(phase)*period})
        outdf.to_csv(outsavpath, index=False)


    if plot_bin_phase:

        binphasedlc = phase_bin_magseries(phase,
                                          phz_flux,
                                          binsize=2e-2,
                                          minbinelems=3)
        binplotphase = binphasedlc['binnedphases']
        binplotmags = binphasedlc['binnedmags']

        ax.scatter(binplotphase, binplotmags, c='orange', alpha=alpha,
                   zorder=4, s=s, rasterized=True, linewidths=0)

