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
from astrobase.lcmath import phase_magseries
from cdips.plotting import savefig

OUTDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/paper_figures/'
CLUSTERDATADIR = '/home/lbouma/proj/cdips/data/cluster_data'
LCDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/'

def plot_quilt_PCs(overwrite=1, paper_aspect_ratio=1):
    """
    paper_aspect_ratio: if true, uses figsize set for paper. if false, uses
    figsize set for poster.
    """

    aspectstr = '' if paper_aspect_ratio else '_poster'
    outpath = os.path.join(OUTDIR, 'quilt_PCs{}.png'.format(aspectstr))
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; skip'.format(outpath))
        return

    # looked through by eye in geeqie. copied from geeqie paths, if they looked
    # good enough to show off in a plot like this. PEP8 forgive me.
    fpaths = [
        '/home/lbouma/proj/cdips/results/fit_gold/sector-6/fitresults/hlsp_cdips_tess_ffi_gaiatwo0003007171311355035136-0006_tess_v01_llc/hlsp_cdips_tess_ffi_gaiatwo0003007171311355035136-0006_tess_v01_llc_fitparameters_phased_mandelagol_fit_empiricalerrs.png',
        '/home/lbouma/proj/cdips/results/fit_gold/sector-6/fitresults/hlsp_cdips_tess_ffi_gaiatwo0003125263468681400320-0006_tess_v01_llc/hlsp_cdips_tess_ffi_gaiatwo0003125263468681400320-0006_tess_v01_llc_fitparameters_phased_mandelagol_fit_empiricalerrs.png',
        '/home/lbouma/proj/cdips/results/fit_gold/sector-6/fitresults/hlsp_cdips_tess_ffi_gaiatwo0003220266049321724416-0006_tess_v01_llc/hlsp_cdips_tess_ffi_gaiatwo0003220266049321724416-0006_tess_v01_llc_fitparameters_phased_mandelagol_fit_empiricalerrs.png',
        '/home/lbouma/proj/cdips/results/fit_gold/sector-7/fitresults/hlsp_cdips_tess_ffi_gaiatwo0003027361888196408832-0007_tess_v01_llc/hlsp_cdips_tess_ffi_gaiatwo0003027361888196408832-0007_tess_v01_llc_fitparameters_phased_mandelagol_fit_empiricalerrs.png',
        '/home/lbouma/proj/cdips/results/fit_gold/sector-7/fitresults/hlsp_cdips_tess_ffi_gaiatwo0003114869682184835584-0007_tess_v01_llc/hlsp_cdips_tess_ffi_gaiatwo0003114869682184835584-0007_tess_v01_llc_fitparameters_phased_mandelagol_fit_empiricalerrs.png',
        '/home/lbouma/proj/cdips/results/fit_gold/sector-7/fitresults/hlsp_cdips_tess_ffi_gaiatwo0005510676828723793920-0007_tess_v01_llc/hlsp_cdips_tess_ffi_gaiatwo0005510676828723793920-0007_tess_v01_llc_fitparameters_phased_mandelagol_fit_empiricalerrs.png',
        '/home/lbouma/proj/cdips/results/fit_gold/sector-7/fitresults/hlsp_cdips_tess_ffi_gaiatwo0005546259498914769280-0007_tess_v01_llc/hlsp_cdips_tess_ffi_gaiatwo0005546259498914769280-0007_tess_v01_llc_fitparameters_phased_mandelagol_fit_empiricalerrs.png',
        '/home/lbouma/proj/cdips/results/fit_gold/sector-7/fitresults/hlsp_cdips_tess_ffi_gaiatwo0005605128927705695232-0007_tess_v01_llc/hlsp_cdips_tess_ffi_gaiatwo0005605128927705695232-0007_tess_v01_llc_fitparameters_phased_mandelagol_fit_empiricalerrs.png',
        '/home/lbouma/proj/cdips/results/fit_gold/sector-7/fitresults/hlsp_cdips_tess_ffi_gaiatwo0005617126180115568256-0007_tess_v01_llc/hlsp_cdips_tess_ffi_gaiatwo0005617126180115568256-0007_tess_v01_llc_fitparameters_phased_mandelagol_fit_empiricalerrs.png'
    ]

    np.random.seed(42)
    spaths = np.random.choice(fpaths, size=3*2, replace=False)
    ylims = [ # for seed 42
        (0.957, 1.015),
        (0.97, 1.017),
        (0.995, 1.003),
        (0.983, 1.008),
        (0.983, 1.008),
        (0.972, 1.008)
    ]
    yticks = [
        [0.975, 1.000], [0.98, 1.00],
        [0.996, 1.000], [0.99, 1.00],
        [0.990, 1.000], [0.98, 1.00]
    ]
    alphas = np.ones_like(spaths)
    #alphas = [
    #    0.45, 0.6, 0.5, 0.45, 0.5, 0.45
    #]
    inds = ['']*6 # ['a)','b)','c)','d)','e)','f)']

    gaiaids = list(map(
        lambda x: int(
            os.path.basename(x).split('gaiatwo')[1].split('-')[0].lstrip('0')
        ), spaths
    ))

    if paper_aspect_ratio:
        f, axs = plt.subplots(nrows=3,ncols=2,figsize=(6,4.5))
    else:
        f, axs = plt.subplots(nrows=3,ncols=2,figsize=(6,3))
    axs = axs.flatten()

    ix = 0
    for fpath, ax, a, ind in zip(spaths, axs, alphas, inds):
        plot_phase_PC(fpath, ax, ind, alpha=a)
        print('{}: {}'.format(ind, gaiaids[ix]))
        ix += 1

    for ix, ax in enumerate(axs):
        ax.set_ylim(ylims[ix])
        if yticks[ix] is not None:
            ax.set_yticks(yticks[ix])
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    #f.text(0.5,0, 'Phase', ha='center')
    f.text(0.5,-0.02, 'Time from transit center [hours]', ha='center',
           fontsize='medium')
    f.text(-0.01,0.5, 'Relative flux', va='center', rotation=90,
           fontsize='medium')

    f.tight_layout(h_pad=0.35, w_pad=0.85, pad=0.8)
    savefig(f, outpath)


def plot_phase_PC(fpath, ax, ind, s=4, alpha=0.3, show_model=True):
    #
    # get data. fpath here is a png file of the phase-folded model LC.
    #
    fitdir = os.path.dirname(fpath)

    fitpkl = glob(os.path.join(fitdir, '*empiricalerrs.pickle'))
    assert len(fitpkl) == 1
    fitpkl = fitpkl[0]
    with open(fitpkl, mode='rb') as f:
        d = pickle.load(f)

    fitcsv = glob(os.path.join(fitdir, '*fitparameters.csv'))
    assert len(fitcsv) == 1
    fitcsv = fitcsv[0]
    fit_df = pd.read_csv(fitcsv, sep="|")

    period = float(fit_df['period'])
    t0 = float(fit_df['epoch'])
    duration = float(fit_df['duration'])/24

    time, flux = d['magseries']['times'], d['magseries']['mags']
    assert d['magseries']['magsarefluxes']

    #
    # phase data
    #
    phzd = phase_magseries(time, flux, period, t0, wrap=True, sort=True)

    phase = phzd['phase']
    phz_flux = phzd['mags']

    #
    # plot data
    #
    ax.scatter(phase*period*24, phz_flux, c='k', alpha=alpha, zorder=3, s=s,
               rasterized=True, linewidths=0)

    ax.text(0.96,0.06,'{:.2f}d'.format(period),
            transform=ax.transAxes, ha='right', va='bottom')

    ax.text(0.04,0.06,'{}'.format(ind),
            transform=ax.transAxes, ha='left', va='bottom')

    ax.set_xlim((-4*duration*24,4*duration*24))

    #
    # get and plot model
    #
    if show_model:
        modeltime = time
        modelflux = d['fitinfo']['fitmags']

        model_phzd = phase_magseries(modeltime, modelflux, period, t0, wrap=True,
                                     sort=True)
        model_phase = model_phzd['phase']
        model_phz_flux = model_phzd['mags']

        ax.plot(model_phase*period*24, model_phz_flux, zorder=2, linewidth=0.5,
                alpha=0.9, color='C0', rasterized=True)




