"""
execution environment: cdips, + pipe-trex .pth file in
/home/lbouma/miniconda3/envs/cdips/lib/python3.7/site-packages

python -u paper_plot_all_figures.py &> logs/paper_plot_all.log &
"""
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

from aperturephot import get_lc_statistics
import lcstatistics as lcs

from cdips.utils import tess_noise_model as tnm
from cdips.plotting import plot_star_catalog as psc
from cdips.plotting import plot_catalog_to_gaia_match_statistics as xms
from cdips.plotting import plot_wcsqa as wcsqa
from cdips.utils import collect_cdips_lightcurves as ccl

from skim_cream import plot_initial_period_finding_results

from collections import Counter

OUTDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/paper_figures/'
CLUSTERDATADIR = '/home/lbouma/proj/cdips/data/cluster_data'
LCDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/'

def main():

    sectors = [6,7]

    # fig N: tls_sde_vs_period_scatter
    plot_tls_sde_vs_period_scatter(sectors, overwrite=0)

    # fig N: average autocorrelation fn of LCs
    plot_avg_acf(sectors, overwrite=0, cleanprevacf=True)

    # fig N: 3x2 quilty of phased PC
    # plot_quilt_PCs(overwrite=1)

    # fig N: LS period vs color evolution in time
    plot_LS_period_vs_color_and_age(sectors, overwrite=1, OC_MG_CAT_ver=0.3)

    # fig N: target star provenance
    plot_target_star_reference_pie_chart(OC_MG_CAT_ver=0.3, overwrite=1)

    # fig N: T magnitude CDF for all CDIPS target stars.
    plot_target_star_cumulative_counts(OC_MG_CAT_ver=0.3, overwrite=1)

    # fig N: wcs quality verification for one photometric reference
    plot_wcs_verification(overwrite=1)

    # fig N: catalog_to_gaia_match_statistics for CDIPS target stars
    plot_catalog_to_gaia_match_statistics(overwrite=1)

    # fig N: histogram of CDIPS target star age.
    plot_target_star_hist_logt(OC_MG_CAT_ver=0.3, overwrite=1)

    # fig N: RMS vs catalog T mag for LC stars
    plot_rms_vs_mag(sectors, overwrite=1)

    # fig N: histogram (or CDF) of T magnitude for LC stars
    plot_cdf_T_mag(sectors, overwrite=1)

    # fig N: histogram of ages of LC stars
    plot_hist_logt(sectors, overwrite=1)

    # fig N: HRD for CDIPS LC stars.
    plot_hrd_scat(sectors, overwrite=1, close_subset=1)
    plot_hrd_scat(sectors, overwrite=1, close_subset=0)

    # fig N: pmRA and pmDEC scatter for CDIPS LC stars.
    plot_pm_scat(sectors, overwrite=1, close_subset=1)
    plot_pm_scat(sectors, overwrite=1, close_subset=0)

    # fig N: positions of field and cluster LC stars (currently all cams)
    plot_cluster_and_field_star_scatter(sectors=sectors, overwrite=1,
                                        galacticcoords=True)
    plot_cluster_and_field_star_scatter(sectors=sectors, overwrite=0)
    plot_cluster_and_field_star_scatter(sectors=[6], overwrite=0, cams=[1],
                                        ccds=[1,2,3,4])

    # fig N: histogram (or CDF) of TICCONT. unfortunately this is only
    # calculated for CTL stars, so by definition it has limited use
    plot_cdf_cont(sectors, overwrite=0)


def savefig(fig, figpath):
    fig.savefig(figpath, dpi=450, bbox_inches='tight')
    print('{}: made {}'.format(datetime.utcnow().isoformat(), figpath))

    pdffigpath = figpath.replace('.png','.pdf')
    fig.savefig(pdffigpath, bbox_inches='tight')
    print('{}: made {}'.format(datetime.utcnow().isoformat(), pdffigpath))

def plot_tls_sde_vs_period_scatter(sectors, overwrite=1):

    outpath = os.path.join(
        OUTDIR, 'tls_sde_vs_period_scatter.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    if len(sectors) != 2:
        raise AssertionError

    f,axs = plt.subplots(nrows=2, sharex=True, figsize=(4,6))

    for sector, ax in zip(sectors, axs):

        pfdir = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/'
                 'results/cdips_lc_periodfinding/sector-{}'.format(sector))
        pfpath = os.path.join(
            pfdir, 'initial_period_finding_results_with_limit.csv')

        df = pd.read_csv(pfpath, sep=',')

        ax.scatter(df['tls_period'], df['tls_sde'], c='k', alpha=1, s=0.2,
                   rasterized=True, linewidths=0)

        ax.scatter(df['tls_period'], df['limit'], c='C1', alpha=1, rasterized=True,
                   linewidths=0, zorder=2, s=1)
        #ax.scatter(df['tls_period'], df['limit'], c='C1', alpha=1, rasterized=True,
        #           linewidths=0, zorder=2, s=0.2)

        txt = ('$N_{{\mathrm{{above}}}}$: '+
                '{}'.format(len(df[df['tls_sde']>df['limit']]))+
                '\n$N_{{\mathrm{{below}}}}$: '+
                '{}'.format(len(df[df['tls_sde']<df['limit']])) )

        ax.text(0.96, 0.96, txt, ha='right', va='top', fontsize='medium',
                transform=ax.transAxes)

        ax.set_xscale('log')

        ax.set_ylim([0,40])
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    f.text(0.5,-0.01, 'TLS peak period [days]', ha='center')
    f.text(-0.03,0.5, 'TLS SDE', va='center', rotation=90)

    f.tight_layout(h_pad=0.2, pad=0.2)
    savefig(f, outpath)



def plot_avg_acf(sectors, size=10000, overwrite=0, percentiles=[25,50,75],
                 cleanprevacf=True):

    outpath = os.path.join(OUTDIR, 'avg_acf.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    #
    # collect acfs for however many random LCs passed
    #
    np.random.seed(42)
    lcpaths = []
    for sector in sectors:
        lcpaths.append(
            np.random.choice(
                glob(os.path.join(LCDIR, 'sector-{}'.format(sector),
                                  'cam?_ccd?', 'hlsp_*llc.fits')),
                size=size,
                replace=False
            )
        )
    lcpaths = nparr(lcpaths).flatten()

    acfdir = os.path.join(OUTDIR, 'avg_acf_data')

    if cleanprevacf:
        # annoyingly fails. maybe nfs problem?
        pass
        #cmd = 'rm -rf {}'.format(acfdir)
        #subprocess.call(cmd)

    if not os.path.exists(acfdir):
        os.mkdir(acfdir)
    lcs.parallel_compute_acf_statistics(
        lcpaths, acfdir, nworkers=40,
        eval_times_hr=np.arange(1,301,1),
        skipepd=True)
    acfstatfiles = glob(os.path.join(acfdir,'*_acf_stats.csv'))
    df = lcs.read_acf_stat_files(acfstatfiles)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(4,3))

    linestyles = ['--','-',':']

    #
    # plot raw lines
    #
    apstr = 'RAW2'
    timelags = np.sort(np.unique(df['LAG_TIME_HR']))

    percentile_dict = {}
    for timelag in timelags:
        percentile_dict[timelag] = {}
        sel = df['LAG_TIME_HR']==timelag
        for percentile in percentiles:
            val = np.nanpercentile(df[sel][apstr+'_ACF'], percentile)
            percentile_dict[timelag][percentile] = np.round(val,7)
        pctile_df = pd.DataFrame(percentile_dict)

    ind = 0
    for ix, row in pctile_df.iterrows():
        pctile = row.name
        label = '{}%'.format(str(pctile))
        timelags = nparr(row.index)
        vals = nparr(row)

        ax.plot(timelags, vals, ls=linestyles[ind], color='gray', zorder=2)
        ax.text(0.75, 0.8, 'Raw', ha='center', va='center',
                fontsize='medium', transform=ax.transAxes, color='gray')


        ind += 1

    #
    # plot TFA lines
    #
    apstr = 'TFA2'

    percentile_dict = {}
    for timelag in timelags:
        percentile_dict[timelag] = {}
        sel = df['LAG_TIME_HR']==timelag
        for percentile in percentiles:
            val = np.nanpercentile(df[sel][apstr+'_ACF'], percentile)
            percentile_dict[timelag][percentile] = np.round(val,7)
        pctile_df = pd.DataFrame(percentile_dict)

    ind = 0
    for ix, row in pctile_df.iterrows():
        pctile = row.name
        label = '{}%'.format(str(pctile))
        timelags = nparr(row.index)
        vals = nparr(row)

        ax.plot(timelags, vals, label=label, ls=linestyles[ind], color='black',
                zorder=3)
        ind += 1

        ax.text(0.42, 0.42, 'TFA detrended', ha='center', va='top',
                fontsize='medium', transform=ax.transAxes)

    ax.legend(loc='lower left', fontsize='medium')

    ax.set_yscale('linear')
    ax.set_xscale('log')
    ax.set_xlabel('Time lag [hr]')
    ax.set_ylabel('Autocorrelation')

    # titlestr = '{:s} - {:d} ACFs - {:s}'.format(
    #     outprefix,
    #     len(acfstatfiles),
    #     '{:s} percentiles'.format(repr(percentiles))
    # )
    # ax.set_title(titlestr, fontsize='small')

    #plt.gca().grid(color='#a9a9a9',
    #               alpha=0.9,
    #               zorder=0,
    #               linewidth=1.0,
    #               linestyle=':')

    ax.set_ylim((-1,1))

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    fig.tight_layout(h_pad=0.35, pad=0.2)
    savefig(fig, outpath)



def plot_LS_period_vs_color_and_age(sectors, overwrite=0, OC_MG_CAT_ver=0.3):
    """
    * plot lomb-scargle periodogram peak period vs Gaia-G magnitude.
      (or vs Bp-Rp).
      should show : there's a lot of science to do here.

      * cut on FAP < 1e-20, or something.

      * selected by e.g., only "good member lists" -- Kharchenko, and
        Cantat-Gaudin. never Dias.

      * focus on any cluster larger than 100 Myr...
    """

    outpath = os.path.join(OUTDIR, 'LS_period_vs_color_and_age.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    pfdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding'
    pfpaths = []
    for sector in sectors:
        thisdir = os.path.join(pfdir, 'sector-{}'.format(sector))
        pfpath = os.path.join(thisdir, 'initial_period_finding_results.csv')
        pfpaths.append(pfpath)

    pfdf = pd.concat((pd.read_csv(f) for f in pfpaths))
    cddf = ccl.get_cdips_pub_catalog(ver=OC_MG_CAT_ver)

    pfdf['source_id'] = pfdf['source_id'].astype(np.int64)
    cddf['source_id'] = cddf['source_id'].astype(np.int64)

    # cols are:
    # 'source_id', 'ls_period', 'ls_fap', 'tls_period', 'tls_sde', 'tls_t0',
    # 'tls_depth', 'tls_duration', 'pspline_detrended', 'xcc', 'ycc', 'ra_x',
    # 'dec_x', 'cluster', 'reference', 'ext_catalog_name', 'ra_y', 'dec_y',
    # 'pmra', 'pmdec', 'parallax', 'phot_g_mean_mag', 'phot_bp_mean_mag',
    # 'phot_rp_mean_mag', 'k13_name_match', 'unique_cluster_name', 'how_match',
    # 'not_in_k13', 'comment', 'k13_logt', 'k13_e_logt'
    mdf = pfdf.merge(cddf, how='left', on='source_id')

    FAP_CUTOFF = 1e-15
    sel = mdf['reference'].str.contains('CantatGaudin_2018')
    sel |= mdf['reference'].str.contains('Kharchenko2013')
    sel |= mdf['reference'].str.contains('GaiaCollaboration2018')
    sel &= ~pd.isnull(mdf['phot_rp_mean_mag'] - mdf['phot_bp_mean_mag'])
    sel &= mdf['ls_fap'] < FAP_CUTOFF
    sel &= ~pd.isnull(mdf['k13_logt'])

    sdf = mdf[sel]

    # first, remove duplicates that match source ids. (these are in multiple
    # sectors) -- take the lower ls_fap for these. then, remove duplicates that
    # are close in source_id and also close in period, since they are
    # photometric blends. also take the lowest ls_fap from these.

    sdf = sdf.sort_values(by=['source_id','ls_fap'])

    sdf = sdf.drop_duplicates(subset='source_id', keep='first')

    sdf['round_source_id'] = np.round(sdf['source_id'], decimals=-14)
    sdf['round_ls_period'] = np.round(sdf['ls_period'], decimals=4)

    sdf = sdf.drop_duplicates(subset=['round_source_id','round_ls_period'],
                              keep='first')

    #TODO could use...
    cnt = Counter(nparr(sdf['unique_cluster_name']))
    mostcommon = cnt.most_common(n=20)
    print('LS FAP<{:.0e}, (CG18|K13|G18) in ref, has age'.format(FAP_CUTOFF))
    print('most common are\n{}'.format(repr(mostcommon)))

    #
    # finally, plot
    #
    bins = [(7.4,8), (8,8.333), (8.333, 8.666), (8.666,9)]#, (9,10)]
    min_all, max_all = 7.4, 9
    ixs = range(len(bins))
    nrows = len(bins)

    f,axs = plt.subplots(nrows=nrows, ncols=1, figsize=(3.5,2*nrows),
                         sharex=True)
    axs = axs.flatten()

    for ix, _bin, ax in zip(ixs, bins, axs):

        min_age = _bin[0]
        max_age = _bin[1]

        # first, overplot the emphasized subset
        sel = (sdf['k13_logt'] >= min_age) & (sdf['k13_logt'] < max_age)
        xval = sdf[sel]['phot_bp_mean_mag'] - sdf[sel]['phot_rp_mean_mag']
        yval = sdf[sel]['ls_period']
        ax.scatter(xval, yval, rasterized=True, s=5, alpha=0.9, linewidths=0,
                   zorder=3, c='black')

        l = (
            '{:.2f}'.format(min_age) +
            r'$\geq \log_{{10}}$(age) $>$' +
            '{:.2f}'.format(max_age) +
            '. {} stars'.format(len(sdf[sel]))
        )
        ax.text(0.97, 0.97, l, ha='right', va='top', fontsize='x-small',
                transform=ax.transAxes)

        # underplot the entire set
        allsel = (sdf['k13_logt'] >= min_all) & (sdf['k13_logt'] < max_all)
        xval = sdf[allsel]['phot_bp_mean_mag'] - sdf[allsel]['phot_rp_mean_mag']
        yval = sdf[allsel]['ls_period']
        ax.scatter(xval, yval, rasterized=True, s=5, alpha=0.9, linewidths=0,
                   zorder=2, c='lightgray')

    for ax in axs:
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize('small')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize('small')

        ax.set_yscale('log')
        ax.set_xlim((-0.6,3.6))
        ax.set_ylim((0.4, 11))

    f.text(0.5,-0.01, '$G_{{\mathrm{{Bp}}}} - G_{{\mathrm{{Rp}}}}$', ha='center')
    f.text(-0.03,0.5, 'Period of Lomb-Scargle peak [days]', va='center', rotation=90)

    txtstr= (
        'LS FAP<{:.0e}, (CG18|K13|G18) in ref, has age\n'.format(FAP_CUTOFF)
    )
    #axs[0].set_title(txtstr, fontsize='small')

    f.tight_layout(h_pad=0.35, pad=0.2)
    savefig(f, outpath)


def plot_quilt_PCs(overwrite=1):

    outpath = os.path.join(OUTDIR, 'quilt_PCs.png')
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
    alphas = np.ones_like(spaths)
    #alphas = [
    #    0.45, 0.6, 0.5, 0.45, 0.5, 0.45
    #]
    inds = ['a)','b)','c)','d)','e)','f)']

    gaiaids = list(map(
        lambda x: int(
            os.path.basename(x).split('gaiatwo')[1].split('-')[0].lstrip('0')
        ), spaths
    ))

    f, axs = plt.subplots(nrows=3,ncols=2,figsize=(6,4.5))
    axs = axs.flatten()

    ix = 0
    for fpath, ax, a, ind in zip(spaths, axs, alphas, inds):
        plot_phase_PC(fpath, ax, ind, alpha=a)
        print('{}: {}'.format(ind, gaiaids[ix]))
        ix += 1

    for ix, ax in enumerate(axs):
        ax.set_ylim(ylims[ix])
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    #f.text(0.5,0, 'Phase', ha='center')
    f.text(0.5,0, 'Time from transit center [hours]', ha='center')
    f.text(-0.01,0.5, 'Relative flux', va='center', rotation=90)

    f.tight_layout(h_pad=0.35, w_pad=0.85, pad=0.95)
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

    ax.text(0.96,0.06,'P={:.2f}d'.format(period),
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



def plot_wcs_verification(overwrite=1):

    outpaths = glob(os.path.join(OUTDIR, '*_quiver_meas_proj_sep.png'))
    if len(outpaths) >= 1 and not overwrite:
        print('found quiver and no overwrite; skip')
        return

    wcsqa.main(
        fitsfile='proj1500-s0006-cam1-ccd1-combinedphotref-onenight.fits',
        refbasedir='/nfs/phtess2/ar0/TESS/FFI/BASE/reference-frames/',
        matchedinpath='proj1500-s0006-cam1-ccd1-combinedphotref-onenight.matched',
        isspocwcs=True,
        outdir=OUTDIR
    )


def plot_catalog_to_gaia_match_statistics(overwrite=1):

    ##########################################
    # Kharchenko+ 2013 catalog
    mwscconcatpath = os.path.join(
        CLUSTERDATADIR, 'MWSC_Gaia_matched_concatenated.csv')
    if not os.path.exists(mwscconcatpath):
        bigdf = xms.get_mwsc_gaia_xmatch_statistics()
        bigdf.to_csv(mwscconcatpath, index=False)
    else:
        bigdf = pd.read_csv(mwscconcatpath)

    # Made via open_cluster_xmatch_utils.Kharchenko2013_position_mag_match_Gaia
    # which annoyingly did not include the G_Rp < 16 cut. which is what we care
    # about...
    sourcepath = os.path.join(CLUSTERDATADIR,
                              'MWSC_Gaia_matched_concatenated_sources.csv')
    if not os.path.exists(sourcepath):
        bigdf['gaia_dr2_match_id'].to_csv(sourcepath,index=False)

    targetpath = os.path.join(CLUSTERDATADIR,
                              'MWSC_Gaia_matched_Gaiainfo.csv')
    targetvot = os.path.join(CLUSTERDATADIR,
                             'mwsc_gaia_matched_concatenated_sources-result.vot.gz')

    if not os.path.exists(targetvot):
        raise RuntimeError('run the gaia2read on mwsc the sourcepath list, '
                           'manually')
        # Upload MWSC_Gaia_matched_concatenated_sources.csv to Gaia archive,
        # run something like the "mwsc_gaia_matched_concatenated_sources" job,
        # """
        # select u.source_id, g.phot_rp_mean_mag
        # from user_lbouma.dias14_seplt5arcsec_gdifflt2_sources import as u,
        # gaiadr2.gaia_source as g
        # WHERE
        # u.source_id = g.source_id
        # """
        # NOTE gaia2read fails because of some bad IDs.

    if not os.path.exists(targetpath):
        tab = parse(targetvot)
        t = tab.get_first_table().to_table()
        gdf = t.to_pandas()
        df = gdf.merge(bigdf, how='left', right_on='gaia_dr2_match_id', left_on='source_id')
        df.to_csv(targetpath,index=False)
    else:
        df = pd.read_csv(targetpath)

    outpath = os.path.join(
        OUTDIR,'catalog_to_gaia_match_statistics_MWSC.png'
    )
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; skip'.format(outpath))
    else:
        df = df[df['phot_rp_mean_mag'] < 16]
        xms.plot_catalog_to_gaia_match_statistics(df, outpath, isD14=False)

    ##########################################
    # Dias 2014 catalog
    d14_df = pd.read_csv(os.path.join(
        CLUSTERDATADIR,'Dias14_seplt5arcsec_Gdifflt2.csv'))

    ##########
    sourcepath = os.path.join(CLUSTERDATADIR,
                              'Dias14_seplt5arcsec_Gdifflt2_sources.csv')
    if not os.path.exists(sourcepath):
        d14_df['source_id'].to_csv(sourcepath,index=False)

    targetpath = os.path.join(CLUSTERDATADIR,
                              'Dias14_seplt5arcsec_Gdifflt2_Gaiainfo.csv')
    targetvot = os.path.join(CLUSTERDATADIR,
                              'dias14_seplt5arcsec_gdifflt2_sources-result.vot.gz')

    if not os.path.exists(targetvot):
        raise RuntimeError('run the gaia2read on the d14 sourcepath list, '
                           'manually')

    if not os.path.exists(targetpath):
        tab = parse(targetvot)
        t = tab.get_first_table().to_table()
        gdf = t.to_pandas()
        df = gdf.merge(d14_df, how='left', right_on='source_id', left_on='source_id')
        df.to_csv(targetpath,index=False)
    else:
        df = pd.read_csv(targetpath)

    outpath = os.path.join(
        OUTDIR,'catalog_to_gaia_match_statistics_Dias14.png'
    )
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; skip'.format(outpath))
    else:
        df = df[df['phot_rp_mean_mag'] < 16]
        xms.plot_catalog_to_gaia_match_statistics(df, outpath, isD14=True)


def plot_target_star_cumulative_counts(OC_MG_CAT_ver=0.3, overwrite=1):

    cdips_df = ccl.get_cdips_catalog(ver=OC_MG_CAT_ver)

    outpath = os.path.join(OUTDIR, 'target_star_cumulative_counts.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    psc.star_catalog_mag_histogram(cdips_df, 'phot_rp_mean_mag', savpath=outpath)


def plot_target_star_hist_logt(OC_MG_CAT_ver=0.3, overwrite=1):

    mdf = ccl.get_cdips_pub_catalog(ver=OC_MG_CAT_ver)

    outpath = os.path.join(OUTDIR, 'target_star_hist_logt.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    n_total = len(mdf)
    mdf = mdf[~pd.isnull(mdf['k13_logt'])]
    n_with_age = len(mdf)

    f,ax = plt.subplots(figsize=(4,3))

    bins = np.arange(5.5,
                     10.5,
                     0.5)
    ax.hist(mdf['k13_logt'], bins=bins, cumulative=False, color='black',
            fill=False, linewidth=0.5)

    txtstr = '$N_{{\mathrm{{total}}}}$: {}'.format(n_total)
    txtstr += '\n$N_{{\mathrm{{with\ ages}}}}$: {}'.format(n_with_age)
    ax.text(
        0.03, 0.97,
        txtstr,
        ha='left', va='top',
        fontsize='x-small',
        transform=ax.transAxes
    )

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    ax.set_xlabel('log$_{10}$(age [years])')
    ax.set_ylabel('Number per bin')
    ax.set_yscale('log')
    ax.set_xlim([5.5, 10.5])
    ax.set_ylim([3e3, 3e5])

    f.tight_layout(pad=0.2)
    savefig(f, outpath)


def plot_target_star_reference_pie_chart(OC_MG_CAT_ver=0.3, overwrite=1):

    cdips_df = ccl.get_cdips_catalog(ver=OC_MG_CAT_ver)

    outpath = os.path.join(OUTDIR, 'target_star_reference_pie_chart.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    #
    # Collect the info:
    # ('Dias2014', 470313), ('Kharchenko2013', 183992), ('CantatGaudin_2018',
    # 177913), ('Zari_2018_UMS', 82875), ('Zari_2018_PMS', 35488)
    #
    cnt = Counter(nparr(cdips_df['reference']))
    atleast2 = 0
    for k in cnt.keys():
        if ',' in k:
            atleast2 += cnt[k]

    mostcommon = cnt.most_common(n=5)
    mostcommon.append(('>2', atleast2))

    s = np.sum([e[1] for e in mostcommon])

    # Single-source, not otherwise counted
    mostcommon.append(('Other', len(cdips_df)-s))

    #
    # make the plot 
    #
    f,ax = plt.subplots(figsize=(4*1.5,3*1.5))

    sizes = [e[1] for e in mostcommon]
    labels = ['D14', 'K13', 'CG18', 'Z18UMS', 'Z18PMS', '$\geq$2', 'Other']

    colors = plt.cm.Greys(np.linspace(0.1,0.8,num=len(labels)))

    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%',
                                      pctdistance=1.15, colors=colors,
                                      textprops=dict(fontsize='large'))

    patterns = ('.', '', 'o', '-', 'x', '*', '')
    for wedge, pattern in zip(wedges, patterns):
        wedge.set_hatch(pattern)

    leg = ax.legend(wedges, labels,
                    title="Reference",
                    loc="center left",
                    fontsize='large',
                    labelspacing=1.3,
                    handleheight=1.7,
                    bbox_to_anchor=(1.07, 0.1, 0.2, 0.8)) #x,y,width,height
    plt.setp(leg.get_title(), fontsize='large')

    ax.axis('equal')

    f.tight_layout(pad=0.2)
    savefig(f, outpath)


def _get_rms_vs_mag_df(sectors):

    cams = [1,2,3,4]
    ccds = [1,2,3,4]
    csvpaths = []

    for sector in sectors:
        for cam in cams:
            for ccd in ccds:

                csvpath = os.path.join(
                    OUTDIR,'sector{}_cam{}_ccd{}_rms_vs_mag_data.csv'.
                    format(sector, cam, ccd)
                )

                csvpaths.append(csvpath)

    df = pd.concat((pd.read_csv(f) for f in csvpaths))
    if len(df) == 0:
        raise AssertionError('need to run rms vs mag first!!')

    return df


def plot_cdf_T_mag(sectors, overwrite=0):

    prestr = 'sector{}_'.format(sectors[0]) if len(sectors)==1 else ''

    outpath = os.path.join(OUTDIR, prestr+'cdf_T_mag.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df(sectors)
    print('N before finite lc length cut: {}'.format(len(df)))
    sel = (df['ndet_tf1'] > 1) | (df['ndet_tf2'] > 1) | (df['ndet_tf3'] > 1)
    df = df[sel]
    print('N after finite lc length cut: {}'.format(len(df)))

    f,ax = plt.subplots(figsize=(4,3))

    magstr = 'TESSMAG'
    bins = np.arange(np.floor(np.min(df[magstr])),
                     np.ceil(np.max(df[magstr]))+1,
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
    ax.set_xlabel('TESS magnitude')
    ax.set_ylabel('Cumulative number')
    ax.set_yscale('log')

    f.tight_layout(pad=0.2)
    savefig(f, outpath)


def plot_hist_logt(sectors, overwrite=0):

    prestr = 'sector{}_'.format(sectors[0]) if len(sectors)==1 else ''

    outpath = os.path.join(OUTDIR, prestr+'hist_logt.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df(sectors)
    cdips_df = ccl.get_cdips_pub_catalog(ver=0.3)
    mdf = df.merge(cdips_df, how='left', right_on='source_id',
                   left_on='Gaia-ID')
    n_total = len(mdf)
    mdf = mdf[~pd.isnull(mdf['k13_logt'])]
    n_with_age = len(mdf)

    # aside for the abstract...
    sel = (mdf['ndet_tf1'] > 1) | (mdf['ndet_tf2'] > 1) | (mdf['ndet_tf3'] > 1)
    print('sectors {}: got {} unique_cluster_names'.format(
        repr(sectors), len(np.unique(mdf[sel]['unique_cluster_name']))
    ))

    f,ax = plt.subplots(figsize=(4,3))

    bins = np.arange(5.5,
                     10.5,
                     0.5)
    ax.hist(mdf['k13_logt'], bins=bins, cumulative=False, color='black',
            fill=False, linewidth=0.5)

    txtstr = '$N_{{\mathrm{{total}}}}$: {}'.format(n_total)
    txtstr += '\n$N_{{\mathrm{{with\ ages}}}}$: {}'.format(n_with_age)
    ax.text(
        0.03, 0.97,
        txtstr,
        ha='left', va='top',
        fontsize='x-small',
        transform=ax.transAxes
    )

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    ax.set_xlabel('log$_{10}$(age [years])')
    ax.set_ylabel('Number per bin')
    ax.set_yscale('log')
    ax.set_xlim([5.5, 10.5])

    f.tight_layout(pad=0.2)
    savefig(f, outpath)



def plot_cdf_cont(sectors, overwrite=0):

    prestr = 'sector{}_'.format(sectors[0]) if len(sectors)==1 else ''

    outpath = os.path.join(OUTDIR, prestr+'cdf_ticcont_isCTL.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df(sectors)
    sel = (df['ndet_tf1'] > 1) | (df['ndet_tf2'] > 1) | (df['ndet_tf3'] > 1)
    df = df[sel]

    f,ax = plt.subplots(figsize=(4,3))

    targetstr = 'TICCONT'
    bins = np.logspace(-3,2,11)
    ax.hist(df[targetstr], bins=bins, cumulative=True, color='black',
            fill=False, linewidth=0.5)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    ax.set_xlabel('nhbr flux / target flux (TICCONT)')
    ax.set_ylabel('cumulative number of LCs')
    ax.set_xscale('log')
    ax.set_yscale('log')

    f.tight_layout(pad=0.2)
    savefig(f, outpath)


def plot_hrd_scat(sectors, overwrite=0, close_subset=1):

    prestr = 'sector{}_'.format(sectors[0]) if len(sectors)==1 else ''

    if close_subset:
        outpath = os.path.join(OUTDIR, prestr+'hrd_scat_close_subset.png')
    else:
        outpath = os.path.join(OUTDIR, prestr+'hrd_scat_all_CDIPS_LCs.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df(sectors)
    sel = (df['ndet_tf1'] > 1) | (df['ndet_tf2'] > 1) | (df['ndet_tf3'] > 1)
    df = df[sel]

    # 2d SCATTER CASE
    if close_subset:
        df = df[df['Parallax[mas]'] > 0]
    plx_as = df['Parallax[mas]']/1000
    if close_subset:
        df = df[ 1/plx_as < 1000 ]

    f,ax = plt.subplots(figsize=(4,3))

    color = df['phot_bp_mean_mag']-df['phot_rp_mean_mag']
    M_omega = df['phot_g_mean_mag'] + 5*np.log10(df['Parallax[mas]']/1000) + 5

    ax.scatter(color,
               M_omega,
               rasterized=True, s=0.1, alpha=1, linewidths=0, zorder=5,
               color='black')

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    ax.set_xlabel('$G_{\mathrm{Bp}} - G_{\mathrm{Rp}}$')
    ax.set_ylabel('$M_\omega = G + 5\log_{10}(\omega_{\mathrm{as}}) + 5$')

    ax.set_ylim((12.5, -4.5))
    ax.set_xlim((-0.7, 4.3))

    if close_subset:
        txtstr= (
            '$\omega>0$, $1/\omega_{\mathrm{as}} < 1000$\n'+'{} stars'.
            format(len(df))
        )
    else:
        txtstr= (
            'no parallax cuts\n{} stars'.format(len(df))
        )
    ax.text(
        0.97, 0.97,
        txtstr,
        ha='right', va='top',
        fontsize='x-small',
        transform=ax.transAxes
    )

    f.tight_layout(pad=0.2)
    savefig(f, outpath)


def plot_pm_scat(sectors, overwrite=0, close_subset=0):

    prestr = 'sector{}_'.format(sectors[0]) if len(sectors)==1 else ''

    if close_subset:
        outpath = os.path.join(OUTDIR, prestr+'pm_scat_close_subset.png')
    else:
        outpath = os.path.join(OUTDIR, prestr+'pm_scat_all_CDIPS_LCs.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df(sectors)
    sel = (df['ndet_tf1'] > 1) | (df['ndet_tf2'] > 1) | (df['ndet_tf3'] > 1)
    df = df[sel]

    # 2d SCATTER CASE
    if close_subset:
        df = df[df['Parallax[mas]'] > 0]
    plx_as = df['Parallax[mas]']/1000
    if close_subset:
        df = df[ 1/plx_as < 1000 ]

    f,ax = plt.subplots(figsize=(4,3))

    xval = df['PM_RA[mas/yr]']
    yval = df['PM_Dec[mas/year]']
    ax.scatter(xval,
               yval,
               rasterized=True, s=0.1, alpha=1, linewidths=0, zorder=5,
               color='black')

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    ax.set_xlabel(r'$\mu_{{\alpha}} \cos\delta$ [mas/yr]')
    ax.set_ylabel('$\mu_{{\delta}}$ [mas/yr]')

    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])

    if close_subset:
        txtstr= (
            '$\omega>0$, $1/\omega_{\mathrm{as}} < 1000$\n'+'{} stars'.
            format(len(df))
        )
    else:
        txtstr= (
            'no parallax cuts\n{} stars'.format(len(df))
        )
    ax.text(
        0.97, 0.97,
        txtstr,
        ha='right', va='top',
        fontsize='x-small',
        transform=ax.transAxes
    )

    f.tight_layout(pad=0.2)
    savefig(f, outpath)


def plot_cluster_and_field_star_scatter(sectors=None, overwrite=0,
                                        cams=[1,2,3,4], ccds=[1,2,3,4],
                                        galacticcoords=False):
    """
    note: being kept separate from other stats collection step b/c here you
    need all LCs + CDIPS LCs, rather than only CDIPS LCs
    """

    csvpaths = []
    N_max = 100000

    prestr = 'sector{}_'.format(sectors[0]) if len(sectors)==1 else ''
    if galacticcoords:
        prestr = prestr + 'galacticcoords_'

    if cams==[1,2,3,4] and ccds==[1,2,3,4]:
        outpath = os.path.join(OUTDIR,
                               prestr+'cluster_field_star_positions.png')
    else:
        prestr = prestr + 'cam{}_ccd{}'.format(
            repr(cams).replace(' ','-').replace(',',''),
            repr(ccds).replace(' ','-').replace(',',''))
        outpath = os.path.join(OUTDIR,
                               prestr+'cluster_field_star_positions.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    for sector in sectors:
        for cam in cams:
          for ccd in ccds:

              # all lightcurves
              lcdir = (
                  '/nfs/phtess2/ar0/TESS/FFI/LC/FULL/s{}/ISP_{}-{}-15??/'.
                  format(str(sector).zfill(4), cam, ccd)
              )
              lcglob = '*_llc.fits'
              alllcpaths = glob(os.path.join(lcdir, lcglob))

              # CDIPS LCs
              cdipslcdir = (
                  '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/cam{}_ccd{}'.
                  format(sector, cam, ccd)
              )

              csvpath = os.path.join(
                  OUTDIR,'sector{}_cam{}_ccd{}_cluster_field_star_positions.csv'.
                  format(sector, cam, ccd)
              )

              csvpaths.append(csvpath)

              get_cluster_and_field_star_positions(alllcpaths, cdipslcdir, csvpath,
                                                   cam, ccd, sector,
                                                   N_desired=N_max)

    df = pd.concat((pd.read_csv(f) for f in csvpaths))
    if galacticcoords:
        coords = SkyCoord(nparr(df['ra'])*u.deg, nparr(df['dec'])*u.deg,
                          frame='icrs')
        glon = coords.galactic.l.value
        glat = coords.galactic.b.value
        df['glon'] = glon
        df['glat'] = glat

    if not galacticcoords:
        figsize = (4.5,4.5) if len(cams)==1 and len(ccds)==4 else (4.5,5.5)
    else:
        figsize = (12/1.5, 5/1.5)

    plot_cluster_and_field_star_positions(
        df, outpath, figsize, galacticcoords
    )


def get_cluster_and_field_star_positions(lcpaths, cdipslcdir, outpath,
                                         cam, ccd, sector,
                                         cdipsvnum=1, N_desired=200):

    if len(lcpaths) > N_desired:
        selpaths = np.random.choice(lcpaths, size=N_desired, replace=False)
    else:
        selpaths = lcpaths

    print('beginning get LC positions on {} LCs'.format(len(selpaths)))

    if os.path.exists(outpath):
        print('found {}, skip'.format(outpath))
        return

    gaiaids, xs, ys, ras, decs, iscdips = [], [], [], [], [], []
    N_paths = np.round(len(selpaths),-4)

    for ix, selpath in enumerate(selpaths):

        if (100 * ix/N_paths) % 1 == 0:
            print('{}: s{}cam{}ccd{} {:.0%} done'.format(
              datetime.utcnow().isoformat(), sector, cam, ccd, ix/N_paths)
            )

        hdul = fits.open(selpath)
        lcgaiaid = hdul[0].header['Gaia-ID']
        gaiaids.append(lcgaiaid)
        xs.append(hdul[0].header['XCC'])
        ys.append(hdul[0].header['YCC'])
        ras.append(hdul[0].header['RA[deg]'])
        decs.append(hdul[0].header['Dec[deg]'])

        cdipsname = (
            'hlsp_cdips_tess_ffi_'
            'gaiatwo{zsourceid}-{zsector}_'
            'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            zsourceid=str(lcgaiaid).zfill(22),
            zsector=str(sector).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )
        if os.path.exists(os.path.join(cdipslcdir,cdipsname)):
            iscdips.append(True)
        else:
            iscdips.append(False)

    xs, ys, ras, decs = nparr(xs), nparr(ys), nparr(ras), nparr(decs)
    gaiaids = nparr(gaiaids)
    iscdips = nparr(iscdips)
    cams = np.ones_like(ras)*cam
    ccds = np.ones_like(ras)*ccd
    sectors = np.ones_like(ras)*sector

    outdf = pd.DataFrame({'x':xs,'y':ys,'ra':ras,'dec':decs,
                          'gaiaid':gaiaids, 'iscdips': iscdips, 'cam':cam,
                          'ccd':ccd, 'sector':sectors})
    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def plot_cluster_and_field_star_positions(df, outpath, figsize, galacticcoords):
    """
    scatter of (ra,dec) for [subset of] stars with lightcurves.

    gray background points: field stars

    blue foreground points: cluster stars
    """

    f, ax = plt.subplots(figsize=figsize)

    iscdips = df['iscdips']

    if not galacticcoords:
        ax.scatter(df[~iscdips]['ra'], df[~iscdips]['dec'], c='k', alpha=0.5,
                   s=0.5, rasterized=True, linewidths=0, zorder=1)
        ax.scatter(df[iscdips]['ra'], df[iscdips]['dec'], c='C0', alpha=0.8,
                   s=0.5, rasterized=True, linewidths=0, zorder=2)
        ax.set_xlabel(r'Right ascension, $\alpha$ [deg]')
        ax.set_ylabel('Declination, $\delta$ [deg]')
    else:
        ax.scatter(df[~iscdips]['glon'], df[~iscdips]['glat'], c='k', alpha=0.5,
                   s=0.5, rasterized=True, linewidths=0, zorder=1)
        ax.scatter(df[iscdips]['glon'], df[iscdips]['glat'], c='C0', alpha=0.8,
                   s=0.5, rasterized=True, linewidths=0, zorder=2)
        ax.set_xlabel('Galactic longitude, $l$ [deg]')
        ax.set_ylabel('Galactic latitude, $b$ [deg]')

    #ax.set_title('black: $G_{\mathrm{Rp}}<13$ field. blue: $G_{\mathrm{Rp}}<16$ cluster.')

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    if not galacticcoords:
        # north up, east left. RA increases to the left (east)
        xlim = ax.get_xlim()
        ax.set_xlim((max(xlim),min(xlim)))

    f.savefig(outpath, bbox_inches='tight', dpi=450)
    print('made {}'.format(outpath))


def get_lc_stats(lcpaths, cdipslcdir, outpath, sector, cdipsvnum=1,
                 N_desired=200):
    """
    given lcpaths (not necessarily CDIPS LCs) get stats and assign whether it
    is a CDIPS LC or not.
    """

    if len(lcpaths) > N_desired:
        selpaths = np.random.choice(lcpaths, size=N_desired, replace=False)
    else:
        selpaths = lcpaths

    print('beginning get LC stats on {} LCs'.format(len(selpaths)))

    if os.path.exists(outpath):
        print('found {}, skip'.format(outpath))
        return

    hdrkeys = ['Gaia-ID','XCC','YCC','RA_OBJ','DEC_OBJ',
               'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag',
               'Parallax[mas]', 'PM_RA[mas/yr]', 'PM_Dec[mas/year]',
               'TESSMAG', 'TMAGPRED', 'TICCONT']
    skeys = ['stdev_tf1','stdev_tf2','stdev_tf3',
             'mad_tf1','mad_tf2','mad_tf3',
             'ndet_tf1','ndet_tf2','ndet_tf3',
             'stdev_sigclip_tf1','stdev_sigclip_tf2','stdev_sigclip_tf3',
             'mad_sigclip_tf1','mad_sigclip_tf2','mad_sigclip_tf3',
            ]

    hd = {}
    for hdrkey in hdrkeys:
        hd[hdrkey] = []
    for skey in skeys:
        hd[skey] = []
    hd['iscdips'] = []

    N_paths = np.round(len(selpaths),-4)

    for ix, selpath in enumerate(selpaths):

        if (100 * ix/N_paths) % 1 == 0:
            print('{}: {:.0%} done'.format(
              datetime.utcnow().isoformat(), ix/N_paths)
            )

        hdul = fits.open(selpath)

        # get all the position and target star info from the LC header
        for hdrkey in hdrkeys:
            hd[hdrkey].append(hdul[0].header[hdrkey])

        # get the LC statistics as a dictionary on sigclipped, orbit-edge
        # masked lightcurves
        d = get_lc_statistics(selpath, sigclip=4.0, tfalcrequired=True,
                              epdlcrequired=False, fitslcnottxt=True,
                              istessandmaskedges=True)

        for skey in skeys:
            hd[skey].append(d[skey])

        lcgaiaid = hdul[0].header['Gaia-ID']

        cdipsname = (
            'hlsp_cdips_tess_ffi_'
            'gaiatwo{zsourceid}-{zsector}_'
            'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            zsourceid=str(lcgaiaid).zfill(22),
            zsector=str(sector).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )
        if os.path.exists(os.path.join(cdipslcdir,cdipsname)):
            hd['iscdips'].append(True)
        else:
            hd['iscdips'].append(False)

    outdf = pd.DataFrame(hd)

    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def plot_rms_vs_mag(sectors, overwrite=0):

    cams = [1,2,3,4]
    ccds = [1,2,3,4]
    csvpaths = []
    N_max = 100000

    prestr = 'sector{}_'.format(sectors[0]) if len(sectors)==1 else ''

    outpath = os.path.join(OUTDIR, prestr+'rms_vs_mag.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    for sector in sectors:
      for cam in cams:
          for ccd in ccds:
              cdipslcdir = (
                  '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/cam{}_ccd{}'.
                  format(sector, cam, ccd)
              )
              lcglob = '*_llc.fits'
              cdipslcpaths = glob(os.path.join(cdipslcdir, lcglob))

              csvpath = os.path.join(
                  OUTDIR,'sector{}_cam{}_ccd{}_rms_vs_mag_data.csv'.
                  format(sector, cam, ccd)
              )

              csvpaths.append(csvpath)

              get_lc_stats(cdipslcpaths, cdipslcdir, csvpath, sector,
                           N_desired=N_max)

    df = pd.concat((pd.read_csv(f) for f in csvpaths))

    _plot_rms_vs_mag(df, outpath, overwrite=overwrite, yaxisval='RMS')


def _plot_rms_vs_mag(df, outpath, overwrite=0, yaxisval='RMS'):

    if yaxisval != 'RMS':
        raise AssertionError('so far only RMS implemented')
    # available:
    # hdrkeys = ['Gaia-ID','XCC','YCC','RA[deg]','Dec[deg]',
    #            'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag',
    #            'TESSMAG', 'TMAGPRED', 'TICCONT']
    # skeys = ['stdev_tf1','stdev_tf2','stdev_tf3',
    #          'mad_tf1','mad_tf2','mad_tf3',
    #          'stdev_sigclip_tf1','stdev_sigclip_tf2','stdev_sigclip_tf3',
    #          'mad_sigclip_tf1','mad_sigclip_tf2','mad_sigclip_tf3',
    #         ]

    mags = nparr(df['TESSMAG'])
    rms = nparr(
        [nparr(df['stdev_tf1']), nparr(df['stdev_tf2']),
         nparr(df['stdev_tf3'])]
    ).min(axis=0)

    N_pt = nparr(df['ndet_tf2'])
    N_TFA = 200 # number of template stars used in TFA

    # TFA overfits by default -- instead of standard deviation need to have
    # "N-1" be "N_pt - N_TFA - 1".
    sel = N_pt >= 202 # need LCs with points
    mags = mags[sel]
    rms = rms[sel]
    N_pt = N_pt[sel]
    corr = (N_pt - 1)/(N_pt -1 - N_TFA)
    rms = rms*np.sqrt(corr.astype(float))

    plt.close('all')
    fig, (a0, a1) = plt.subplots(nrows=2, ncols=1, sharex=True,
                               figsize=(4,5),
                               gridspec_kw= {'height_ratios':[3, 1.5]})

    a0.scatter(mags, rms, c='k', alpha=0.2, zorder=-5, s=0.5,
               rasterized=True, linewidths=0)

    if yaxisval=='RMS':
        Tmag = np.linspace(6, 16, num=200)

        # RA, dec. (90, -66) is southern ecliptic pole. these are "good
        # coords", but we aren't plotting sky bkgd anyway!
        fra, fdec = 120, 0  # sector 6 cam 1 center
        coords = np.array([fra*np.ones_like(Tmag), fdec*np.ones_like(Tmag)]).T
        out = tnm.noise_model(Tmag, coords=coords, exptime=1800)

        noise_star = out[2,:]
        noise_sky = out[3,:]
        noise_ro = out[4,:]
        noise_sys = out[5,:]
        noise_star_plus_ro = np.sqrt(noise_star**2 + noise_ro**2 + noise_sky**2
                                     + noise_sys**2)

        a0.plot(Tmag, noise_star_plus_ro, ls='-', zorder=-2, lw=1, color='C1',
                label='Model = photon + read + sky + floor')
        a0.plot(Tmag, noise_star, ls='--', zorder=-3, lw=1, color='gray',
                label='Photon')
        a0.plot(Tmag, noise_ro, ls='-.', zorder=-4, lw=1, color='gray',
                label='Read')
        a0.plot(Tmag, noise_sky, ls=':', zorder=-4, lw=1, color='gray',
                label='Unresolved stars (sky)')
        a0.plot(Tmag, noise_sys, ls='-', zorder=-4, lw=0.5, color='gray',
                label='Systematic floor')

    a1.plot(Tmag, noise_star_plus_ro/noise_star_plus_ro, ls='-', zorder=-2,
            lw=1, color='C1', label='Photon + read')

    coords = np.array([fra*np.ones_like(mags), fdec*np.ones_like(mags)]).T
    out = tnm.noise_model(mags, coords=coords, exptime=1800)
    noise_star = out[2,:]
    noise_sky = out[3,:]
    noise_ro = out[4,:]
    noise_sys = out[5,:]
    noise_star_plus_ro = np.sqrt(noise_star**2 + noise_ro**2 + noise_sky**2 +
                                 noise_sys**2)
    a1.scatter(mags, rms/noise_star_plus_ro, c='k', alpha=0.2, zorder=-5,
               s=0.5, rasterized=True, linewidths=0)

    a0.legend(loc='lower right', fontsize='xx-small')
    #a0.legend(loc='upper left', fontsize='xx-small')
    a0.set_yscale('log')
    a1.set_xlabel('TESS magnitude', labelpad=0.8)
    a0.set_ylabel('RMS [30 minutes]', labelpad=0.8)
    a1.set_ylabel('RMS / Model', labelpad=1)

    a0.set_ylim([1e-5, 1e-1])
    a1.set_ylim([0.5,10])
    a1.set_yscale('log')
    for a in (a0,a1):
        a.set_xlim([5.8,16.2])
        a.yaxis.set_ticks_position('both')
        a.xaxis.set_ticks_position('both')
        a.get_yaxis().set_tick_params(which='both', direction='in')
        a.get_xaxis().set_tick_params(which='both', direction='in')
        for tick in a.xaxis.get_major_ticks():
            tick.label.set_fontsize('small')
        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize('small')

    fig.tight_layout(h_pad=-0.3, pad=0.2)
    fig.savefig(outpath, dpi=400)
    print('%sZ: made plot: %s' % (datetime.utcnow().isoformat(), outpath))



if __name__ == "__main__":
    main()
