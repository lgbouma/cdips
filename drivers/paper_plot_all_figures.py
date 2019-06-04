"""
execution environment: cdips, + pipe-trex .pth file in
/home/lbouma/miniconda3/envs/cdips/lib/python3.7/site-packages

python -u paper_plot_all_figures.py &> logs/paper_plot_all.log &
"""
from glob import glob
import datetime, os, pickle, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from numpy import array as nparr
from astropy.io import fits
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy import units as u

from astrobase import lcmath

from aperturephot import get_lc_statistics

from cdips.utils import tess_noise_model as tnm

OUTDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/paper_figures/'

def main():
    # fig N: RMS vs catalog T mag
    plot_rms_vs_mag(overwrite=0)

    # fig N: histogram (or CDF) of stellar magnitude (T mag)
    plot_cdf_T_mag(overwrite=0)

    # fig N: histogram (or CDF) of TICCONT. unfortunately this is only
    # calculated for CTL stars, so by definition it has limited use
    plot_cdf_cont(overwrite=0)

    # fig N: HRD for CDIPS stars.
    plot_hrd_scat(overwrite=0, close_subset=1)
    plot_hrd_scat(overwrite=0, close_subset=0)

    # fig N: pmRA and pmDEC scatter for CDIPS stars.
    plot_pm_scat(overwrite=0, close_subset=1)
    plot_pm_scat(overwrite=0, close_subset=0)

    # fig N: positions of field and cluster stars (currently just cam 1)
    plot_cluster_and_field_star_scatter(overwrite=0)

    #
    # fig N: wcs quality verification
    #
    plot_wcs_verification()

    pass

def savefig(fig, figpath):
    fig.savefig(figpath, dpi=450, bbox_inches='tight')
    print('{}: made {}'.format(datetime.utcnow().isoformat(), figpath))

def _get_rms_vs_mag_df():

    sectornum = 6
    cams = [1,2,3,4]
    ccds = [1,2,3,4]
    csvpaths = []

    for cam in cams:
        for ccd in ccds:

            csvpath = os.path.join(
                OUTDIR,'cam{}_ccd{}_rms_vs_mag_data.csv'.
                format(cam, ccd)
            )

            csvpaths.append(csvpath)

    df = pd.concat((pd.read_csv(f) for f in csvpaths))
    if len(df) == 0:
        raise AssertionError('need to run rms vs mag first!!')

    return df

def plot_cdf_T_mag(overwrite=0):

    outpath = os.path.join(OUTDIR, 'cdf_T_mag.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df()

    f,ax = plt.subplots(figsize=(4,3))

    magstr = 'TESSMAG'
    bins = np.arange(np.floor(np.min(df[magstr])),
                     np.ceil(np.max(df[magstr]))+1,
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
    ax.set_xlabel('TESS magnitude')
    ax.set_ylabel('cumulative number of LCs')
    ax.set_yscale('log')

    f.tight_layout(pad=0.2)
    savefig(f, outpath)


def plot_cdf_cont(overwrite=0):

    outpath = os.path.join(OUTDIR, 'cdf_ticcont_isCTL.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df()

    f,ax = plt.subplots(figsize=(4,3))

    targetstr = 'TICCONT'
    bins = np.logspace(-3,2,11)
    ax.hist(df[targetstr], bins=bins, cumulative=True)

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


def plot_hrd_scat(overwrite=0, close_subset=1):

    if close_subset:
        outpath = os.path.join(OUTDIR, 'hrd_scat_close_subset.png')
    else:
        outpath = os.path.join(OUTDIR, 'hrd_scat_all_CDIPS_LCs.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df()

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

    if close_subset:
        txtstr= (
            '$\omega>0$, $1/\omega_{\mathrm{as}} < 1000$, '+'{} stars'.
            format(len(df))
        )
    else:
        txtstr= (
            '{} stars (no parallax cuts)'.format(len(df))
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


def plot_pm_scat(overwrite=0, close_subset=0):

    if close_subset:
        outpath = os.path.join(OUTDIR, 'pm_scat_close_subset.png')
    else:
        outpath = os.path.join(OUTDIR, 'pm_scat_all_CDIPS_LCs.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df()

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
               rasterized=True, s=0.1, alpha=1, linewidths=0, zorder=5)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    ax.set_xlabel('pmRA[mas/yr]')
    ax.set_ylabel('pmDEC[mas/yr]')

    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])

    if close_subset:
        txtstr= (
            '$\omega>0$, $1/\omega_{\mathrm{as}} < 1000$, '+'{} stars'.
            format(len(df))
        )
    else:
        txtstr= (
            '{} stars (no parallax cuts)'.format(len(df))
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




def plot_cluster_and_field_star_scatter(overwrite=0):
    """
    note: being kept separate from other stats collection step b/c here you
    need all LCs + CDIPS LCs, rather than only CDIPS LCs
    """

    outpath = os.path.join(OUTDIR, 'cam1_cluster_field_star_positions.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    sectornum = 6
    cam = 1
    ccds = [1,2,3,4]
    csvpaths = []
    N_max = 100000

    for ccd in ccds:

        # all lightcurves
        lcdir = (
            '/nfs/phtess2/ar0/TESS/FFI/LC/FULL/s0006/ISP_{}-{}-15??/'.
            format(cam, ccd)
        )
        lcglob = '*_llc.fits'
        alllcpaths = glob(os.path.join(lcdir, lcglob))

        # CDIPS LCs
        cdipslcdir = (
            '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam{}_ccd{}'.
            format(cam, ccd)
        )

        csvpath = os.path.join(
            OUTDIR,'cam{}_ccd{}_cluster_field_star_positions.csv'.
            format(cam, ccd)
        )

        csvpaths.append(csvpath)

        get_cluster_and_field_star_positions(alllcpaths, cdipslcdir, csvpath,
                                             sectornum, N_desired=N_max)

    df = pd.concat((pd.read_csv(f) for f in csvpaths))

    plot_cluster_and_field_star_positions(
        df, outpath
    )


def get_cluster_and_field_star_positions(lcpaths, cdipslcdir, outpath,
                                         sectornum, cdipsvnum=1,
                                         N_desired=200):

    if len(lcpaths) > N_desired:
        selpaths = np.random.choice(lcpaths, size=N_desired, replace=False)
    else:
        selpaths = lcpaths

    print('beginning get LC positions on {} LCs'.format(len(selpaths)))

    if os.path.exists(outpath):
        print('found {}, skip'.format(outpath))
        return

    gaiaids, xs, ys, ras, decs, iscdips = [], [], [], [], [], []
    for selpath in selpaths:
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
            zsector=str(sectornum).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )
        if os.path.exists(os.path.join(cdipslcdir,cdipsname)):
            iscdips.append(True)
        else:
            iscdips.append(False)

    xs, ys, ras, decs = nparr(xs), nparr(ys), nparr(ras), nparr(decs)
    gaiaids = nparr(gaiaids)
    iscdips = nparr(iscdips)

    outdf = pd.DataFrame({'x':xs,'y':ys,'ra':ras,'dec':decs,
                          'gaiaid':gaiaids, 'iscdips': iscdips})
    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def plot_cluster_and_field_star_positions(df, outpath):
    """
    scatter of (ra,dec) for [subset of] stars with lightcurves.

    gray background points: field stars

    blue foreground points: cluster stars
    """

    f, ax = plt.subplots(figsize=(4.5,4.5))

    iscdips = df['iscdips']

    ax.scatter(df[~iscdips]['ra'], df[~iscdips]['dec'], c='k', alpha=0.5,
               s=0.5, rasterized=True, linewidths=0, zorder=1)
    ax.scatter(df[iscdips]['ra'], df[iscdips]['dec'], c='C0', alpha=0.8,
               s=0.5, rasterized=True, linewidths=0, zorder=2)

    ax.set_title('black: $G_{\mathrm{Rp}}<13$ field. blue: $G_{\mathrm{Rp}}<16$ cluster.')

    ax.set_xlabel('ra [deg]')
    ax.set_ylabel('dec [deg]')

    f.savefig(outpath, bbox_inches='tight', dpi=350)
    print('made {}'.format(outpath))


def get_lc_stats(lcpaths, cdipslcdir, outpath, sectornum, cdipsvnum=1,
                 N_desired=200):

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
             'stdev_sigclip_tf1','stdev_sigclip_tf2','stdev_sigclip_tf3',
             'mad_sigclip_tf1','mad_sigclip_tf2','mad_sigclip_tf3',
            ]

    hd = {}
    for hdrkey in hdrkeys:
        hd[hdrkey] = []
    for skey in skeys:
        hd[skey] = []
    hd['iscdips'] = []

    for selpath in selpaths:
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
            zsector=str(sectornum).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )
        if os.path.exists(os.path.join(cdipslcdir,cdipsname)):
            hd['iscdips'].append(True)
        else:
            hd['iscdips'].append(False)

    outdf = pd.DataFrame(hd)

    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def plot_rms_vs_mag(overwrite=0):

    outpath = os.path.join(OUTDIR, 'rms_vs_mag.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    sectornum = 6
    cams = [1,2,3,4]
    ccds = [1,2,3,4]
    csvpaths = []
    N_max = 100000

    for cam in cams:
        for ccd in ccds:
            cdipslcdir = (
                '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam{}_ccd{}'.
                format(cam, ccd)
            )
            lcglob = '*_llc.fits'
            alllcpaths = glob(os.path.join(cdipslcdir, lcglob))

            csvpath = os.path.join(
                OUTDIR,'cam{}_ccd{}_rms_vs_mag_data.csv'.
                format(cam, ccd)
            )

            csvpaths.append(csvpath)

            get_lc_stats(alllcpaths, cdipslcdir, csvpath, sectornum,
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
        coords = np.array([90*np.ones_like(Tmag), -66*np.ones_like(Tmag)]).T
        out = tnm.noise_model(Tmag, coords=coords, exptime=1800)

        noise_star = out[2,:]
        noise_ro = out[4,:]
        noise_star_plus_ro = np.sqrt(noise_star**2 + noise_ro**2)

        a0.plot(Tmag, noise_star_plus_ro, ls='-', zorder=-2, lw=1, color='C1',
                label='Photon + read')
        a0.plot(Tmag, noise_star, ls='--', zorder=-3, lw=1, color='C3',
                label='Photon')
        a0.plot(Tmag, noise_ro, ls='--', zorder=-4, lw=1, color='C4',
                label='Read')

    a1.plot(Tmag, noise_star_plus_ro/noise_star_plus_ro, ls='-', zorder=-2,
            lw=1, color='C1', label='Photon + read')

    coords = np.array([90*np.ones_like(mags), -66*np.ones_like(mags)]).T
    out = tnm.noise_model(mags, coords=coords, exptime=1800)
    noise_star = out[2,:]
    noise_ro = out[4,:]
    noise_star_plus_ro = np.sqrt(noise_star**2 + noise_ro**2)
    a1.scatter(mags, rms/noise_star_plus_ro, c='k', alpha=0.2, zorder=-5,
               s=0.5, rasterized=True, linewidths=0)

    a0.legend(loc='upper left', fontsize='xx-small')
    a0.set_yscale('log')
    a1.set_xlabel('TESS magnitude', labelpad=0.8)
    a0.set_ylabel('RMS [30 minutes]', labelpad=0.8)
    a1.set_ylabel('RMS / (Photon+Read)', labelpad=1)

    a0.set_ylim([1e-5, 1e-1])
    a1.set_ylim([-0.05, 2.05])
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



def plot_wcs_verification():
    pass


if __name__ == "__main__":
    main()
