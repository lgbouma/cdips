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

from cdips.plotting import plot_star_catalog as psc
from cdips.plotting import plot_catalog_to_gaia_match_statistics as xms

OUTDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/paper_figures/'
CLUSTERDATADIR = '/home/lbouma/proj/cdips/data/cluster_data'

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
    bigdf['gaia_dr2_match_id'].to_csv(sourcepath,index=False)

    targetpath = os.path.join(CLUSTERDATADIR,
                              'MWSC_Gaia_matched_Gaiainfo.csv')

    if not os.path.exists(targetpath):
        raise RuntimeError('run the gaia2read on mwsc the sourcepath list, '
                           'manually')
        gaia2readcmd = "gaia2read --header --extra --idfile MWSC_Gaia_matched_concatenated_sources.csv > MWSC_Gaia_matched_Gaiainfo.csv"

    # merge bigdf against relevant columns of gaia dr2
    gdf = pd.read_csv(targetpath, delim_whitespace=True)

    import IPython; IPython.embed()
    assert 0


    outpath = os.path.join(
        OUTDIR,'catalog_to_gaia_match_statistics_MWSC.png'
    )
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; skip'.format(outpath))
    else:
        xms.plot_catalog_to_gaia_match_statistics(df, outpath, isD14=False)

    ##########################################
    # Dias 2014 catalog
    d14_df = pd.read_csv(os.path.join(
        CLUSTERDATADIR,'Dias14_seplt5arcsec_Gdifflt2.csv'))

    ##########
    sourcepath = os.path.join(CLUSTERDATADIR,
                              'Dias14_seplt5arcsec_Gdifflt2_sources.csv')
    d14_df['source_id'].to_csv(sourcepath,index=False)

    targetpath = os.path.join(CLUSTERDATADIR,
                              'Dias14_seplt5arcsec_Gdifflt2_Gaiainfo.csv')

    if not os.path.exists(targetpath):
        raise RuntimeError('run the gaia2read on the d14 sourcepath list, '
                           'manually')
        gaia2readcmd = "gaia2read --header --extra --idfile Dias14_seplt5arcsec_Gdifflt2_sources.csv > Dias14_seplt5arcsec_Gdifflt2_Gaiainfo.csv"

    # merge bigdf against relevant columns of gaia dr2
    gdf = pd.read_csv(targetpath, delim_whitespace=True)
    ##########


    outpath = os.path.join(
        OUTDIR,'catalog_to_gaia_match_statistics_Dias14.png'
    )
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; skip'.format(outpath))
    else:
        xms.plot_catalog_to_gaia_match_statistics(d14_df, outpath, isD14=True)

def plot_target_star_cumulative_counts(OC_MG_CAT_ver=0.3, overwrite=1):

    catalogpath = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/OC_MG_FINAL_GaiaRp_lt_16_v{}.csv'.
        format(OC_MG_CAT_ver)
    )
    df = pd.read_csv(catalogpath, sep=';')

    outpath = os.path.join(OUTDIR, 'target_star_cumulative_counts.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    psc.star_catalog_mag_histogram(df, 'phot_rp_mean_mag', savpath=outpath)




def main():

    plot_catalog_to_gaia_match_statistics(overwrite=1)

    assert 0
    # fig N: wcs quality verification
    plot_wcs_verification()

    #FIXME 
    #FIXME 
    #FIXME 
    #FIXME 

    sectors = [6,7]

    # fig N: RMS vs catalog T mag
    plot_rms_vs_mag(sectors, overwrite=1)

    # fig N: histogram (or CDF) of stellar magnitude (T mag)
    plot_cdf_T_mag(sectors, overwrite=1)

    # fig N: histogram (or CDF) of TICCONT. unfortunately this is only
    # calculated for CTL stars, so by definition it has limited use
    plot_cdf_cont(sectors, overwrite=1)

    # fig N: HRD for CDIPS stars.
    plot_hrd_scat(sectors, overwrite=1, close_subset=1)
    plot_hrd_scat(sectors, overwrite=1, close_subset=0)

    # fig N: pmRA and pmDEC scatter for CDIPS stars.
    plot_pm_scat(sectors, overwrite=1, close_subset=1)
    plot_pm_scat(sectors, overwrite=1, close_subset=0)

    # fig N: cumulative counts of CDIPS target stars.
    plot_target_star_cumulative_counts(OC_MG_CAT_ver=0.3, overwrite=1)

    # fig N: positions of field and cluster stars (currently all cams)
    plot_cluster_and_field_star_scatter(sectors, overwrite=0)


def savefig(fig, figpath):
    fig.savefig(figpath, dpi=450, bbox_inches='tight')
    print('{}: made {}'.format(datetime.utcnow().isoformat(), figpath))


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
    ax.set_ylabel('cumulative number of LCs')
    ax.set_yscale('log')

    f.tight_layout(pad=0.2)
    savefig(f, outpath)


def plot_cdf_cont(sectors, overwrite=0):

    prestr = 'sector{}_'.format(sectors[0]) if len(sectors)==1 else ''

    outpath = os.path.join(OUTDIR, prestr+'cdf_ticcont_isCTL.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    df = _get_rms_vs_mag_df(sectors)

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
    ax.set_xlabel('$G_{\mathrm{BP}} - G_{\mathrm{RP}}$')
    ax.set_ylabel('$M_\omega = G + 5\log_{10}(\omega_{\mathrm{as}}) + 5$')

    ax.set_ylim((12.5, -4.5))
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


def plot_cluster_and_field_star_scatter(sectors, overwrite=0):
    """
    note: being kept separate from other stats collection step b/c here you
    need all LCs + CDIPS LCs, rather than only CDIPS LCs
    """

    cams = [1,2,3,4]
    ccds = [1,2,3,4]
    csvpaths = []
    N_max = 100000

    prestr = 'sector{}_'.format(sectors[0]) if len(sectors)==1 else ''

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

    plot_cluster_and_field_star_positions(
        df, outpath
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


def plot_wcs_verification():
    pass


if __name__ == "__main__":
    main()
