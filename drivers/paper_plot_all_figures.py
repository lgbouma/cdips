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

OUTDIR = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/paper_figures/'

def main():
    #
    # fig N: positions of field and cluster stars
    #
    plot_cluster_and_field_star_scatter(overwrite=0)

    #
    # fig N: RMS vs catalog T mag
    #
    plot_rms_vs_mag(overwrite=1)

    #
    # fig N: wcs quality verification
    #
    plot_wcs_verification()

    pass


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

    hdrkeys = ['Gaia-ID','XCC','YCC','RA[deg]','Dec[deg]',
               'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag',
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
                              epdlcrequired=True, fitslcnottxt=True,
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


def _plot_rms_vs_mag(df, outpath, overwrite=0, yaxisval='RMS'):

    df = _get_rms_vs_mag_data(cam=1)
    import IPython; IPython.embed()
    assert 0
    mags = nparr(df['Tmag'])
    rms = nparr(df['rms_tfa2'])
    #FIXME might want to do the minimum??

    plt.close('all')
    fig, ax = plt.subplots(figsize=(4,3))

    ax.scatter(mags, rms, c='k', alpha=0.12, zorder=-5, s=0.5,
               rasterized=True, linewidths=0)

    if yaxisval=='RMS':
        Tmag = np.linspace(6, 16, num=200)

        # RA, dec. (90, -66) is southern ecliptic pole. these are "good
        # coords", but we don't care!
        coords = np.array([90*np.ones_like(Tmag), -66*np.ones_like(Tmag)]).T
        out = noise_model(Tmag, coords=coords, exptime=1800)

        noise_star = out[2,:]
        noise_ro = out[4,:]
        noise_star_plus_ro = np.sqrt(noise_star**2 + noise_ro**2)

        ax.plot(Tmag, noise_star_plus_ro, ls='-', zorder=-2, lw=1, color='C1',
                label='Photon + read')
        ax.plot(Tmag, noise_star, ls='--', zorder=-3, lw=1, color='C3',
                label='Photon')
        ax.plot(Tmag, noise_ro, ls='--', zorder=-4, lw=1, color='C4',
                label='Read')

    ax.legend(loc='upper left', fontsize='xx-small')
    ax.set_yscale('log')
    xlabel = 'Instrument magnitude'
    if catmagstr=='cat_mag':
        xlabel = 'TESS magnitude' # NOTE: this is a lie. but a white one -- we checked, they're the same!! just cross-matching is insanely annoying 'Gaia Rp magnitude'
    if catmagstr=='Tmag':
        xlabel = 'TESS magnitude'
    ax.set_xlabel(xlabel, labelpad=0.8)
    ax.set_ylabel('RMS (30 minutes)', labelpad=0.8)
    ax.set_xlim(percentiles_xlim)
    ax.set_ylim(percentiles_ylim)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')

    if catmagstr=='Tmag':
        savname = os.path.join(outdir,'{:s}_vs_TESS_mag_{:s}.png'.
            format(yaxisval, apstr.upper()))
    fig.tight_layout(pad=0.2)
    fig.savefig(savname, dpi=400)
    print('%sZ: made plot: %s' % (datetime.utcnow().isoformat(), savname))


def plot_rms_vs_mag(overwrite=0):

    outpath = os.path.join(OUTDIR, 'rms_vs_mag.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    sectornum = 6
    cam = 1
    ccds = [1] #FIXME add ccds
    csvpaths = []
    N_max = 100

    for ccd in ccds:

        # all lightcurves
        lcdir = (
            '/nfs/phtess2/ar0/TESS/FFI/LC/FULL/s0006/ISP_{}-{}-15??/'.
            format(cam, ccd)
        )
        lcglob = '*_llc.fits'
        alllcpaths = glob(os.path.join(lcdir, lcglob))

        cdipslcdir = (
            '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6/cam{}_ccd{}'.
            format(cam, ccd)
        )

        csvpath = os.path.join(
            OUTDIR,'cam{}_ccd{}_rms_vs_mag_data.csv'.
            format(cam, ccd)
        )

        csvpaths.append(csvpath)

        get_lc_stats(alllcpaths, cdipslcdir, csvpath, sectornum,
                     N_desired=N_max)

    df = pd.concat((pd.read_csv(f) for f in csvpaths))

    _plot_rms_vs_mag(df, outpath, overwrite=overwrite, title='s0006,cam1')
    # FIXME FIXME FIXME FIXME may30,2019 - left off here



def plot_cluster_and_field_star_scatter(overwrite=0):

    outpath = os.path.join(OUTDIR, 'cam1_cluster_field_star_positions.png')
    if os.path.exists(outpath) and not overwrite:
        print('found {} and not overwrite; return'.format(outpath))
        return

    sectornum = 6
    cam = 1
    ccds = [1] #FIXME add ccds
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

    plot_cluster_and_field_star_positions(df, outpath, title='s0006,cam1')


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


def plot_cluster_and_field_star_positions(df, outpath, title='s0006,cam1'):
    """
    scatter of (ra,dec) for [subset of] stars with lightcurves.

    gray background points: field stars

    blue foreground points: cluster stars
    """

    f, ax = plt.subplots(figsize=(4,4))

    iscdips = df['iscdips']

    ax.scatter(df[~iscdips]['ra'], df[~iscdips]['dec'], c='k', alpha=0.5,
               s=0.5, rasterized=True, linewidths=0, zorder=1)
    ax.scatter(df[iscdips]['ra'], df[iscdips]['dec'], c='C0', alpha=0.8,
               s=0.5, rasterized=True, linewidths=0, zorder=2)

    ax.set_title(title)

    ax.set_xlabel('ra [deg]')
    ax.set_ylabel('dec [deg]')

    f.savefig(outpath, bbox_inches='tight', dpi=350)
    print('made {}'.format(outpath))


def plot_wcs_verification():
    pass


if __name__ == "__main__":
    main()
