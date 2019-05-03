from glob import glob
import os, textwrap
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from astrobase import periodbase, checkplot
from astrobase.checkplot.png import _make_phased_magseries_plot
from astrobase.lcmath import sigclip_magseries
from astrobase.lcfit.eclipses import gaussianeb_fit_magseries

from cdips.lcproc import mask_orbit_edges as moe

from astropy.io import fits
from astropy import units as u, constants as const

from scipy.interpolate import interp1d
from numpy import array as nparr

def _given_mag_get_flux(mag):

    mag_0, f_0 = 12, 1e4
    flux = f_0 * 10**( -0.4 * (mag - mag_0) )
    flux /= np.nanmedian(flux)

    return flux

def two_periodogram_checkplot(lc_sr, hdr, mask_orbit_edges=True,
                              fluxap='TFASR2', nworkers=32):

    time, mag = lc_sr['TMID_BJD'], lc_sr[fluxap]
    try:
        time, mag = moe.mask_orbit_start_and_end(time, mag)
    except AssertionError:
        # got more gaps than expected. ignore.
        pass

    flux = _given_mag_get_flux(mag)
    err = np.ones_like(flux)*1e-4

    spdm = periodbase.stellingwerf_pdm(time, flux, err, magsarefluxes=True,
                                       startp=0.1, endp=19, nworkers=nworkers)
    tlsp = periodbase.tls_parallel_pfind(time, flux, err, magsarefluxes=True,
                                         startp=0.1, endp=19, tlsoversample=7,
                                         tlsmintransits=2, sigclip=[50.,5.],
                                         nworkers=nworkers)

    objectinfo = {}
    keys = ['objectid','ra','decl','pmra','pmdecl','teff','gmag']
    hdrkeys = ['Gaia-ID', 'RA_OBJ', 'DEC_OBJ', 'PM_RA[mas/yr]',
               'PM_Dec[mas/year]', 'teff_val', 'phot_g_mean_mag']
    for k,hk in zip(keys,hdrkeys):
        if hk in hdr:
            objectinfo[k] = hdr[hk]
        else:
            objectinfo[k] = np.nan

    fig = checkplot.twolsp_checkplot_png(tlsp, spdm, time, flux, err,
                                         magsarefluxes=True,
                                         objectinfo=objectinfo,
                                         sigclip=[50.,5.], plotdpi=100,
                                         phasebin=3e-2, phasems=6.0,
                                         phasebinms=14.0, unphasedms=6.0,
                                         figsize=(30,24), returnfigure=True,
                                         circleoverlay=1.5*21, yticksize=20)

    return fig, tlsp, spdm



def plot_raw_tfa_bkgd(time, rawmag, tfamag, bkgdval, ap_index, savpath=None,
                      obsd_midtimes=None, xlabel='BJDTDB', customstr='',
                      tfatime=None, returnfig=True, is_tfasr=True,
                      figsize=(30,24)):
    """
    Plot 3 row, 1 column plot with rows of:
        * raw mags vs time
        * TFA mags vs time.
        * bkgd val vs time.

    args:
        time, rawmag, tfamag, tfamag (np.ndarray)

        ap_index (int): integer, e.g., "2" for aperture #2.

    kwargs:
        obsd_midtimes (np.ndarray): times for which to underplot lines, to show
        the ephemeris.

        customstr: string that goes on top right of plot, under a line that
        quotes the RMS

        tfatime: if passed, "time" is used to plot rawmag and bkgdval,
        "tfatime" is used to plot tfamag. Otherwise "time" is used for all of
        them.
    """

    if isinstance(tfatime,np.ndarray):
        try:
            tfatime, tfamag = moe.mask_orbit_start_and_end(tfatime, tfamag)
        except AssertionError:
            # got more gaps than expected. ignore.
            pass
    else:
        raise NotImplementedError
    try:
        _, rawmag = moe.mask_orbit_start_and_end(time, rawmag)
        time, bkgdval = moe.mask_orbit_start_and_end(time, bkgdval)
    except AssertionError:
        # got more gaps than expected. ignore.
        pass


    plt.close('all')
    nrows = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=figsize)

    axs = axs.flatten()

    apstr = 'AP{:d}'.format(ap_index)
    if is_tfasr:
        stagestrs = ( ['RM{:d}'.format(ap_index),
                       'TF{:d}SR'.format(ap_index),
                       'BKGD{:d}'.format(ap_index)] )
    else:
        stagestrs = ( ['RM{:d}'.format(ap_index),
                       'TF{:d}'.format(ap_index),
                       'BKGD{:d}'.format(ap_index)] )

    yvals = [rawmag,tfamag,bkgdval]
    nums = list(range(len(yvals)))

    for ax, yval, txt, num in zip(axs, yvals, stagestrs, nums):

        if isinstance(tfatime, np.ndarray) and 'TF' in txt:
            ax.scatter(tfatime, yval, c='black', alpha=0.9, zorder=2, s=50,
                       rasterized=True, linewidths=0)
        else:
            ax.scatter(time, yval, c='black', alpha=0.9, zorder=2, s=50,
                       rasterized=True, linewidths=0)

        if num in [0]:
            txt_x, txt_y = 0.99, 0.98
            mag = yval
            stdmmag = np.nanstd(mag)*1e3
            if stdmmag > 0.1:
                stattxt = '$\sigma$ = {:.1f} mmag{}'.format(stdmmag, customstr)
                ndigits = 2
            elif stdmmag > 0.01:
                stattxt = '$\sigma$ = {:.2f} mmag{}'.format(stdmmag, customstr)
                ndigits = 3
            else:
                stattxt = '$\sigma$ = {:.3f} mmag{}'.format(stdmmag, customstr)
                ndigits = 4
            _ = ax.text(txt_x, txt_y, stattxt, horizontalalignment='right',
                    verticalalignment='top', fontsize='medium', zorder=3,
                    transform=ax.transAxes)
        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='x-small')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='x-small')

        if isinstance(obsd_midtimes, np.ndarray):
            ylim = ax.get_ylim()
            ax.set_ylim((max(ylim), min(ylim)))
            ax.vlines(obsd_midtimes, max(ylim), min(ylim), color='orangered',
                      linestyle='--', zorder=1, lw=2, alpha=0.3)
            ax.set_ylim((max(ylim), min(ylim)))

    if not isinstance(obsd_midtimes, np.ndarray):
        for ax in axs:
            ylim = ax.get_ylim()
            ax.set_ylim((max(ylim), min(ylim)))

    axs[-1].set_xlabel(xlabel, fontsize='small')

    # make the y label
    ax_hidden = fig.add_subplot(111, frameon=False)
    ax_hidden.tick_params(labelcolor='none', top=False, bottom=False,
                          left=False, right=False)
    ax_hidden.set_ylabel('bkgd | TFASR | IRM', fontsize='small', labelpad=5)

    if not savpath:
        savpath = 'temp_{:s}.png'.format(apstr)

    fig.tight_layout(h_pad=0.)
    if returnfig:
        return fig
    else:
        fig.savefig(savpath, dpi=250, bbox_inches='tight')
        print('%sZ: made plot: %s' % (datetime.utcnow().isoformat(), savpath))


def scatter_increasing_ap_size(lc_sr, infodict=None, obsd_midtimes=None,
                               xlabel='BJDTDB', customstr='', figsize=(30,24),
                               returnfig=True):
    """
    Plot 3 row, 1 column plot with rows of:
        * TFASR ap 1 (smallest)
        * TFASR ap 2 (detection)
        * TFASR ap 3 (biggest).
    We use flux instead of mags, for easiest comparison of depths.

    Similar args/kwargs to plot_raw_tfa_bkgd
    """

    plt.close('all')
    nrows = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=figsize)

    axs = axs.flatten()

    stagestrs = ['TFASR1','TFASR2','TFASR3']

    time = lc_sr['TMID_BJD']
    yvals = [_given_mag_get_flux(lc_sr[i]) for i in stagestrs]

    try:
        masked_yvals = []
        times = []
        for yval in yvals:
            masked_yvals.append(moe.mask_orbit_start_and_end(time, yval)[1])
            times.append(moe.mask_orbit_start_and_end(time, yval)[0])
        yvals = masked_yvals
        time = times[0]
    except AssertionError:
        # got more gaps than expected. ignore.
        pass

    nums = list(range(len(yvals)))

    for ax, yval, txt, num in zip(axs, yvals, stagestrs, nums):

        ax.scatter(time, yval, c='black', alpha=0.9, zorder=2, s=50,
                   rasterized=True, linewidths=0)

        if num in [0]:
            txt_x, txt_y = 0.99, 0.98
            mag = yval
            stdmmag = np.nanstd(mag)*1e3
            if stdmmag > 0.1:
                stattxt = '$\sigma$ = {:.1f} mmag{}'.format(stdmmag, customstr)
                ndigits = 2
            elif stdmmag > 0.01:
                stattxt = '$\sigma$ = {:.2f} mmag{}'.format(stdmmag, customstr)
                ndigits = 3
            else:
                stattxt = '$\sigma$ = {:.3f} mmag{}'.format(stdmmag, customstr)
                ndigits = 4
            _ = ax.text(txt_x, txt_y, stattxt, horizontalalignment='right',
                    verticalalignment='top', fontsize='medium', zorder=3,
                    transform=ax.transAxes)

        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize=26)
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='xx-large')

        if isinstance(infodict, dict):
            # show transit depth in detection aperture (ap2). depth is saved as
            # 1-delta.
            ax.axhline(infodict['depth'], lw=2, alpha=0.3, color='C0')

        if isinstance(obsd_midtimes, np.ndarray):
            ylim = ax.get_ylim()
            ax.set_ylim((min(ylim), max(ylim)))
            ax.vlines(obsd_midtimes, min(ylim), max(ylim), color='orangered',
                      linestyle='--', zorder=1, lw=2, alpha=0.3)
            ax.set_ylim((min(ylim), max(ylim)))

    if not isinstance(obsd_midtimes, np.ndarray):
        for ax in axs:
            ylim = ax.get_ylim()
            ax.set_ylim((min(ylim), max(ylim)))

    axs[-1].set_xlabel(xlabel, fontsize='small')

    # make the y label
    ax_hidden = fig.add_subplot(111, frameon=False)
    ax_hidden.tick_params(labelcolor='none', top=False, bottom=False,
                          left=False, right=False)
    ax_hidden.set_ylabel('[flux] big ap | detection (medium) aperture | small ap',
                         fontsize='small', labelpad=5)

    fig.tight_layout(h_pad=-0.1)
    if returnfig:
        return fig
    else:
        raise NotImplementedError
        fig.savefig(savpath, dpi=250, bbox_inches='tight')
        print('%sZ: made plot: %s' % (datetime.utcnow().isoformat(), savpath))


def _get_full_infodict(tlsp, hdr, mdf):

    # odd_even_mismatch: stdevns btwn odd and even transit depths
    d = {
        'period':tlsp['tlsresult']['period'],
        't0':tlsp['tlsresult']['T0'],
        'duration':tlsp['tlsresult']['duration'],
        'depth':tlsp['tlsresult']['depth'],
        'rp_rs':tlsp['tlsresult']['rp_rs'], #Rp/Rstar, different bc LD
        'snr':tlsp['tlsresult']['snr'],
        'snr_pink_per_transit':tlsp['tlsresult']['snr_pink_per_transit'],
        'odd_even_mismatch':tlsp['tlsresult']['odd_even_mismatch'],
        'sde':tlsp['tlsresult']['SDE']
    }

    # dict from CDIPS star catalog with the following keys:
    #  ['cluster', 'dec', 'dist', 'ext_catalog_name', 'parallax',
    #   'phot_bp_mean_mag', 'phot_g_mean_mag', 'phot_rp_mean_mag', 'pmdec',
    #   'pmra', 'ra', 'reference', 'source_id']
    c = mdf.T.to_dict().popitem()[1]

    h = {
        'teff':hdr['teff_val'],
        'rstar':hdr['radius_val'],
        'AstExcNoiseSig':hdr['AstExcNoiseSig'] # N-sigma of astrometric detection
    }

    if h['rstar'] != 'NaN':
        rp = (d['rp_rs'] *  h['rstar']*u.Rsun).to(u.Rearth).value
        d['rp'] = '{:.2f}'.format(rp)
    else:
        d['rp'] = 'NaN'

    if h['teff'] != 'NaN':
        h['teff'] = '{:.0f}'.format(h['teff'])
    if h['rstar'] != 'NaN':
        h['rstar'] = '{:.2f}'.format(h['rstar'])

    # if you have Gaia Rstar, use that to estimate stellar mass, and the
    # circular transit duration timescale
    mamadf = pd.read_csv('../data/Mamajek_Rstar_Mstar_Teff_SpT.txt',
                         delim_whitespace=True)

    if h['rstar'] != 'NaN':
        fn = interp1d(nparr(mamadf['Rsun']), nparr(mamadf['Msun']),
                      kind='quadratic', bounds_error=False,
                      fill_value='extrapolate')
        mstar = fn(float(h['rstar']))
        rstar = float(h['rstar'])*u.Rsun

        a = _get_a_given_P_and_Mstar(d['period']*u.day, mstar*u.Msun)
        tdur_circ = (rstar*(d['period']*u.day)/(np.pi*a)).to(u.hr)

        h['mstar'] = '{:.2f}'.format(mstar)
        h['circduration'] = '{:.1f}'.format(tdur_circ)

    #FIXME TODO
    # if you're not given gaia radius, but do have teff, then estimate Rstar
    # using the relations worked out in TIC8.
    #
    # NOTE: you need the extinction to be subtracted for this ---
    # G = Gobs - A_G

    else:
        h['mstar'] = 'NaN'
        h['circduration'] = 'NaN'

    megad = {**d, **c, **h}

    return megad

def _get_a_given_P_and_Mstar(period, mstar):

    return (
        const.G * mstar / (4*np.pi*np.pi) * period**2
    )**(1/3.)


def transitcheckdetails(tfasrmag, tfatime, tlsp, mdf, hdr, suppdf,
                        obsd_midtimes=None, figsize=(30,24), returnfig=True,
                        sigclip=[50,5]):

    try:
        time, tfasrmag = moe.mask_orbit_start_and_end(tfatime, tfasrmag)
    except AssertionError:
        # got more gaps than expected. ignore.
        time, tfasrmag = tfatime, tfasrmag
    flux = _given_mag_get_flux(tfasrmag)

    stime, sflux, _ = sigclip_magseries(time, flux, np.ones_like(flux)*1e-4,
                                        magsarefluxes=True, sigclip=sigclip)

    d = _get_full_infodict(tlsp, hdr, mdf)

    #FIXME
    # [period (time), epoch (time), pdepth (mags), pduration (phase),
    # psdepthratio, secondaryphase]
    initebparams = [d['period'], d['t0'], 1-d['depth'],
                    d['duration']/d['period'], 0.2, 0.5 ]
    ebfitd = gaussianeb_fit_magseries(time, flux, np.ones_like(flux)*1e-4,
                                      initebparams, sigclip=None,
                                      plotfit=False, magsarefluxes=True,
                                      verbose=True)
    fitparams = ebfitd['finalparams']
    fitparamerrs = ebfitd['finalparamerrs']

    psdepthratio = fitparams[4]
    psdepthratioerr = fitparamserrs[4]

    #
    # if primary/secondary ratio is >~ 6, could be a planet. (Most extreme case
    # is probably KELT-9b, with delta_tra/delta_occ of ~= 10).
    # if primary/secondary ratio is very near 1, could be a planet, phased at 2x the
    # true orbital period.
    #
    # so it's really only in the primary/secondary range of ~1.2-7 that we can
    # be pretty certain it's an EB.
    #
    d['psdepthratio'] = psdepthratio
    d['psdepthratioerr'] = psdepthratioerr

    import IPython; IPython.embed() #FIXME check
    assert 0
    #FIXME


    ##########################################

    plt.close('all')
    fig = plt.figure(figsize=figsize)

    # ax0: flux v time, top left
    # ax1: text on right
    # ax2: primary transit
    # ax3: occultation
    # ax4: odd
    # ax5: even
    ax0 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (1, 1))
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax5 = plt.subplot2grid((3, 3), (2, 1))

    #
    # ax0: flux v time, top left
    #
    ax0.scatter(stime, sflux, c='black', alpha=0.9, zorder=2, s=50,
                rasterized=True, linewidths=0)
    ax0.set_xlabel('BJDTDB')
    ax0.set_ylabel('TFASR2')

    if isinstance(obsd_midtimes, np.ndarray):
        ylim = ax0.get_ylim()
        ax0.set_ylim((min(ylim), max(ylim)))
        ax0.vlines(obsd_midtimes, min(ylim), max(ylim), color='orangered',
                   linestyle='--', zorder=1, lw=2, alpha=0.3)
        ax0.set_ylim((min(ylim), max(ylim)))

    #
    # ax1: text on right
    #
    txt = (
    """
    P = {period:.3f} day
    $t_0$ = {t0:.3f} BJD
    $R_p$ = {rp:s} $R_\oplus$
    $R_p/R_\star$ = {rp_rs:.3f}
    $T_{{14}}/P$ = {tdur_by_period:.3f}
    $T_{{14}}$ = {duration:.2f} hr
    SNR = {snr:.1f}, SNRpink/tra = {snr_pink_per_transit:.1f}

    $\delta_{{odd}}$ vs $\delta_{{even}}$ = {odd_even_mismatch:.1f} $\sigma$
    $\delta_{{tra}}/\delta_{{occ}}$ = {psdepthratio:.2f} $\pm$ {psdepthratioerr:.2f}

    Star: DR2 {sourceid}
    $R_\star$ = {rstar:s} $R_\odot$, $M_\star$ = {mstar:s} $M_\odot$
    Teff = {teff:s} K
    RA = {ra:.3f}, DEC = {dec:.3f}
    G = {phot_g_mean_mag:.1f}, Rp = {phot_rp_mean_mag:.1f}, Bp = {phot_bp_mean_mag:.1f}
    pmRA = {pmra:.1f}, pmDEC = {pmdec:.1f}
    $\omega$ = {plx_mas:.2f} $\pm$ {plx_mas_err:.2f} mas
    d = 1/$\omega_{{as}}$ = {dist_pc:.0f} pc
    AstExc: {AstExcNoiseSig:.1f} $\sigma$
    $R_\star$+$M_\star$->$T_{{b0}}$: {circduration:s} hr

    Cluster: {cluster:s}
    Reference: {reference:s}
    Othername: {ext_catalog_name:s}
    xmatchdist: {xmatchdist:s}
    """
    )
    try:
        outstr = txt.format(
            period=d['period'],
            t0=d['t0'],
            rp=str(d['rp']),
            rp_rs=d['rp_rs'],
            tdur_by_period=d['duration']/d['period'],
            duration=d['duration']*24,
            circduration=d['circduration'],
            snr=d['snr'],
            snr_pink_per_transit=np.nanmean(d['snr_pink_per_transit']),
            odd_even_mismatch=d['odd_even_mismatch'],
            sourceid=hdr['Gaia-ID'],
            rstar=str(d['rstar']),
            mstar=str(d['mstar']),
            teff=str(d['teff']),
            ra=float(d['ra']),
            dec=float(d['dec']),
            psdepthratio=float(d['psdepthratio']),
            psdepthratioerr=float(d['psdepthratioerr']),
            phot_g_mean_mag=d['phot_g_mean_mag'],
            phot_rp_mean_mag=d['phot_rp_mean_mag'],
            phot_bp_mean_mag=d['phot_bp_mean_mag'],
            pmra=float(d['pmra']),
            pmdec=float(d['pmdec']),
            plx_mas=float(suppdf['Parallax[mas][6]']),
            plx_mas_err=float(suppdf['Parallax_error[mas][7]']),
            dist_pc=1/(1e-3 * float(hdr['Parallax[mas]'])),
            AstExcNoiseSig=d['AstExcNoiseSig'],
            cluster=d['cluster'],
            reference=d['reference'],
            ext_catalog_name=d['ext_catalog_name'],
            xmatchdist=','.join(
                ['{:.1e}'.format(float(l)) for l in str(d['dist']).split(',')]
            )
        )
    except Exception as e:
        outstr = 'got bug {}'.format(e)
        print(outstr)
        import IPython; IPython.embed()
        assert 0
        pass
        #print('error making outstr. FIX ME!')
        #print(e)
        #import IPython; IPython.embed()
        #assert 0

    txt_x, txt_y = 0.01, 0.99
    ax1.text(txt_x, txt_y, textwrap.dedent(outstr),
             ha='left', va='top', fontsize=40, zorder=2,
             transform=ax1.transAxes)
    ax1.set_axis_off()

    #
    # ax2: primary transit
    #
    phasebin = 6e-3
    minbinelems=2
    tdur_by_period=d['duration']/d['period']
    plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
    _make_phased_magseries_plot(ax2, 0, stime, sflux, np.ones_like(sflux)/1e4,
                                d['period'], d['t0'], True, True, phasebin,
                                minbinelems, plotxlim, 'tls',
                                xliminsetmode=False, magsarefluxes=True,
                                phasems=6.0, phasebinms=12.0, verbose=True)

    #
    # ax3: occultation
    #
    plotxlim=(-2.0*tdur_by_period+0.5,2.0*tdur_by_period+0.5)
    _make_phased_magseries_plot(ax3, 0, stime, sflux, np.ones_like(sflux)/1e4,
                                d['period'], d['t0'], True, True, phasebin,
                                minbinelems, plotxlim, 'tls',
                                xliminsetmode=False, magsarefluxes=True,
                                phasems=6.0, phasebinms=12.0, verbose=True)

    #
    # ax4: odd
    #
    if not isinstance(obsd_midtimes, np.ndarray):
        raise NotImplementedError

    even_midtimes = obsd_midtimes[::2]
    odd_midtimes = obsd_midtimes[1::2]

    delta_t = 0.245*d['period']
    even_windows = np.array((even_midtimes - delta_t, even_midtimes+delta_t))

    even_mask = np.zeros_like(stime).astype(bool)
    for even_window in even_windows.T:
        even_mask |= np.array(
            (stime > np.min(even_window)) & (stime < np.max(even_window))
        )
    odd_mask = ~even_mask

    plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
    _make_phased_magseries_plot(ax4, 0, stime[odd_mask], sflux[odd_mask],
                                np.ones_like(sflux[odd_mask])/1e4,
                                d['period'], d['t0'], True, True, phasebin,
                                minbinelems, plotxlim, 'tls',
                                xliminsetmode=False, magsarefluxes=True,
                                phasems=6.0, phasebinms=12.0, verbose=True)

    #
    # ax5: even
    #
    plotxlim=(-2.0*tdur_by_period,2.0*tdur_by_period)
    _make_phased_magseries_plot(ax5, 0, stime[even_mask], sflux[even_mask],
                                np.ones_like(sflux[even_mask])/1e4,
                                d['period'], d['t0'], True, True, phasebin,
                                minbinelems, plotxlim, 'tls',
                                xliminsetmode=False, magsarefluxes=True,
                                phasems=6.0, phasebinms=12.0, verbose=True)

    fig.tight_layout(h_pad=0)
    if returnfig:
        return fig, d
