"""
ported from pipe-trex/wcsqualityassurance.py for CDIPS paper plots.
assumption is that *.matched files are generated when running the actual
reduction. (these are grmatch output of projected-vs-measured centroid match).
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from numpy import array as nparr

from astropy.io import fits
from astropy import wcs

from mpl_toolkits.axes_grid1 import make_axes_locatable

from cdips.plotting import savefig

def main(
    fitsfile='proj1500-s0006-cam1-ccd1-combinedphotref-onenight.fits',
    refbasedir='/nfs/phtess2/ar0/TESS/FFI/BASE/reference-frames/',
    matchedinpath='proj1500-s0006-cam1-ccd1-combinedphotref-onenight.matched',
    isspocwcs=True,
    outdir=None
):

    matchedpath = os.path.join(refbasedir, matchedinpath)
    fitsfile = os.path.join(refbasedir, fitsfile)

    df = pd.read_csv(
        matchedpath,
        delim_whitespace=True,
        names=('id,ra,dec,xi,eta,G,Rp,Bp,plx,pmra,pmdec,varflag,x_proj,y_proj,'+
               'Ident,x_meas,y_meas,Bg,Amp,S,D,K,Flux,S/N').split(',')
    )

    #
    # projected and measured positions seem to systematically differ by 0.5
    # pixels in row and column position. NOTE: ensure this is OK.
    #
    if isspocwcs:

        df['x_proj'] -= 0.5
        df['y_proj'] -= 0.5

        #
        # SPOC rows/cols were trimmed, which further shifts WCS
        #
        hdulist = fits.open(fitsfile)
        hdr = hdulist[0].header

        df['x_proj'] -= (hdr['SCCSA']-1)
        df['y_proj'] -= (hdr['SCIROWS']-1)

        hdulist.close()

    else:
        df['x_proj'] -= 0.5
        df['y_proj'] -= 0.5

    df['sep'] = sep(df['x_meas'], df['y_meas'], df['x_proj'], df['y_proj'])

    #
    # trim out the saturated stars!! name-dependent b/c this is bad general
    # practice -- but OK for this subset where i verified that these outliers
    # are the saturated ~0.1%
    #
    if os.path.basename(fitsfile)=='proj1500-s0006-cam1-ccd1-combinedphotref-onenight.fits':
        df = df[df['sep'] < 4]
        print('WRN!: omitting saturated outlier stars')

    #
    # make plots
    #
    plt.close('all')

    pre = os.path.splitext(os.path.basename(matchedpath))[0]
    if isspocwcs:
        pre = pre+'_spocwcs'
    outpath = os.path.join(outdir, '{}_sep_hist.png'.format(pre))
    plot_sep_hist(df, outpath)
    print('made {}'.format(outpath))

    outpath = os.path.join(outdir, '{}_quiver_meas_proj_sep.png'.format(pre))
    plot_quiver_meas_proj_sep(df, outpath)
    print('made {}'.format(outpath))

    # # not very useful plot given quiver, but leave it in
    # outpath = os.path.join(outdir, '{}_scatter_x_y_sep.png'.format(pre))
    # plot_scatter_x_y_sep(df, outpath)


def sep(x0,y0,x1,y1):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)


def plot_scatter_x_y_sep(df, outpath):

    f, ax = plt.subplots(figsize=(4,4))

    norm = mpl.colors.Normalize(vmin=-2.,vmax=1.)

    cax = ax.scatter(df['x_meas'], df['y_meas'], c=np.log10(df['sep']),
                     cmap='viridis', s=1, rasterized=True, linewidths=0,
                     zorder=1, norm=norm)

    ax.set_xlabel('Column index [px]')
    ax.set_ylabel('Row index [px]')

    cbar = f.colorbar(cax, extend='both')
    cbar.set_label(r'$|\vec{{x}}_{{\mathrm{{measured}}}} - \vec{{x}}_{{\mathrm{{projected}}}}|$ [pixels]')

    savefig(f, outpath)


def plot_sep_hist(df, outpath):

    f, ax = plt.subplots(figsize=(4,3))

    weights = np.ones_like(nparr(df['sep']))/float(len(df))

    ax.hist(df['sep'], bins=np.logspace(-2, 1, 19), weights=weights,
            color='black', fill=False, linewidth=0.5)

    ax.text(
        0.98, 0.98,
        'mean={:.3f}px\nstdev={:.3f}px\nmedian={:.3f}px\n90$^{{\mathrm{{th}}}}$pctile={:.3f}px'.
        format(df['sep'].mean(),df['sep'].std(),
               df['sep'].median(),df['sep'].quantile(q=0.9)),
        va='top', ha='right',
        transform=ax.transAxes
    )
    ax.set_xscale('log')
    ax.set_xlabel(r'$|\vec{{x}}_{{\mathrm{{measured}}}} - \vec{{x}}_{{\mathrm{{projected}}}}|$ [pixels]')
    #ax.set_xlabel('Measured - projected centroid separation [pixels]')
    ax.set_ylabel('Relative fraction')

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    savefig(f, outpath)



def plot_quiver_meas_proj_sep(df, outpath):

    f, ax = plt.subplots(figsize=(4,4))

    norm = mpl.colors.Normalize(vmin=-2.,vmax=1.)

    cax = ax.quiver(df['x_meas'], df['y_meas'],
                    df['x_proj']-df['x_meas'],
                    df['y_proj']-df['y_meas'],
                    np.log10(df['sep']),
                    angles='xy',
                    scale_units='xy',
                    scale=0.02,#1,
                    cmap='viridis',
                    norm=norm)

    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)

    cbar = f.colorbar(cax, cax=cax1, extend='both')
    cbar.set_label(r'$|\vec{{x}}_{{\mathrm{{measured}}}} - \vec{{x}}_{{\mathrm{{projected}}}}|$ [pixels]')

    ax.set_xlabel('Column index [px]')
    ax.set_ylabel('Row index [px]')

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    savefig(f, outpath)
