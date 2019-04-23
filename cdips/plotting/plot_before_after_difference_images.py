import os, pickle, re, subprocess, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from datetime import datetime
from numpy import array as nparr
from astropy.io import fits

def read_fits(fits_file,ext=0):
    '''
    Shortcut function to get the header and data from a fits file and a given
    extension.

    '''

    hdulist = fits.open(fits_file)
    img_header = hdulist[ext].header
    img_data = hdulist[ext].data
    hdulist.close()

    return img_data, img_header


def plot_before_after_difference_images(subimgfile, calfile, outdir, trim=False):

    trimstr = '' if not trim else 'TRIM_'
    outpath = ( os.path.join( outdir, ('before_after_'+trimstr+
        os.path.basename(subimgfile).replace('.fits','.png'))))


    # this would be 100000x better as an actual python package. (todo)
    sub_img, _ = read_fits(subimgfile)
    cal_img, _ = read_fits(calfile)

    if trim:
        xlow, xhigh = 400, 1000
        ylow, yhigh = 0, 600
        sub_img = sub_img[ylow:yhigh,xlow:xhigh]
        cal_img = cal_img[ylow:yhigh,xlow:xhigh]

    plt.close('all')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(6,4))

    # calibrated image
    vmin, vmax = 10, int(1e3)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    cset1 = axs[0].imshow(cal_img, cmap='binary_r', vmin=vmin, vmax=vmax,
                          norm=norm)


    # difference image
    diff_vmin, diff_vmax = -1000, 1000
    diffnorm = colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=diff_vmin,
                                 vmax=diff_vmax)

    toplen = 57
    top = cm.get_cmap('Oranges_r', toplen)
    bottom = cm.get_cmap('Blues', toplen)
    newcolors = np.vstack((top(np.linspace(0, 1, toplen)),
                           np.zeros(((256-2*toplen),4)),
                           bottom(np.linspace(0, 1, toplen))))
    newcmp = ListedColormap(newcolors, name='lgb_cmap')

    cset2 = axs[1].imshow(sub_img, cmap=newcmp, vmin=diff_vmin,
                          vmax=diff_vmax, norm=diffnorm)
    # looked pretty good
    #cset2 = axs[1].imshow(sub_img, cmap='RdBu_r', vmin=diff_vmin,
    #                      vmax=diff_vmax, norm=diffnorm)

    # tweaking
    for ax in axs.flatten():
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

    # colorbars
    divider0 = make_axes_locatable(axs[0])
    divider1 = make_axes_locatable(axs[1])
    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cb1 = fig.colorbar(cset1, ax=axs[0], cax=cax0, extend='both')
    cb2 = fig.colorbar(cset2, ax=axs[1], cax=cax1, extend='both')
    cb2.set_ticks([-1e3,-1e2,-1e1,0,1e1,1e2,1e3])
    cb2.set_ticklabels(['-$10^3$','-$10^2$','-$10^1$','0',
                        '$10^1$','$10^2$','$10^3$'])
    for cb in [cb1, cb2]:
        cb.ax.tick_params(direction='in')
        cb.ax.tick_params(labelsize='small')

    fig.tight_layout(h_pad=0.1, w_pad=0.1, pad=-1)

    fig.savefig(outpath, bbox_inches='tight', dpi=400)
    print('{}: made {}'.format(datetime.utcnow().isoformat(), outpath))


if __name__ == "__main__":

    datadir = '../data/subtracted_demo_images/projid_1378_cam4_ccd3/'
    subimgfile = os.path.join(
        datadir, 'rsub-d2f9343c-tess2018230145941-s0001-4-3-0120_cal_img_bkgdsub-xtrns.fits')
    calfile = os.path.join(
        datadir, 'tess2018230145941-s0001-4-3-0120_cal_img.fits')

    plot_before_after_difference_images(subimgfile, calfile, datadir, trim=True)
    plot_before_after_difference_images(subimgfile, calfile, datadir)
