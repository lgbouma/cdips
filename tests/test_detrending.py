from cdips.lcproc import detrend as dtr

from glob import glob
import os, textwrap, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr
from datetime import datetime

def plot_detrending_from_tfa(time, tfatime, rawflux, tfaflux, flat_flux,
                             trend_flux, ap_index=2, obsd_midtimes=None,
                             returnfig=False, savpath=None):

    plt.close('all')
    nrows = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(18,8))

    axs = axs.flatten()

    apstr = 'AP{:d}'.format(ap_index)
    stagestrs = ( ['RM{:d}'.format(ap_index),
                   'TF{:d}'.format(ap_index),
                   'DTR{:d}'.format(ap_index)] )

    yvals = [rawflux,tfaflux,flat_flux]
    nums = list(range(len(yvals)))

    for ax, yval, txt, num in zip(axs, yvals, stagestrs, nums):

        if 'TF' in txt or 'DTR' in txt:
            ax.scatter(tfatime, yval, c='black', alpha=0.9, zorder=2, s=10,
                       rasterized=True, linewidths=0)
        elif 'BKGD' in txt or 'RM' in txt:
            ax.scatter(time, yval, c='black', alpha=0.9, zorder=2, s=10,
                       rasterized=True, linewidths=0)

        if 'TF' in txt and len(stagestrs)==3:
            ax.scatter(tfatime, trend_flux, c='red', alpha=0.9, zorder=1, s=5,
                       rasterized=True, linewidths=0)

        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='large')

    if not isinstance(obsd_midtimes, np.ndarray):
        for ax in axs:
            ylim = ax.get_ylim()
            ax.set_ylim((min(ylim), max(ylim)))

    axs[-1].set_xlabel('BJDTDB', fontsize='large')
    axs[-1].xaxis.get_offset_text().set_fontsize('large')

    # make the y label
    ax_hidden = fig.add_subplot(111, frameon=False)
    ax_hidden.tick_params(labelcolor='none', top=False, bottom=False,
                          left=False, right=False)

    axs[0].set_ylabel('raw flux IRM2', fontsize='large', labelpad=27)
    axs[1].set_ylabel('tfa flux TFA2', fontsize='large', labelpad=27)
    axs[2].set_ylabel('detrended flux DTR2', fontsize='large', labelpad=27)

    if not savpath:
        savpath = 'temp_{:s}.png'.format(apstr)

    fig.tight_layout(h_pad=0.)
    if returnfig:
        return fig
    else:
        fig.savefig(savpath, dpi=250, bbox_inches='tight')
        print('%sZ: made plot: %s' % (datetime.utcnow().isoformat(), savpath))


def plot_detrending_from_raw(time, tfatime, rawflux, tfaflux, flat_flux,
                             trend_flux, ap_index=2, obsd_midtimes=None,
                             returnfig=False, savpath=None):

    plt.close('all')
    nrows = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(18,8))

    axs = axs.flatten()

    apstr = 'AP{:d}'.format(ap_index)
    stagestrs = ( ['RM{:d}'.format(ap_index),
                   'DTR{:d}'.format(ap_index)] )

    yvals = [rawflux,flat_flux]
    nums = list(range(len(yvals)))

    for ax, yval, txt, num in zip(axs, yvals, stagestrs, nums):

        if 'TF' in txt or 'DTR' in txt:
            ax.scatter(tfatime, yval, c='black', alpha=0.9, zorder=2, s=10,
                       rasterized=True, linewidths=0)
        elif 'BKGD' in txt or 'RM' in txt:
            ax.scatter(time, yval, c='black', alpha=0.9, zorder=2, s=10,
                       rasterized=True, linewidths=0)

        if 'RM' in txt and len(stagestrs)==2:
            ax.scatter(tfatime, trend_flux, c='red', alpha=0.9, zorder=1, s=5,
                       rasterized=True, linewidths=0)

        ax.get_yaxis().set_tick_params(which='both', direction='in',
                                       labelsize='large')
        ax.get_xaxis().set_tick_params(which='both', direction='in',
                                       labelsize='large')

    if not isinstance(obsd_midtimes, np.ndarray):
        for ax in axs:
            ylim = ax.get_ylim()
            ax.set_ylim((min(ylim), max(ylim)))

    axs[-1].set_xlabel('BJDTDB', fontsize='large')
    axs[-1].xaxis.get_offset_text().set_fontsize('large')

    # make the y label
    ax_hidden = fig.add_subplot(111, frameon=False)
    ax_hidden.tick_params(labelcolor='none', top=False, bottom=False,
                          left=False, right=False)

    axs[0].set_ylabel('raw flux IRM2', fontsize='large', labelpad=27)
    axs[1].set_ylabel('detrended flux DTR2', fontsize='large', labelpad=27)

    if not savpath:
        savpath = 'temp_{:s}.png'.format(apstr)

    fig.tight_layout(h_pad=0.)
    if returnfig:
        return fig
    else:
        fig.savefig(savpath, dpi=250, bbox_inches='tight')
        print('%sZ: made plot: %s' % (datetime.utcnow().isoformat(), savpath))


def test_detrending(source_id=None):

    df = pd.read_csv('data/example_data_{}.csv'.format(source_id))

    outpng = '{}_detrend_test_from_tfa.png'.format(source_id)
    flat_flux, trend_flux = dtr.detrend_flux(
        nparr(df.tfatime), nparr(df.tfaflux), break_tolerance=0.5
    )
    plot_detrending_from_tfa(nparr(df.time), nparr(df.tfatime),
                             nparr(df.rawflux), nparr(df.tfaflux),
                             flat_flux, trend_flux, ap_index=2,
                             returnfig=False, savpath=outpng)

    outpng = '{}_detrend_test_from_raw.png'.format(source_id)
    flat_flux, trend_flux = dtr.detrend_flux(
        nparr(df.time), nparr(df.rawflux), break_tolerance=0.5
    )
    plot_detrending_from_raw(nparr(df.time), nparr(df.tfatime),
                             nparr(df.rawflux), nparr(df.tfaflux),
                             flat_flux, trend_flux, ap_index=2,
                             returnfig=False, savpath=outpng)



if __name__ == "__main__":
    test_detrending(source_id='5326491313765089792')
    test_detrending(source_id='5334408965769940608')
