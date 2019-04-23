"""
nsub and rsub should have the same photometric performance. (because there
don't seem to be any obviously good arguments for why one should be better than
the other.)

nsub means:
D_xy = I_xy - Mxy,
M_xy = (R \otimes K)_xy, s.t.    chi^2 = \sum (I_xy - M_xy) minimized.

rsub means:
D_xy = R_xy - Mxy,
M_xy = (I \otimes K)_xy, s.t.    chi^2 = \sum (R_xy - M_xy) minimized.

In nsub, you convolve the high SNR median-stacked frame to match a lower SNR
image frame. (Usually, this is a blurring).

In rsub, you convolve the low SNR median-stacked frame to match a high SNR
image frame. (Usually this would be a sharpening).

In THEORY, either direction should be fine; it shouldn't matter. The
convolution should be able to sharpen just as well as it can blur.

ANYWAY it should be clear why if you just switch between rsub/nsub, you'll get
a sign error in your differenced magnitudes.
"""

import os, itertools
from glob import glob
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.io import fits

from numpy import array as nparr

if __name__ == "__main__":

    yaxisval='RMS'
    apstr='TF2'
    percentiles_xlim=None
    percentiles_ylim=None

    plt.close('all')
    fig, ax = plt.subplots(figsize=(4,3))

    # this plot is lines for each percentile in [2,25,50,75,98],
    # binned for every magnitude interval.
    markers = itertools.cycle(('o', 'v', '>', 'D', 's', 'P'))

    #1190 is nsub, 1192 is rsub
    for projid, sstr in zip([1190,1192],['nsub','rsub']):

        pctile_df = pd.read_csv('../data/stats_files_1190_vs_1192/'+
                                '{}_percentiles_RMS_vs_med_mag_TF2.csv'.
                                format(projid))

        for ix, row in pctile_df.iterrows():
            pctile = row.name
            label = '{} - {}%'.format(sstr, str(pctile))

            midbins = nparr(row.index)
            vals = nparr(row)

            ax.plot(midbins, vals, label=label, marker=next(markers))

        ax.legend(loc='best', fontsize='xx-small')

        ax.set_yscale('log')
        ax.set_xlabel('{:s} median instrument magnitude'.
                      format(apstr.upper()))
        ax.set_ylabel('{:s} {:s}'.
                      format(apstr.upper(), yaxisval))
        if percentiles_xlim:
            ax.set_xlim(percentiles_xlim)
        if percentiles_ylim:
            ax.set_ylim(percentiles_ylim)

        savname = ( os.path.join(
            '../data/stats_files_1190_vs_1192/','percentiles_{:s}_vs_med_mag_{:s}.png'.
            format(yaxisval, apstr.upper())
        ))
        fig.tight_layout()
        fig.savefig(savname, dpi=250)
