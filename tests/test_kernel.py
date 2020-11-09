import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib import cm

from datetime import datetime

kerneldir ='../data/kernels/'
testkernel = os.path.join(kerneldir,
                          'rsub-d2f9343c-tess2018360042939-s0006-1-1-0126_cal_img_bkgdsub-xtrns.fits-kernel')

def read_kernel_img(kernelpath):

    with open(kernelpath, mode='r') as f:
        lines = f.readlines()

    for ix, l in enumerate(lines):
        if l.startswith('# Image:'):
            startnum = ix

    imglines = [l for l in lines[startnum+1:]]

    clines = [','.join(l.split()) for l in imglines]
    clines = [c[2:]+'\n' for c in clines]

    temppath = os.path.abspath(kernelpath)+'temp'
    with open(temppath, mode='w') as f:
        f.writelines(clines)

    header = ['u','v','p0-q0','p0-q1','p0-q2','p1-q1','p1-q2','p2-q2']

    df = pd.read_csv(temppath, names=header)

    os.remove(temppath)

    return df


def _make_kernel_plot(_kernel, kernelsize, order):

    plt.close('all')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots(ncols=1, nrows=1)

    ##########################################

    vmin, vmax = -0.025, 0.025
    linnorm = colors.Normalize(vmin=vmin, vmax=vmax)

    #extent=[horizontal_min,horizontal_max,vertical_min,vertical_max]
    extent = [-3.5,3.5,-3.5,3.5]

    cset1 = ax.imshow(_kernel.T, cmap='RdBu', vmin=vmin, vmax=vmax,
                      norm=linnorm, interpolation='none', origin='lower',
                      extent=extent)

    for i in range(kernelsize):
        for j in range(kernelsize):

            u = i - int(np.floor(kernelsize/2))
            v = j - int(np.floor(kernelsize/2))

            ax.text(u, v-0.2, '{:.4f}'.format(_kernel[i,j]), ha='center',
                    va='top', fontsize='xx-small')

    ax.set_xlabel('u')
    ax.set_ylabel('v')

    divider0 = make_axes_locatable(ax)
    cax0 = divider0.append_axes('right', size='5%', pad=0.05)
    cb1 = fig.colorbar(cset1, ax=ax, cax=cax0, extend='both')

    fig.tight_layout(h_pad=0, w_pad=-14, pad=0)

    outpath = 'test_kernel_plots/kernel_{}.png'.format(order)
    fig.savefig(outpath, bbox_inches='tight', dpi=400)
    print('{}: made {}'.format(datetime.utcnow().isoformat(), outpath))




def plot_kernel(kernelpath, kernelsize, order='p0-q0'):

    #
    # read the image into a numpy array
    #
    df = read_kernel_img(kernelpath)

    sumkernel = np.zeros((kernelsize,kernelsize))

    orders = [ 'p0-q0', 'p0-q1', 'p0-q2', 'p1-q1', 'p1-q2', 'p2-q2' ]

    for order in orders:

        kernel = np.zeros((kernelsize,kernelsize))

        # e.g., -3 to +3 for a kernelsize of 7
        for i in range(kernelsize):
            for j in range(kernelsize):

                u = i - int(np.floor(kernelsize/2))
                v = j - int(np.floor(kernelsize/2))

                sel = (df['u'] == u) & (df['v'] == v)

                kernel[i,j] = float(df[sel][order])
                sumkernel[i,j] += float(df[sel][order])

        #
        # make the plot
        #
        _make_kernel_plot(kernel, kernelsize, order)

    #
    # do the same, for summed kernel
    #
    _make_kernel_plot(sumkernel, kernelsize, 'summed')


if __name__ == "__main__":

    kernelpath = testkernel
    kernelsize = 7

    plot_kernel(kernelpath, kernelsize)
