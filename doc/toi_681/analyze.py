import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from glob import glob

from cdips.lcproc import mask_orbit_edges as moe

fpaths = np.sort(glob('tess*.fits'))

for fpath in fpaths:
    print(fpath)
    hdul = fits.open(fpath)

    time = hdul[1].data['TIME']
    flux = np.sum(np.sum(hdul[1].data['FLUX'], axis=1), axis=1)

    try:
        time, flux = moe.mask_orbit_start_and_end(time, flux, orbitgap=0.2,
                                                  orbitpadding=48/(24))
    except:
        print('skipping')
        pass

    f,ax = plt.subplots(figsize=(4,3))
    ax.scatter(time, flux, c='k', s=1)
    f.savefig(fpath.replace('.fits','.png'))
