from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

import numpy as np, pandas as pd
import os
from glob import glob
from parse import search
from numpy import array as nparr

def main(sectorstr="s0002"):

    ffidir = (
        '/home/luke/local/tess-trex/fullframeimages/set_single_time_{}'.
        format(sectorstr)
    )

    ffipaths = glob(os.path.join(ffidir, 'tess*_ffir.fits'))

    cams,ccds,crval1s,crval2s = [],[],[],[]
    for ffipath in ffipaths:

        sr = search("{}-s{}-{}-{}-{}.fits", os.path.basename(ffipath))
        cam = int(sr[2])
        ccd = int(sr[3])

        hdulist = fits.open(ffipath)
        hdr = hdulist[1].header

        crval1 = hdr['CRVAL1']
        crval2 = hdr['CRVAL2']

        hdulist.close()

        cams.append(cam)
        ccds.append(ccd)
        crval1s.append(crval1)
        crval2s.append(crval2)

    ra = nparr(crval1s)
    dec = nparr(crval2s)
    c = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    elon = nparr(c.barycentrictrueecliptic.lon.value)
    elat = nparr(c.barycentrictrueecliptic.lat.value)

    df = pd.DataFrame(
        {'cam':nparr(cams), 'ccd':nparr(ccds),
         'crval_ra':ra, 'crval_dec':dec,
         'crval_elon':elon, 'crval_elat':elat
        }
    )

    outdir = '../results/camera_orientations/'
    outname = '{}_orientations.csv'.format(sectorstr)
    outpath = os.path.join(outdir, outname)

    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))

if __name__=="__main__":

    sectorstr='s0002'

    main(sectorstr=sectorstr)
