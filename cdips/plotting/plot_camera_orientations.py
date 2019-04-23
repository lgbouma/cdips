import os
import numpy as np, pandas as pd
from tessmaps import tessmaps as tm

from astropy.coordinates import SkyCoord
from astropy import units as u

def main(sectorstr='s0002'):

    datadir = '../results/camera_orientations/'
    datapath = os.path.join(datadir,'{}_orientations.csv'.format(sectorstr))
    df = pd.read_csv(datapath)

    elons, elats = df['crval_elon']*u.deg, df['crval_elat']*u.deg
    coords = SkyCoord(lon=elons, lat=elats, frame='barycentrictrueecliptic')

    names = np.array(
        [str(cam)+'-'+str(ccd) for cam, ccd in zip(df['cam'],df['ccd'])]
    )

    annotate = np.ones_like(names).astype(bool)

    savname = '{}_orientations.png'.format(sectorstr)
    savdir = '../results/camera_orientations/' # fix this to your preferred directory!
    title = 'camera ccd orientations'
    sector_number = 1

    tm.make_rect_map(sector_number, coords, names=names,
                     annotate_bools=annotate, title=title,
                     bkgnd_cmap='Blues', savname=savname, savdir=savdir)

    print('if you got here w/out errors, it worked!')

if __name__=="__main__":
    main()
