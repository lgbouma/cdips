"""
Cross-match TOI list against the Kounkel&Covey string catalog. What matches do
you find?

$ python check_for_cdips_match.py > output.txt &
"""
import os
import pandas as pd, numpy as np
from numpy import array as nparr

from collections import Counter

from cdips.utils import collect_cdips_lightcurves as ccl

from astropy import units as u, constants as c
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

import socket

def main():

    if socket.gethostname() != 'brik':
        raise ValueError('this runs on brik')

    kc19path = (
        '/home/luke/Dropbox/proj/stringcheese/data/'
        'kounkel_table1_sourceinfo.csv'
    )
    df = pd.read_csv(kc19path)

    ra = nparr(df['ra_x'])
    dec = nparr(df['dec_x'])

    catalog_coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree),
                             frame='icrs')

    toidate = '2019-12-05'
    toidf = ccl.get_toi_catalog(ver=toidate)

    toi_ra, toi_dec = (nparr(toidf['TIC Right Ascension']),
                       nparr(toidf['TIC Declination']))
    toi_coord = SkyCoord(ra=toi_ra, dec=toi_dec, unit=(u.degree, u.degree),
                         frame='icrs')
    names = nparr(toidf['Full TOI ID'])
    #
    # # toi 251;  tic 146520535
    # names = ['toi 251', 'tic 146520535']
    # clist = ['23:32:14.89 -37:15:21.11', '05:06:35.91 -20:14:44.2']
    #
    scols = ['source_id', 'group_id', 'name', 'age', 'sep_arcsec']

    match_list = []
    name_list = []
    for this_coord, n in zip(toi_coord, names):

        seps = this_coord.separation(catalog_coord).to(u.arcsec)

        df['sep_arcsec'] = seps

        sdf = df[scols].sort_values(by='sep_arcsec')

        if (
            sdf.iloc[0]['sep_arcsec']<1
        ):

            print(42*'#')
            print(n)
            print(sdf.head(n=3).to_string(index=False))
            print('\n')

            match_list.append(sdf.iloc[0])
            name_list.append(n)

    match_df = pd.DataFrame(np.array(match_list), columns=scols)
    match_df['toi'] = name_list
    outpath = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/toi_youngstar_xmatching/'
        '{}_KC19_TOI_match.csv'.format(toidate)
    )
    match_df.to_csv(outpath, index=False, sep=';')
    print('made {}'.format(outpath))


if __name__ == "__main__":
    main()
