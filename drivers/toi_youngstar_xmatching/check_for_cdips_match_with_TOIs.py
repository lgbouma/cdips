"""
Cross-match TOI list (or whatever other target list) against the CDIPS target
star catalog. What matches do you find?

$ python check_for_cdips_match.py > toi_list_match_check.txt &
"""
import os
import pandas as pd, numpy as np
from numpy import array as nparr

from collections import Counter

from cdips.utils import collect_cdips_lightcurves as ccl

from astropy import units as u, constants as c
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

def main():

    df = ccl.get_cdips_pub_catalog(ver=0.4)

    ra = nparr(df['ra'])
    dec = nparr(df['dec'])

    cdips_coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree),
                           frame='icrs')

    toidate = '2020-03-09'
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
    scols = ['source_id', 'cluster', 'reference', 'sep_arcsec']

    match_list = []
    name_list = []
    for this_coord, n in zip(toi_coord, names):

        seps = this_coord.separation(cdips_coord).to(u.arcsec)

        df['sep_arcsec'] = seps

        sdf = df[scols].sort_values(by='sep_arcsec')

        if (
            sdf.iloc[0]['sep_arcsec']<1
            and
            sdf.iloc[0]['reference'] != 'Zari_2018_UMS'
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
        '{}_CDIPS_TOI_match.csv'.format(toidate)
    )
    match_df.to_csv(outpath, index=False, sep=';')
    print('made {}'.format(outpath))


if __name__ == "__main__":
    main()
