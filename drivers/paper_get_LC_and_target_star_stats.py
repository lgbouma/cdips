"""
analyze stats of both cdips target stars, and cdips light curves.

* how many are there total?
* what fraction come from which sources?
* how many are single-source claims?
* how many are multi-source claims?

python -u paper_get_LC_and_target_star_stats.py &> logs/get_LC_and_target_star_stats.log &
"""

import pandas as pd, numpy as np
from numpy import array as nparr
from collections import Counter
from cdips.utils import collect_cdips_lightcurves as ccl
import os
from astropy import units as u, constants as c
from astropy.coordinates import SkyCoord

def main():
    print('#'*42)
    paper_get_CDIPS_LC_stats(sector=6)
    print('#'*42)
    paper_get_CDIPS_LC_stats(sector=7)
    print('#'*42)
    paper_get_CDIPS_star_stats()

def paper_get_CDIPS_LC_stats(sector=None, ndet_cut=500, ncluster_cut=200):

    statdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/'+
        'results/cdips_lc_stats/sector-{}'.format(sector)
    )

    statfile = os.path.join(statdir, 'supplemented_cdips_lc_statistics.txt')

    df = pd.read_csv(statfile, sep=';')

    df = df[df['ndet_rm2'] > ndet_cut]

    print('Sector {}'.format(sector))
    get_CDIPS_stats(df, ncluster_cut=ncluster_cut, find_cluster_prox=True)


def paper_get_CDIPS_star_stats():

    df = ccl.get_cdips_pub_catalog(ver=0.4)

    get_CDIPS_stats(df)


def get_CDIPS_stats(df, ncluster_cut=100, find_cluster_prox=False):
    """
    df (pd.DataFrame): dataframe with stats information from either the
    (publication-level) target star catalog (ccl.get_cdips_catalog), or the
    supplemented_cdips_lc_statistics.txt dataframes.
    """

    n_stars = len(df)

    #
    # I was worried about whether we had many stars at all in the Orion
    # vicinity.
    #
    if find_cluster_prox:

        ra = nparr(df['RA[deg][2]'])
        dec = nparr(df['Dec[deg][3]'])

        lc_coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

        m42_c = "05 35 17.3 -05 23 28"
        m42_coord = SkyCoord(m42_c, unit=(u.hourangle, u.deg))

        seps = m42_coord.separation(lc_coord)

        # http://spider.seds.org/ngc/revngcic.cgi?NGC1976
        r_cut = 40 * u.arcmin

        d_pc = 412
        d_lower = d_pc - 100
        d_upper = d_pc + 100

        plx_upper = 1/d_lower
        plx_lower = 1/d_upper

        plx = nparr(df['Parallax[mas][6]'])/1e3

        sel = (seps < r_cut) & (plx > plx_lower) & (plx < plx_upper)

        m42_stars = df[sel]

        N_m42_stars = len(m42_stars)

        print('Got {} LCs in the vicinity of M42 (100pc, 40 arcmin)'.
              format(N_m42_stars))


    ucname = np.array(df['unique_cluster_name'])
    cname_cnt = Counter(ucname)
    cname_mostcommon = cname_cnt.most_common(n=ncluster_cut)

    refs = np.array(df['reference'])
    cnt = Counter(refs)
    mostcommon = cnt.most_common(n=20)

    fracmostcommon = [(k, np.round(v/len(df), 3)) for k,v in mostcommon]

    ismult = np.array([',' in r for r in refs])
    issing = ~ismult

    n_single = len(refs[issing])
    n_mult = len(refs[ismult])

    print(
    """
    Total number of stars: {n_stars}

    Top {ncluster_cut} most common unique cluster names are:
    {cname_mostcommon}

    Top 20 most common references are:
    {mostcommon}

    By fraction, top 20 most common references are:
    {fracmostcommon}

    There are {n_single} single-source claims.

    There are {n_mult} claims from multiple sources.
    """.format(
        n_stars=n_stars,
        ncluster_cut=ncluster_cut,
        cname_mostcommon=repr(cname_mostcommon),
        mostcommon=repr(mostcommon),
        fracmostcommon=repr(fracmostcommon),
        n_single=n_single,
        n_mult=n_mult
    )
    )

if __name__ == "__main__":
    main()
