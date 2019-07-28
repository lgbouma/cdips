"""
analyze stats of both cdips target stars, and cdips light curves.

* how many are there total?
* what fraction come from which sources?
* how many are single-source claims?
* how many are multi-source claims?

python paper_get_LC_and_target_star_stats.py &> logs/get_LC_and_target_star_stats.log &
"""

import pandas as pd, numpy as np
from collections import Counter
from cdips.utils import collect_cdips_lightcurves as ccl

def main():
    print('#'*42)
    paper_get_CDIPS_LC_stats(sector=6)
    print('#'*42)
    paper_get_CDIPS_LC_stats(sector=7)
    print('#'*42)
    paper_get_CDIPS_star_stats()

def paper_get_CDIPS_LC_stats(sector=None):

    statdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/'+
        'results/cdips_lc_stats/sector-{}'.format(sector)
    )

    statfile = os.path.join(statsdir, 'supplemented_cdips_lc_statistics.txt')

    df = pd.read_csv(statfile)

    get_CDIPS_stats(df)


def paper_get_CDIPS_star_stats():

    df = ccl.get_cdips_pub_catalog(ver=0.3)

    get_CDIPS_stats(df)


def get_CDIPS_stats(df):
    """
    df (pd.DataFrame): dataframe with stats information from either the
    (publication-level) target star catalog (ccl.get_cdips_catalog), or the
    supplemented_cdips_lc_statistics.txt dataframes.
    """

    n_stars = len(df)

    ucname = np.array(df['unique_cluster_name'])
    cname_cnt = Counter(ucname)
    cname_mostcommon = cname_cnt.most_common(n=30)

    refs = np.array(df['reference'])
    cnt = Counter(refs)
    mostcommon = cnt.most_common(n=20)

    ismult = np.array([',' in r for r in refs])
    issing = ~ismult

    n_single = len(refs[issing])
    n_mult = len(refs[ismult])

    print(
    """
    Total number of stars: {n_stars}

    Top 30 most common unique cluster names are:
    {cname_mostcommon}

    Top 20 most common references are:
    {mostcommon}

    There are {n_single} single-source claims.

    There are {n_mult} claims from multiple sources.
    """.format(
        n_stars=n_stars,
        cname_mostcommon=repr(cname_mostcommon),
        mostcommon=repr(mostcommon),
        n_single=n_single,
        n_mult=n_mult
    )
    )

if __name__ == "__main__":
    main()
