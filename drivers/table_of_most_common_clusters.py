"""
make TeX table like:

    sector 6                       sector 7

unique name | N_LCs | comment    unique name | N_LCs | comment

where comment is like "flag,Type,n_Type,SType" from Kharchenko.

*N_LCs: number of light curves with at least 500 points, and labels mtched to
unique name. More light curves in these cluisters may exist in the dataset, but
as unlabelled neighbor stars.

To make the latex header tables and associated csv files:
$ python table_of_most_common_clusters.py
"""

import os
import pandas as pd, numpy as np
from numpy import array as nparr

from collections import Counter

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils import get_vizier_catalogs as gvc

from astropy import units as u, constants as c
from astropy.coordinates import SkyCoord

def get_supp_stats_df(sector=None, ndet_cut=500):

    statdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/'+
        'results/cdips_lc_stats/sector-{}'.format(sector)
    )

    statfile = os.path.join(statdir, 'supplemented_cdips_lc_statistics.txt')

    df = pd.read_csv(statfile, sep=';')

    df = df[df['ndet_rm2'] > ndet_cut]

    return df

def make_table():

    df_s6 = get_supp_stats_df(sector=6)
    df_s7 = get_supp_stats_df(sector=7)
    k13_index = gvc.get_k13_index()

    allentries = []
    for df in [df_s6, df_s7]:

        ucname = np.array(df['unique_cluster_name'])
        ucname = ucname[~pd.isnull(ucname)]
        cname_cnt = Counter(ucname)
        cname_mostcommon = cname_cnt.most_common(n=5000)

        entries = []
        for name, count in cname_mostcommon:

            sel = k13_index['Name'] == name

            if len(k13_index[sel]) == 0:
                comment = 'no K13 match'

            elif len(k13_index[sel]) > 1:

                if len(k13_index[sel]) == 2:

                    # there are a couple doubles. in these cases take that
                    # non-duplicate information.
                    sel &= (k13_index['flag'] != '&')

                    if len(k13_index[sel]) != 1:
                        raise NotImplementedError('wtf')

                    comment = ','.join([
                        str(k13_index[sel]['flag'].iloc[0]),
                        str(k13_index[sel]['Type'].iloc[0]),
                        str(k13_index[sel]['n_Type'].iloc[0]),
                        str(k13_index[sel]['SType'].iloc[0])
                    ])

                else:

                    raise ValueError('got >2 cluster matches')

            else:

                comment = ','.join([
                    str(k13_index[sel]['flag'].iloc[0]),
                    str(k13_index[sel]['Type'].iloc[0]),
                    str(k13_index[sel]['n_Type'].iloc[0]),
                    str(k13_index[sel]['SType'].iloc[0])
                ])

            entries.append(
                (name, count, comment)
            )

        allentries.append(entries)

    # sector 6                         sector 7
    # unique name | N_LCs | comment    unique name | N_LCs | comment
    #
    # where comment (or description) is like "flag,Type,n_Type,SType" from
    # Kharchenko.
    tab_s6 = pd.DataFrame(allentries[0], columns=['Name','$N_{\\mathrm{lc}}$','Description'])
    tab_s7 = pd.DataFrame(allentries[1], columns=['Name','$N_{\\mathrm{lc}}$','Description'])

    outfull_s6 = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/paper_I/table_s6_most_common_clusters_full.csv'
    outfull_s7 = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/paper_I/table_s7_most_common_clusters_full.csv'
    outtex_s6 = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/paper_I/table_s6_most_common_clusters.tex'
    outtex_s7 = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/paper_I/table_s7_most_common_clusters.tex'
    outpaths = [outfull_s6, outfull_s7, outtex_s6, outfull_s7]

    tab_s6.to_csv(outfull_s6, index=False)
    tab_s7.to_csv(outfull_s7, index=False)

    tab_s6.head(n=20).to_latex(outtex_s6, index=False, column_format='lll')
    tab_s7.head(n=20).to_latex(outtex_s7, index=False, column_format='lll')

    for p in outpaths:
        print('made {}'.format(p))

    for p in [outtex_s6, outtex_s7]:
        print('{}: cutting head and bottom off tabular tables...'.format(p))
        with open(p, 'r') as f:
            lines = f.readlines()

        startline = [ix for ix, l in enumerate(lines) if
                     l.startswith(r'\midrule')][0]
        endline = [ix for ix, l in enumerate(lines) if
                   l.startswith(r'\bottomrule')][0]

        sel_lines = lines[startline+1: endline]

        with open(p, 'w') as f:
            f.writelines(sel_lines)



if __name__ == "__main__":
    make_table()
