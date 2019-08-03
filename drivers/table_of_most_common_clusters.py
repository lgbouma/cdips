"""
make table like:

    sector 6                       sector 7

unique name | N_LCs | comment    unique name | N_LCs | comment

where comment is like "flag,Type,n_Type,SType" from Kharchenko.

*N_LCs: number of light curves with at least 500 points, and labels mtched to
unique name. More light curves in these cluisters may exist in the dataset, but
as unlabelled neighbor stars.


"""

import os
import pandas as pd, numpy as np
from numpy import array as nparr

from collections import Counter

from cdips.utils import collect_cdips_lightcurves as ccl

from astropy import units as u, constants as c
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

def get_supp_stats_df(sector=None, ndet_cut=500):

    statdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/'+
        'results/cdips_lc_stats/sector-{}'.format(sector)
    )

    statfile = os.path.join(statdir, 'supplemented_cdips_lc_statistics.txt')

    df = pd.read_csv(statfile, sep=';')

    df = df[df['ndet_rm2'] > ndet_cut]

    return df

def get_k13_index():
    #
    # the ~3784 row table
    #
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/A+A/558/A53')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    k13_index = catalogs[1].to_pandas()
    for c in k13_index.columns:
        if c != 'N':
            k13_index[c] = k13_index[c].str.decode('utf-8')

    return k13_index




def make_table():

    df_s6 = get_supp_stats_df(sector=6)
    df_s7 = get_supp_stats_df(sector=7)
    k13_index = get_k13_index()

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
    # where comment is like "flag,Type,n_Type,SType" from Kharchenko.
    tab_s6 = pd.DataFrame(allentries[0], columns=['name','n_lc','description'])
    tab_s7 = pd.DataFrame(allentries[1], columns=['name','n_lc','description'])

    outfull_s6 = '../paper_I/table_s6_most_common_clusters_full.csv'
    outfull_s7 = '../paper_I/table_s7_most_common_clusters_full.csv'
    outtab_s6 = '../paper_I/table_s6_most_common_clusters.tex'
    outtab_s7 = '../paper_I/table_s7_most_common_clusters.tex'

    tab_s6.to_csv(outfull_s6, index=False)
    tab_s7.to_csv(outfull_s7, index=False)


    import IPython; IPython.embed()

if __name__ == "__main__":
    make_table()
