"""
functions to directly parse vizier tables.

run_v05_vizier_to_csv
get_vizier_table_as_dataframe
"""

import os
import numpy as np, pandas as pd

from astroquery.vizier import Vizier
from cdips.paths import DATADIR
clusterdir = os.path.join(DATADIR, 'cluster_data')

VIZIERDICT05 = {
"CantatGaudin2020a": ["J/A+A/633/A99", 1, "Source|Cluster", ''],
"CastroGinard20": ["J/A+A/635/A45", 1, "Source|Cluster", ''],
"Meingast2021": ["J/A+A/645/A84", 0, "GaiaDR2|Cluster", ''],
"Ujjwal2020": ["J/AJ/159/166", 1, "Gaia|Group", ''], # moving groups
"Meingast2019": ["J/A+A/621/L3", 13, "Source", 'Hyades'],
"Damiani2019t1": ["J/A+A/623/A112", 0, "DR2Name", 'ScoOB2_PMS'],
"Damiani2019t2": ["J/A+A/623/A112", 1, "DR2Name", 'ScoOB2_UMS'],
"Goldman2018": ["J/ApJ/868/32", 0, "DR2Name", 'LCC'],
"Roccatagliata2020": ["J/A+A/638/A85", 0, "GaiaDR2", "Taurus"],
"RoserSchilbach2020t1": ["J/A+A/638/A9", 0, "Source", "PscEri"],
"RoserSchilbach2020t2": ["J/A+A/638/A9", 1, "Source", "Pleaides"],
"Ratzenbock2020": ["J/A+A/639/A64", 0, "GaiaDR2", "PscEri"],
"EsplinLuhman2019": ["J/AJ/158/54", 0, "Gaia", "Taurus"],
"Furnkranz2019t1": ["J/A+A/624/L11", 0, 'Source', 'ComaBer'],
"Furnkranz2019t2": ["J/A+A/624/L11", 1, 'Source', 'ComaBerNeighborGroup']
}

def run_v05_vizier_to_csv():
    """
    A driver to get all the source_ids and cluster names from VIZIERDICT05
    written to /data/cluster_data/v05.
    """

    outdir = os.path.join(clusterdir, 'v05')

    for outname, t in VIZIERDICT05.items():

        vizier_search_str, table_num, srccolumns, groupname = t

        if "|" not in srccolumns:
            dstcolumns = "source_id" # dr2
        elif srccolumns.count('|') == 1:
            dstcolumns = "source_id|cluster"
        else:
            raise NotImplementedError('need to define dstcolumns')

        print(79*'-')
        print(f'{outname}')

        outcsvpath = os.path.join(outdir, f'{outname}_viziercols.csv')

        if not os.path.exists(outcsvpath):

            df = get_vizier_table_as_dataframe(
                vizier_search_str, srccolumns, dstcolumns,
                table_num=table_num
            )

            if groupname != '':
                df['cluster'] = groupname

            df.to_csv(outcsvpath, index=False)
            print(f'Wrote {outcsvpath}')

        else:
            print(f'Found {outcsvpath}')



def get_vizier_table_as_dataframe(vizier_search_str, srccolumns, dstcolumns,
                                  table_num=0, verbose=1,
                                  whichcataloglist='default'):
    """
    Download a table from Vizier (specified by `vizier_search_str`). Get
    columns specified by the |-separated string, `columns`. Write the
    resulting table to a CSV file, `outcsvpath`.  Sometimes, the table number
    on Vizier (`table_num`) must be specified.  In the context of CDIPS catalog
    construction, this is mostly for making "source|cluster|age" output.

    Args:
        vizier_search_str (str): specifies Vizier search.
        srccolumns (str): columns in the Vizier catalog to get e.g.,
            "RA|DE|age"
        dstcolumns (str): columns to rename them to, e.g., "ra|dec|log10age".
        whichcataloglist (str): "default", else a string to specify the
            catalog, e.g., ""J/A+A/640/A1""
    """

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs(vizier_search_str)
    if whichcataloglist == 'default':
        catalogs = Vizier.get_catalogs(catalog_list.keys())
    else:
        catalogs = Vizier.get_catalogs(whichcataloglist)

    tab = catalogs[table_num]
    if verbose:
        print(f'initial number of members: {len(tab)}')

    df = tab.to_pandas()

    # rename columns to homogeneous format. e.g.,
    # df = df.rename(columns={"Gaia": "source_id", "Group": "cluster"})
    df = df.rename(
        columns={
            k:v for k,v in zip(srccolumns.split("|"), dstcolumns.split("|"))
        }
    )

    if 'source_id' in dstcolumns.split("|"):
        # strip "Gaia DR2 6083332579305755520" style format if present
        try:
            if np.any(df.source_id.str.contains('Gaia DR2 ')):
                df['source_id'] = df.source_id.str.replace('Gaia DR2 ', '')
        except AttributeError:
            pass

        # if NaNs are present, omit them. (e.g., from EsplinLuhman19)
        if np.any(pd.isnull(df.source_id)):
            N = len(df[pd.isnull(df.source_id)])
            print(f'WRN! Found {N} null source_id values of {len(df)}. Omitting.')
            df = df[~pd.isnull(df.source_id)]

        df['source_id'] = df['source_id'].astype('int64')

    sdf = df[dstcolumns.split("|")]

    return sdf
