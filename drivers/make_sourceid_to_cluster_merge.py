"""
given a list of all the light-curves, make a dataframe matching light-curve to metadata

to make the list, something like

echo sector-?/cam?_ccd?/*.fits | xargs ls > foo.txt

is needed, with a merge for the two-digit cases.
"""

from cdips.utils.catalogs import get_cdips_catalog
import pandas as pd

lclistpath = '/nfs/phtess1/ar1/TESS/PROJ/lbouma/full_lc_list_20200120.txt'

with open(lclistpath, 'r') as f:
    lines = f.readlines()

lines = [l.rstrip('\n') for l in lines]

df = get_cdips_catalog()

sourceids = [l.split('/')[-1].split('gaiatwo')[-1].split('-')[0].lstrip('0') for l in lines]

lcdf = pd.DataFrame({'path':lines, 'source_id':sourceids})

lcdf.source_id = lcdf.source_id.astype(str)
df.source_id = df.source_id.astype(str)

mdf = lcdf.merge(df, on='source_id', how='left')

mdf.to_csv('/nfs/phtess1/ar1/TESS/PROJ/lbouma/source_ids_to_cluster_merge_20200120.csv', index=False, sep=';')
