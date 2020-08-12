"""
Given a list of all the light-curves, make a dataframe matching light-curve to
metadata.

To make the list of all LC paths, use
    /nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/get_lc_list.sh
"""

import os
import pandas as pd
from cdips.utils.catalogs import get_cdips_catalog
from cdips.utils import today_YYYYMMDD

lclistpath = f'/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/lc_list_{today_YYYYMMDD()}.txt'

with open(lclistpath, 'r') as f:
    lines = f.readlines()

lines = [l.rstrip('\n') for l in lines]

df = get_cdips_catalog()

sourceids = [l.split('/')[-1].split('gaiatwo')[-1].split('-')[0].lstrip('0') for l in lines]

lcdf = pd.DataFrame({'path':lines, 'source_id':sourceids})

lcdf.source_id = lcdf.source_id.astype(str)
df.source_id = df.source_id.astype(str)

mdf = lcdf.merge(df, on='source_id', how='left')

outdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/lc_metadata'
outpath = os.path.join(
    outdir, f'source_ids_to_cluster_merge_{today_YYYYMMDD()}.csv'
)
mdf.to_csv(outpath, index=False, sep=';')
print(f'made {outpath}')
