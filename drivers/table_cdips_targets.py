"""
make TeX table with

source_id;cluster;reference;ext_catalog_name;ra;dec;pmra;pmdec;parallax;phot_g_mean_mag;phot_bp_mean_mag;phot_rp_mean_mag;k13_name_match;unique_cluster_name;how_match;not_in_k13;comment;k13_logt;k13_e_logt

To make the latex data table:
$ python table_of_cdips_targets.py
"""

import os
import pandas as pd, numpy as np
from numpy import array as nparr

from cdips.utils import collect_cdips_lightcurves as ccl

cdips_df = ccl.get_cdips_pub_catalog(ver=0.3)

N_rows = 4
sdf = cdips_df.sample(n=N_rows, random_state=43)

outtex = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/paper_I/table_cdips_targets.tex'
sdf.sort_values(by='source_id').T.to_latex(outtex, index=False)

print('made {}'.format(outtex))

# trim out the header and footer
print('{}: cutting head and bottom off tabular tables...'.format(outtex))
with open(outtex, 'r') as f:
    lines = f.readlines()

startline = [ix for ix, l in enumerate(lines) if
             l.startswith(r'\midrule')][0]
endline = [ix for ix, l in enumerate(lines) if
           l.startswith(r'\bottomrule')][0]

sel_lines = lines[startline+1: endline]

with open(outtex, 'w') as f:
    f.writelines(sel_lines)


