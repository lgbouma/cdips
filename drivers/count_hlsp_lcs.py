"""
Given HLSP light curves, count them across sectors / cameras / ccds
"""
from glob import glob
import os
from os.path import join
import numpy as np, pandas as pd

sectors = range(40, 56)
cams = range(1, 5)
ccds = range(1, 5)

basedir = '/ar1/PROJ/luke/proj/CDIPS_LCS'

outdict = {}

for sector in sectors:
    for cam in cams:
        for ccd in ccds:
            print(sector, cam, ccd)
            n_lcs = len(
                glob(join( basedir,
                     f'sector-{sector}', f"cam{cam}_ccd{ccd}", "*fits")
                )
            )

            key = f"s{sector}_{cam}-{ccd}"

            outdict[key] = n_lcs

outdf = pd.DataFrame(outdict, index=[0])

outpath = join(basedir, 'count_hlsp_lcs.csv')
outdf.to_csv(outpath, index=False)
print(f"Wrote {outpath}")
