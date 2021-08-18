"""
Test the "reformat_headers" function in lcproc.reformat_lcs_for_mast, to make
sure the PCA calculation looks OK.
"""
import numpy as np
import os
from glob import glob
from astrobase import imageutils as iu
from cdips.lcproc import detrend as dtr
from cdips.lcproc import reformat_lcs_for_mast as rlm

camccddir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-14/cam1_ccd1'
assert os.path.exists(camccddir)

symlinkdir = '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS/'

lcpaths = glob(os.path.join(symlinkdir, 'sector-14/cam1_ccd1/*llc.fits'))

np.random.seed(42)
lcpaths = np.random.choice(lcpaths, 1, replace=False)

projid = iu.get_header_keyword(lcpaths[0], 'PROJID')

cam = 1
ccd = 1
sector = 14

eigveclist, n_comp_df = dtr.prepare_pca(cam, ccd, sector, projid)

cdipsvnum=1
OC_MG_CAT_ver = 0.6

rlm.reformat_headers(lcpaths, camccddir, sector, cdipsvnum, OC_MG_CAT_ver,
                     eigveclist=eigveclist, n_comp_df=n_comp_df)

