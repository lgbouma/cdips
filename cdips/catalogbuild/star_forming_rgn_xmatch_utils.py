# -*- coding: utf-8 -*-
'''
functions to download and wrangle catalogs of cluster members that I want to
crossmatch against Gaia-DR2.

called from `homogenize_cluster_lists.py`
'''
import os, pickle, subprocess, itertools
from glob import glob

import numpy as np, pandas as pd
from numpy import array as nparr

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.io.votable import from_table, writeto, parse

from astroquery.gaia import Gaia

from astrobase.timeutils import precess_coordinates
from datetime import datetime

def Zari18_stars_to_csv():
    """
    The Zari+ 2018 work constructs large samples of young stars within 500 pc.
    It includes many upper main sequence and pre-main sequence members of
    the Sco-Cen, Orion and Vela star-forming regions. There are also less
    massive regions represented in Taurus, Perseus, Cepheus, etc.

    Annoyingly, no attempt is made to map between the star and the name of
    the cluster. So the "cluster name" column for this one will be "N/A".
    """

    indir = '/home/luke/Dropbox/proj/cdips/data/cluster_data/'
    inpaths = [
        os.path.join(indir,'Zari_2018_pms_tab.vot'),
        os.path.join(indir,'Zari_2018_ums_tab.vot')
    ]

    for inpath in inpaths:
        tab = parse(inpath)

        t = tab.get_first_table().to_table()

        df = pd.DataFrame({'source':t['Source'],
                           'cluster':np.repeat("N/A", len(t))})

        outpath = inpath.replace('.vot','_cut_only_source_cluster_MATCH.csv')
        outdir = os.path.join(indir, 'moving_groups')
        outpath = os.path.join(
            outdir,
            os.path.basename(outpath)
        )
        df.to_csv(outpath, index=False)
        print('made {}'.format(outpath))
