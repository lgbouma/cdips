# -*- coding: utf-8 -*-
'''
functions to download and wrangle catalogs of cluster members that I want to
crossmatch against Gaia-DR2.

called from `homogenize_cluster_lists.py`

Contents:
    Zari18_stars_to_csv
    CantatGaudin2019_velaOB2_to_csv
    VillaVelez18_check
'''
import os, pickle, subprocess, itertools, socket
from glob import glob

import numpy as np, pandas as pd
from numpy import array as nparr

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.io.votable import from_table, writeto, parse

from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

from astrobase.timeutils import precess_coordinates
from datetime import datetime

from cdips.paths import DATADIR
datadir = DATADIR
clusterdatadir = os.path.join(DATADIR, 'cluster_data')

def Zari18_stars_to_csv():
    """
    The Zari+ 2018 work constructs large samples of young stars within 500 pc.
    It includes many upper main sequence and pre-main sequence members of
    the Sco-Cen, Orion and Vela star-forming regions. There are also less
    massive regions represented in Taurus, Perseus, Cepheus, etc.

    Annoyingly, no attempt is made to map between the star and the name of
    the cluster. So the "cluster name" column for this one will be "N/A".
    """

    indir = clusterdatadir
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


def CantatGaudin2019_velaOB2_to_csv():
    # https://ui.adsabs.harvard.edu/abs/2019A%26A...626A..17C/abstract

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/A+A/626/A17')
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    tab = catalogs[0]
    df = tab.to_pandas()

    outdf = df[['Source','Pop']]

    outdf = outdf.rename(columns={"Source": "source_id", "Pop": "cluster"})

    # Quoting Tristan, discussing Fig3 from the paper:
    #  """
    #  the clumps that are labeled are known "open clusters". The diffuse
    #  stellar distribution in the middle and around was called the Vela OB2
    #  association, supposed to be in front of the clusters or maybe in
    #  between, and everyone thought they were unrelated objects, different
    #  age, different history. In the figure, the colour code indicates stars
    #  that have the same age and velocity.
    #  My conclusion is that there is no object that can be called "Vela OB2"
    #  association. There are clusters, and a diffuse distribution of stars around
    #  each cluster. And the sum of all those fluffy distributions is "the
    #  association". Aggregates of young stars are so sub-structured that it
    #  doesn't even make sense to label all the clumps, and it can even be
    #  difficult to guess which clump observers referred to when they looked at
    #  those regions in the 19th century. There are ongoing debates over whether
    #  NGC 1746 exists, for instance. Which are the stars that were originally
    #  classified as NGC 1746, and does it even matter?
    #  """

    outdf['cluster'] = np.core.defchararray.add(
        np.repeat('cg19velaOB2_pop', len(outdf)),
        nparr(outdf['cluster']).astype(str)
    )

    outpath = os.path.join(clusterdatadir, 'moving_groups',
                           'CantatGaudin2019_velaOB2_MATCH.csv')
    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def VillaVelez18_check():
    assert os.path.exists(os.path.join(
        clusterdatadir, 'moving_groups',
        'VillaVelez_2018_DR2_PreMainSequence_MATCH.csv')
    )

