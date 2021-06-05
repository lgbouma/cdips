"""
Goal: get Gaia DR2 IDs of stars in "clusters". These lists define the sample of
target stars for which we make light-curves.

See ../doc/list_of_cluster_member_lists.ods for an organized spreadsheet of the
different member lists.
"""

import matplotlib.pyplot as plt, pandas as pd, numpy as np
import os, pickle, subprocess, itertools, socket
from glob import glob

from numpy import array as nparr

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.io import ascii
from astropy.io.votable import from_table, writeto, parse

from astroquery.gaia import Gaia

from astrobase.timeutils import precess_coordinates
from datetime import datetime

from cdips.catalogbuild.concatenate_merge import (
    get_target_catalog
)
from cdips.catalogbuild.simbad_xmatch_utils import (
    run_SIMBAD_to_csv, SIMBAD_bibcode_to_GaiaDR2_csv
)
from cdips.catalogbuild.vizier_xmatch_utils import (
    run_v05_vizier_to_csv
)
from cdips.catalogbuild.nasa_xmatch_utils import (
    NASAExoArchive_to_csv
)
from cdips.catalogbuild.open_cluster_xmatch_utils import (
    GaiaCollaboration2018_clusters_to_csv,
    # NOTE: commented because <=v0.4 deprecated
    #Kharchenko2013_position_mag_match_Gaia,
    #Dias2014_nbhr_gaia_to_nearestnbhr,
    KounkelCovey2019_clusters_to_csv,
    Kounkel2020_to_csv,
    Kounkel2018_orion_to_csv,
    CantatGaudin20b_to_csv,
)
from cdips.catalogbuild.moving_group_xmatch_utils import (
    # NOTE: commented because <=v0.4 deprecated
    # make_Gagne18_BANYAN_XI_GaiaDR2_crossmatch,
    # make_Gagne18_BANYAN_XII_GaiaDR2_crossmatch,
    # make_Gagne18_BANYAN_XIII_GaiaDR2_crossmatch,
    # make_Rizzuto11_GaiaDR2_crossmatch,
    # make_Kraus14_GaiaDR2_crossmatch,
    # make_Luhman12_GaiaDR2_crossmatch,
    # make_Bell17_GaiaDR2_crossmatch,
    # make_Roser11_GaiaDR2_crossmatch,
    # make_Oh17_GaiaDR2_crossmatch,
    Tian2020_to_csv,
    Pavlidou2021_to_csv,
    Gagne2020_to_csv,
    Rizzuto2017_to_csv,
    Kerr2021_to_csv
)
from cdips.catalogbuild.star_forming_rgn_xmatch_utils import (
    Zari18_stars_to_csv,
    VillaVelez18_check,
    CantatGaudin2019_velaOB2_to_csv
)

from cdips.paths import DATADIR, LOCALDIR
clusterdatadir = os.path.join(DATADIR, 'cluster_data')
localdir = LOCALDIR

def main():

    ###########
    # <= v0.4 #
    ###########
    # OCs
    GaiaCollab18 = 0
    K13 = 0
    D14 = 0
    KC19 = 0
    K18 = 0
    CG18 = 0
    # MGs
    do_BANYAN_XI = 0
    do_BANYAN_XII = 0
    do_BANYAN_XIII = 0
    do_Kraus_14 = 0
    do_Roser_11 = 0
    do_Bell_17 = 0
    do_Oh17 = 0
    do_Rizzuto11 = 0
    do_VillaVelez18 = 0
    # Star forming regions
    do_Zari18 = 0
    do_CG19_vela = 0

    ###########
    # >= v0.5 #
    ###########
    # vizier calls: ["CantatGaudin2020a", "CastroGinard20", "Meingast2021",
    # "Meingast2019", "Damiani2019", "Briceno2019", "Goldman2018",
    # "Roccatagliata2020", "RoserSchilbach2020", "Ratzenbock2020",
    # "EsplinLuhman2019", "Ujjwal2020", "Furnkranz2019"]
    do_v05_vizier_calls = 0
    do_v05_simbad_bibcodes = 0
    do_SIMBAD_otype_calls = 0
    do_Kounkel20 = 0
    do_CantatGaudin20b = 0
    do_Tian20 = 0
    do_Pavlidou21 = 0
    do_Gagne20 = 0
    do_Rizzuto17 = 0
    do_NASAExoArchive = 0
    do_Kerr2021 = 0

    do_the_merge = 1
    catalog_vnum = '0.6'

    if do_the_merge:
        get_target_catalog(catalog_vnum)

    if do_Kerr2021:
        Kerr2021_to_csv()
    if do_NASAExoArchive:
        NASAExoArchive_to_csv()
    if do_v05_vizier_calls:
        run_v05_vizier_to_csv()
    if do_Kounkel20:
        Kounkel2020_to_csv()
    if do_CantatGaudin20b:
        CantatGaudin20b_to_csv()
    if do_Tian20:
        Tian2020_to_csv()
    if do_Gagne20:
        Gagne2020_to_csv()
    if do_SIMBAD_otype_calls:
        run_SIMBAD_to_csv(get_longtypes=1)
        run_SIMBAD_to_csv(get_longtypes=0)
    if do_v05_simbad_bibcodes:
        # CottenSong2016 Kraus2014 Oh2017 Gagne 2018abc
        bibcodes = ['2016ApJS..225...15C', '2014AJ....147..146K',
                    '2017AJ....153..257O', '2018ApJ...862..138G',
                    '2018ApJ...860...43G', '2018ApJ...856...23G' ]
        for b in bibcodes:
            SIMBAD_bibcode_to_GaiaDR2_csv(
                b, os.path.join(clusterdatadir, 'v05')
            )
    if do_Pavlidou21:
        Pavlidou2021_to_csv()
    if do_Rizzuto17:
        Rizzuto2017_to_csv()
    if KC19:
        KounkelCovey2019_clusters_to_csv()
    if K18:
        Kounkel2018_orion_to_csv()
    if GaiaCollab18:
        GaiaCollaboration2018_clusters_to_csv()
    if CG18:
        CantatGaudin2018_to_csv()
    if do_Zari18:
        Zari18_stars_to_csv()
    if do_CG19_vela:
        CantatGaudin2019_velaOB2_to_csv()

    # # NOTE: DEPRECATED CALLS; >=v0.5 used SIMBAD_bibcode_to_GaiaDR2_csv
    # if do_BANYAN_XI:
    #     make_Gagne18_BANYAN_XI_GaiaDR2_crossmatch()
    # if do_BANYAN_XII:
    #     make_Gagne18_BANYAN_XII_GaiaDR2_crossmatch()
    # if do_BANYAN_XIII:
    #     make_Gagne18_BANYAN_XIII_GaiaDR2_crossmatch()
    # if do_Kraus_14:
    #     make_Kraus14_GaiaDR2_crossmatch()
    #if do_Oh17:
    #    make_Oh17_GaiaDR2_crossmatch()
    # NOTE: omitted entirely in >=v0.5
    #if do_Rizzuto11:
    #    make_Rizzuto11_GaiaDR2_crossmatch()
    #if do_Roser_11:
    #    make_Roser11_GaiaDR2_crossmatch()
    #if do_Bell_17:
    #    make_Bell17_GaiaDR2_crossmatch()
    #if K13:
    #    Kharchenko2013_position_mag_match_Gaia()
    #if D14:
    #    Dias2014_nbhr_gaia_to_nearestnbhr()

    ## NOTE: the VillaVelez github source has an int64 conversion error. use
    ## your hacked source instead.
    #if do_VillaVelez18:
    #    VillaVelez18_check()

if __name__ == "__main__":
    main()
