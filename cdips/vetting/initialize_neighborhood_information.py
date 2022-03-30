"""
Contents:
    get_group_and_neighborhood_information
    get_neighborhood_information
"""
#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

###########
# imports #
###########

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, socket

from cdips.utils.catalogs import (
    get_cdips_pub_catalog,
    ticid_to_toiid
)

from cdips.utils.gaiaqueries import (
    given_source_ids_get_gaia_data,
    query_neighborhood
)

from astrobase.services.identifiers import gaiadr2_to_tic
from astrobase.services.gaia import objectid_search

from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

#############
# functions #
#############

def get_neighborhood_information(
    source_id,
    overwrite=0,
    min_n_nbhrs=1000,
    manual_gmag_limit=None
    ):
    """
    Given a source_id for a star (potentially a field star), acquire
    information necessary for neighborhood diagnostic plots.

    Parameters:

        source_id: Gaia DR2 source_id

        overwrite: Whether the Gaia cache gets overwritten.

        manual_gmag_limit: G < manual_gmag_limit for the neighborhood
    """

    #
    # Get the targetname
    #
    ticid = gaiadr2_to_tic(str(source_id))
    toiid = ticid_to_toiid(ticid)

    if isinstance(toiid, str):
        targetname = toiid
    else:
        targetname = 'TIC{}.01'.format(ticid)

    #
    # Get Gaia information for target.
    #
    enforce_all_sourceids_viable = True
    savstr = '_nbhdonly'

    target_d = objectid_search(
        source_id,
        columns=('source_id', 'ra','dec', 'ra_error', 'dec_error',
                 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                 'l','b', 'parallax, parallax_error', 'pmra','pmra_error',
                 'pmdec','pmdec_error', 'radial_velocity'),
        forcefetch=True,
        gaia_mirror='vizier'
    )
    target_df = pd.read_csv(target_d['result'])
    assert len(target_df) == 1

    # now acquire the mean properties of the group, and query the neighborhood
    # based on those properties. the number of neighbor stars to randomly
    # select is min(5* the number of group members, 5000). (cutoff group
    # bounds based on parallax because further groups more uncertain).
    bounds = {}
    params = ['parallax', 'ra', 'dec']

    plx_mean = float(target_df.parallax)

    n_nbhrs = 0
    n_std = 5
    n_std_incr = 10
    n_std_max = 200

    if plx_mean > 10:
        n_std = 5
        n_std_incr = 20
        n_std_max = 1000

    while n_nbhrs < min_n_nbhrs:

        if n_std > n_std_max:
            return None

        LOGINFO('trying when bounding by {} stdevns'.format(n_std))

        for param in params:
            mult = 1 if 'parallax' in param else 2
            bounds[param+'_upper'] = (
                float(target_df[param]) + mult*n_std*float(target_df[param + '_error'])
            )
            bounds[param+'_lower'] = (
                float(target_df[param]) - mult*n_std*float(target_df[param + '_error'])
            )

        if bounds['parallax_lower'] < 0:
            bounds['parallax_lower'] = 0
        if bounds['ra_upper'] > 360:
            bounds['ra_upper'] = 359.999
        if bounds['ra_lower'] < 0:
            bounds['ra_lower'] = 0
        if bounds['dec_upper'] > 90:
            bounds['dec_upper'] = 89.999
        if bounds['dec_lower'] < -90:
            bounds['dec_lower'] = -89.999

        n_max = int(1e4)

        if manual_gmag_limit is None:
            manual_gmag_limit = 17

        groupname = '{}'.format(source_id)
        # only force overwrite if iterating
        if n_nbhrs == 0:
            nbhd_df = query_neighborhood(bounds, groupname, n_max=n_max,
                                         overwrite=overwrite,
                                         manual_gmag_limit=manual_gmag_limit)
        else:
            nbhd_df = query_neighborhood(bounds, groupname, n_max=n_max,
                                         overwrite=True,
                                         manual_gmag_limit=manual_gmag_limit)

        n_nbhrs = len(nbhd_df)
        LOGINFO(42*'=')
        LOGINFO('Got {} neighborhods, when minimum was {}'.
              format(n_nbhrs, min_n_nbhrs))
        LOGINFO(42*'=')

        n_std += n_std_incr

    n_std = 3
    pmdec_min = np.nanmean(nbhd_df['pmdec']) - n_std*np.nanstd(nbhd_df['pmdec'])
    pmdec_max = np.nanmean(nbhd_df['pmdec']) + n_std*np.nanstd(nbhd_df['pmdec'])
    pmra_min = np.nanmean(nbhd_df['pmra']) - n_std*np.nanstd(nbhd_df['pmra'])
    pmra_max = np.nanmean(nbhd_df['pmra']) + n_std*np.nanstd(nbhd_df['pmra'])

    pmdec_min = min((pmdec_min, float(target_df['pmdec'])))
    pmdec_max = max((pmdec_max, float(target_df['pmdec'])))
    pmra_min = min((pmra_min, float(target_df['pmra'])))
    pmra_max = max((pmra_max, float(target_df['pmra'])))

    return (targetname, groupname, target_df, nbhd_df,
            pmdec_min, pmdec_max, pmra_min, pmra_max)


def get_group_and_neighborhood_information(
    source_id,
    overwrite=0,
    force_groupname=None,
    force_references=None,
    force_cdips_match=True,
    manual_gmag_limit=None,
    CATALOG_VERSION=0.6):
    """
    Given a source_id for a cluster member, acquire information necessary for
    neighborhood diagnostic plots. (Namely, find all the group members, then do
    a Gaia query of everything).

    Parameters:

        source_id: Gaia DR2 source_id

        overwrite: Whether the Gaia cache gets overwritten.

        Optional kwargs: force_groupname and force_references. If passed, for
        example as the arrays ["kc19group_1222"] and ["Kounkel_2019"], plot
        generation will be forced without verifying that the target "source_id"
        is a member of the group.
    """

    if not isinstance(source_id, np.int64):
        source_id = np.int64(source_id)

    cdips_df = get_cdips_pub_catalog(ver=CATALOG_VERSION)
    row = cdips_df[cdips_df.source_id == source_id]

    if len(row['cluster']) == 0:
        LOGWARNING(f'Did not find any group matches for GAIA DR2 {source_id}.')
        return None

    #
    # Get numpy arrays of references and cluster names for this star.
    # This is needed b/c multiple memberships are comma-separated.
    #
    if 'reference' in row:
        references = np.array(row['reference'].iloc[0].split(','))
    elif 'reference_id' in row:
        references = np.array(row['reference_id'].iloc[0].split(','))
    clusters = np.array(row['cluster'].iloc[0].split(','))
    if force_references and force_groupname:
        references = force_references
        clusters = np.array([force_groupname])

    assert len(references) == len(clusters)

    if row is None and force_cdips_match:
        LOGINFO('Failed to get CDIPS target list match for {}'.format(source_id))
        return None

    #
    # Given the numpy array of clusters, find the best cluster membership list
    # to make the report with.
    #
    from cdips.catalogbuild.membership_lists import RANKED_MEMBERSHIP_DICT

    references_in_ensembles = RANKED_MEMBERSHIP_DICT[CATALOG_VERSION]['isgroup']
    references_in_field = RANKED_MEMBERSHIP_DICT[CATALOG_VERSION]['isfield']

    is_in_group = np.any(np.in1d(references, references_in_ensembles))
    is_in_field  = np.any(np.in1d(references, references_in_field))

    if is_in_group:
        # At least one of the references given corresponds to an actual coeval
        # group, and not a list of young field stars.  In this case, take the
        # highest precedence group (lowest index) as the one to make the plot
        # for.
        referencename = references_in_ensembles[
            int(min(np.argwhere(np.in1d(references_in_ensembles, references))))
        ]

        groupname = clusters[references == referencename][0]

    else:
        LOGWARNING(f'Did not find any group matches for GAIA DR2 {source_id}.')
        return None

    #
    # Get the targetname
    #
    ticid = gaiadr2_to_tic(str(source_id))
    toiid = ticid_to_toiid(ticid)

    if isinstance(toiid, str):
        targetname = toiid
    else:
        targetname = 'TIC{}.01'.format(ticid)

    #
    # Get the group members!
    #

    # (We avoided the field star case earlier.)
    cdips_df = cdips_df[~pd.isnull(cdips_df.cluster)]

    group_df = cdips_df[
        # avoids e.g., "kc19group_981" matching on "kc19group_98". deals with
        # cases beginning of string, middle of string, and end of string.
        (
            (cdips_df.cluster.str.contains(groupname+','))
            |
            (cdips_df.cluster.str.contains(','+groupname+','))
            |
            (cdips_df.cluster.str.contains(','+groupname+'$'))
            |
            (cdips_df.cluster == groupname)
        )
        &
        (cdips_df.reference_id.str.contains(referencename))
    ]

    group_source_ids = np.array(group_df['source_id']).astype(np.int64)
    np.testing.assert_array_equal(group_df['source_id'], group_source_ids)

    #
    # Given the source ids, get all the relevant Gaia information.
    #
    enforce_all_sourceids_viable = True
    savstr = f'_{groupname.replace(" ","_")}_{referencename}'

    group_df_dr2 = given_source_ids_get_gaia_data(
        group_source_ids, groupname.replace(" ","_"), overwrite=overwrite,
        enforce_all_sourceids_viable=enforce_all_sourceids_viable,
        n_max=min((len(group_source_ids), 10000)),
        savstr=savstr
    )

    target_d = objectid_search(
        source_id,
        columns=('source_id', 'ra','dec', 'phot_g_mean_mag',
                 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'l','b',
                 'parallax, parallax_error', 'pmra','pmra_error',
                 'pmdec','pmdec_error', 'radial_velocity'),
        forcefetch=True,
        gaia_mirror='vizier'
    )
    target_df = pd.read_csv(target_d['result'])
    assert len(target_df) == 1

    # now acquire the mean properties of the group, and query the neighborhood
    # based on those properties. the number of neighbor stars to randomly
    # select is min(5* the number of group members, 5000). (cutoff group
    # bounds based on parallax because further groups more uncertain).
    bounds = {}
    params = ['parallax', 'ra', 'dec']

    plx_mean = group_df_dr2['parallax'].mean()
    if plx_mean > 5:
        n_std = 5
    elif plx_mean > 3:
        n_std = 4
    else:
        n_std = 3

    LOGINFO(f'bounding by {n_std} stdevns')

    for param in params:
        bounds[param+'_upper'] = (
            group_df_dr2[param].mean() + n_std*group_df_dr2[param].std()
        )
        bounds[param+'_lower'] = (
            group_df_dr2[param].mean() - n_std*group_df_dr2[param].std()
        )

    if bounds['parallax_lower'] < 0:
        bounds['parallax_lower'] = 0
    if bounds['ra_upper'] > 360:
        bounds['ra_upper'] = 359.999
    if bounds['ra_lower'] < 0:
        bounds['ra_lower'] = 0
    if bounds['dec_upper'] > 90:
        bounds['dec_upper'] = 89.999
    if bounds['dec_lower'] < -90:
        bounds['dec_lower'] = -89.999

    assert bounds['ra_upper'] < 360
    assert bounds['ra_lower'] > 0
    assert bounds['parallax_lower'] >= 0

    n_max = min((50*len(group_df_dr2), 10000))

    if manual_gmag_limit is None:
        manual_gmag_limit = np.nanpercentile(group_df_dr2.phot_g_mean_mag,95)

    mstr = savstr
    nbhd_df = query_neighborhood(bounds, groupname.replace(" ","_"), n_max=n_max,
                                 overwrite=overwrite,
                                 manual_gmag_limit=manual_gmag_limit,
                                 mstr=mstr)

    # ensure no overlap between the group members and the neighborhood sample.
    common = group_df_dr2.merge(nbhd_df, on='source_id', how='inner')
    snbhd_df = nbhd_df[~nbhd_df.source_id.isin(common.source_id)]

    n_std = 5
    pmdec_min = group_df_dr2['pmdec'].mean() - n_std*group_df_dr2['pmdec'].std()
    pmdec_max = group_df_dr2['pmdec'].mean() + n_std*group_df_dr2['pmdec'].std()
    pmra_min = group_df_dr2['pmra'].mean() - n_std*group_df_dr2['pmra'].std()
    pmra_max = group_df_dr2['pmra'].mean() + n_std*group_df_dr2['pmra'].std()

    pmdec_min = min((pmdec_min, float(target_df['pmdec'])))
    pmdec_max = max((pmdec_max, float(target_df['pmdec'])))
    pmra_min = min((pmra_min, float(target_df['pmra'])))
    pmra_max = max((pmra_max, float(target_df['pmra'])))

    return (targetname, groupname, referencename, group_df_dr2, target_df,
            snbhd_df, pmdec_min, pmdec_max, pmra_min, pmra_max)
