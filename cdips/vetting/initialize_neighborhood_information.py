import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, socket

from cdips.utils.catalogs import (
    get_cdips_pub_catalog_entry,
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

##########
# config #
##########

k13_dir_d = {
    'phn12':'/home/lbouma/proj/cdips/data/cluster_data/MWSC_1sigma_members_Gaia_matched',
    'phtess2':'/home/lbouma/proj/cdips/data/cluster_data/MWSC_1sigma_members_Gaia_matched',
    'brik':'/home/luke/Dropbox/proj/cdips/data/cluster_data/MWSC_1sigma_members_Gaia_matched',
    'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/cluster_data/MWSC_1sigma_members_Gaia_matched'
}
kc19_path_d = {
    'phn12':'/home/lbouma/proj/cdips/data/cluster_data/kc19_string_table1.csv',
    'phtess2':'/home/lbouma/proj/cdips/data/cluster_data/kc19_string_table1.csv',
    'brik':'/home/luke/Dropbox/proj/cdips/data/cluster_data/kc19_string_table1.csv',
    'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/cluster_data/kc19_string_table1.csv'
}
k18_path_d = {
    'phn12':'/home/lbouma/proj/cdips/data/cluster_data/Kounkel_2018_orion_table2_cut_only_source_cluster.csv',
    'phtess2':'/home/lbouma/proj/cdips/data/cluster_data/Kounkel_2018_orion_table2_cut_only_source_cluster.csv',
    'brik':'/home/luke/Dropbox/proj/cdips/data/cluster_data/Kounkel_2018_orion_table2_cut_only_source_cluster.csv',
    'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/cluster_data/Kounkel_2018_orion_table2_cut_only_source_cluster.csv'
}

K13_GAIA_DIR = k13_dir_d[socket.gethostname()]
KC19_PATH = kc19_path_d[socket.gethostname()]
K18_PATH = k18_path_d[socket.gethostname()]

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
    n_std_incr = 5
    n_std_max = 200

    if plx_mean > 10:
        n_std = 5
        n_std_incr = 20
        n_std_max = 1000

    while n_nbhrs < min_n_nbhrs:

        if n_std > n_std_max:
            return None

        print('trying when bounding by {} stdevns'.format(n_std))

        for param in params:
            mult = 1 if 'parallax' in param else 2
            bounds[param+'_upper'] = (
                float(target_df[param]) + mult*n_std*float(target_df[param + '_error'])
            )
            bounds[param+'_lower'] = (
                float(target_df[param]) - mult*n_std*float(target_df[param + '_error'])
            )

        if bounds['ra_upper'] > 360:
            bounds['ra_upper'] = 359.99
        if bounds['ra_lower'] < 0:
            bounds['ra_lower'] = 0
        if bounds['parallax_lower'] < 0:
            bounds['parallax_lower'] = 0

        n_max = int(1e4)
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
        print(42*'=')
        print('Got {} neighborhods, when minimum was {}'.
              format(n_nbhrs, min_n_nbhrs))
        print(42*'=')

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
    mmbr_dict=None,
    k13_notes_df=None,
    overwrite=0,
    force_groupname=None,
    force_references=None,
    force_cdips_match=True):
    """
    Given a source_id for a cluster member, acquire information necessary for
    neighborhood diagnostic plots. (Namely, find all the group members, then do
    a Gaia query of everything).

    Parameters:

        source_id: Gaia DR2 source_id

        mmbr_dict: optional. This is the dictionary returned by
        `cdips.plotting.vetting_pdf.cluster_membership_check`. If passed, it's
        used to also include Kharchenko "matches" (where the match is done via
        the scheme desribed in the CDIPS-I paper appendix). Otherwise, only
        CG18, KC19, and K13 are directly matched.

        k13_notes_df: optional. If passing mmbr_dict, pass this as well.

        overwrite: Whether the Gaia cache gets overwritten.

        Optional kwargs: force_groupname and force_references. If passed, for
        example as the strings "kc19group_1222" and "Kounkel_2019", plot
        generation will be forced without verifying that the target "source_id"
        is a member of the group.
    """

    row = get_cdips_pub_catalog_entry(source_id, ver=0.4)

    if row is None and force_cdips_match:
        print('Failed to get CDIPS target list match for {}'.format(source_id))
        return None

    # multiple memberships ,-separated. get lists of references, cluster names,
    # ext catalog names.
    if row.has_key('reference'):
        references = np.array(row['reference'].iloc[0].split(','))
    elif row.has_key('reference_id'):
        references = np.array(row['reference_id'].iloc[0].split(','))
    clusters = np.array(row['cluster'].iloc[0].split(','))
    if row.has_key('ext_catalog_name'):
        ext_catalog_names = np.array(row['ext_catalog_name'].iloc[0].split(','))
    else:
        ext_catalog_names = ''
    if force_references and force_groupname:
        references = force_references
        clusters = np.array([force_groupname])

    #
    # See if target source hits any of Cantat-Gaudin+2018, Kounkel&Covey2019,
    # Kharchenko+2013, or Kharchenko "match" (in order of precedence).
    # Kharchenko "match" is through the indirect naming scheme from the
    # appendix.
    #
    group_in_k18 = False
    group_in_cg18 = False
    group_in_k13 = False
    group_in_kc19 = False

    if 'Kounkel_2018_Ori' in references:
        group_in_k18 = True
        groupname = clusters[np.in1d(references, 'Kounkel_2018_Ori')][0]

    elif 'CantatGaudin_2018' in references:
        group_in_cg18 = True
        groupname = clusters[np.in1d(references, 'CantatGaudin_2018')][0]

    elif 'Kounkel_2019' in references:
        group_in_kc19 = True
        groupname = clusters[np.in1d(references, 'Kounkel_2019')][0]

    elif 'Kharchenko2013' in references and isinstance(mmbr_dict, dict):
        group_in_k13 = True
        groupname = clusters[np.in1d(references, 'Kharchenko2013')][0]
        mwscid = mmbr_dict['mwscid']

    elif isinstance(mmbr_dict, dict):
        if mmbr_dict['mwscid'] == 'N/A':
            pass
        else:
            # get the kharchenko2013 name, & use it for the search
            mwscid = mmbr_dict['mwscid']
            group_in_k13 = True
            groupname = k13_notes_df[k13_notes_df.MWSC == mwscid].Name.iloc[0]

    if not (group_in_kc19 or group_in_k13 or group_in_cg18 or group_in_k18):
        print('WRN! Did not get any valid group matches for {}'.
              format(source_id))
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
    if group_in_k18:
        cutoff_probability = 1
        k18_df = pd.read_csv(K18_PATH, sep=',')
        sel = (k18_df.cluster == groupname)
        group_df = k18_df[sel]
        group_source_ids = np.array(group_df['source_id']).astype(np.int64)
        np.testing.assert_array_equal(group_df['source_id'], group_source_ids)

    elif group_in_cg18:
        cutoff_probability = 0.1
        v = Vizier(column_filters={"Cluster":groupname})
        v.ROW_LIMIT = 3000
        cg18_vizier_str ='J/A+A/618/A93'
        catalog_list = v.find_catalogs(cg18_vizier_str)
        catalogs = v.get_catalogs(catalog_list.keys())

        group_tab = catalogs[1]
        group_df = group_tab.to_pandas()
        assert len(group_df) < v.ROW_LIMIT
        np.testing.assert_array_equal(group_tab['Source'], group_df['Source'])

        group_df = group_df[group_df['PMemb'] > cutoff_probability]
        group_source_ids = np.array(group_df['Source']).astype(np.int64)

    elif group_in_k13:
        # use the output of cdips.open_cluster_xmatch_utils
        k13starpath = os.path.join(K13_GAIA_DIR,
                                   '{}_{}_1sigma_members.csv'.
                                   format(mwscid, groupname))

        cutoff_probability = 0.61
        k13_df = pd.read_csv(k13starpath)
        group_source_ids = np.array(k13_df['gaia_dr2_match_id']).astype(np.int64)

    elif group_in_kc19:
        cutoff_probability = 1
        kc19_df = pd.read_csv(KC19_PATH)
        try:
            group_id = kc19_df[
                kc19_df.source_id.astype(str) == str(source_id)
            ].group_id.iloc[0]
        except IndexError as e:
            if not force_groupname is None and 'kc19' in force_groupname:
                group_id = int(force_groupname.split('_')[-1])
            else:
                raise IndexError(e)
        group_df = kc19_df[kc19_df.group_id == group_id]
        group_source_ids = np.array(group_df['source_id']).astype(np.int64)
        np.testing.assert_array_equal(group_df['source_id'], group_source_ids)

    #
    # Given the source ids, get all the relevant Gaia information.
    #
    if group_in_k18:
        enforce_all_sourceids_viable = True
        savstr='_k18'
    elif group_in_cg18:
        enforce_all_sourceids_viable = True
        savstr='_cg18'
    elif group_in_kc19:
        enforce_all_sourceids_viable = True
        savstr='_kc19'
    elif group_in_k13:
        enforce_all_sourceids_viable = False
        savstr='_k13match'
    else:
        enforce_all_sourceids_viable = True

    group_df_dr2 = given_source_ids_get_gaia_data(
        group_source_ids, groupname, overwrite=overwrite,
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
        n_std = 4.5
    else:
        n_std = 4

    if group_in_kc19:
        n_std = 2

    print('bounding by {} stdevns'.format(n_std))

    for param in params:
        bounds[param+'_upper'] = np.minimum(
            group_df_dr2[param].mean() + n_std*group_df_dr2[param].std(),
            360-1e-5
        )
        bounds[param+'_lower'] = np.maximum(
            group_df_dr2[param].mean() - n_std*group_df_dr2[param].std(),
            1e-5
        )

    if bounds['parallax_lower'] < 0:
        bounds['parallax_lower'] = 0

    assert bounds['ra_upper'] < 360
    assert bounds['ra_lower'] > 0
    assert bounds['parallax_lower'] >= 0

    n_max = min((50*len(group_df_dr2), 10000))
    nbhd_df = query_neighborhood(bounds, groupname, n_max=n_max,
                                 overwrite=overwrite,
                                 is_cg18_group=group_in_cg18,
                                 is_kc19_group=group_in_kc19,
                                 is_k13_group=group_in_k13,
                                 is_k18_group=group_in_k18)

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

    return (targetname, groupname, group_df_dr2, target_df, snbhd_df,
            cutoff_probability, pmdec_min, pmdec_max, pmra_min, pmra_max,
            group_in_k13, group_in_cg18, group_in_kc19, group_in_k18)

