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
    'phtess2':'/home/lbouma/proj/cdips/data/cluster_data/MWSC_1sigma_members_Gaia_matched',
    'brik':'/home/luke/Dropbox/proj/cdips/data/cluster_data/MWSC_1sigma_members_Gaia_matched',
    'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/cluster_data/MWSC_1sigma_members_Gaia_matched'
}
kc19_path_d = {
    'phtess2':'/home/lbouma/proj/cdips/data/cluster_data/kc19_string_table1.csv',
    'brik':'/home/luke/Dropbox/proj/cdips/data/cluster_data/kc19_string_table1.csv',
    'ast1607-astro':'/Users/luke/Dropbox/proj/cdips/data/cluster_data/kc19_string_table1.csv'
}

K13_GAIA_DIR = k13_dir_d[socket.gethostname()]
KC19_PATH = kc19_path_d[socket.gethostname()]

#############
# functions #
#############

def get_neighborhood_information(
    source_id,
    mmbr_dict=None,
    k13_notes_df=None
    overwrite = 0):
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
    """

    row = get_cdips_pub_catalog_entry(source_id, ver=0.4)

    # multiple memberships ,-separated. get lists of references, cluster names,
    # ext catalog names.
    references = row['reference'].iloc[0].split(',')
    clusters = row['cluster'].iloc[0].split(',')
    ext_catalog_names = row['ext_catalog_name'].iloc[0].split(',')

    #
    # See if target source hits any of Cantat-Gaudin+2018, Kounkel&Covey2019,
    # Kharchenko+2013, or Kharchenko "match" (in order of precedence).
    # Kharchenko "match" is through the indirect naming scheme from the
    # appendix.
    #
    group_in_cg18 = False
    group_in_k13 = False
    group_in_kc19 = False

    if 'CantatGaudin_2018' in references:
        group_in_cg18 = True
        groupname = clusters[np.in1d(references, 'CantatGaudin_2018')]

    elif 'Kounkel_2019' in references:
        group_in_kc19 = True
        groupname = clusters[np.in1d(references, 'Kounkel_2019')]

    elif 'Kharchenko2013' in references:
        group_in_k13 = True
        groupname = clusters[np.in1d(references, 'Kharchenko2013')]

    elif isinstance(mmbr_dict, dict):
        if mmbr_dict['mwscid'] == 'N/A':
            pass
        else:
            # get the kharchenko2013 name, & use it for the search
            mwscid = mmbr_dict['mwscid']
            group_in_k13 = True
            groupname = k13_notes_df[k13_notes_df.MWSC == mwscid].Name.iloc[0]

    if not (group_in_kc19 or group_in_k13 or group_in_cg18):
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
    if group_in_cg18:
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
        kc19_df = kc19_df[kc19_df['group_id'] == group_id]
        group_source_ids = np.array(kc19_df['source_id']).astype(np.int64)

    #
    # Given the source ids, get all the relevant Gaia information.
    #
    if group_in_cg18:
        enforce_all_sourceids_viable = True
    elif group_in_kc19:
        enforce_all_sourceids_viable = True
    elif group_in_k13:
        enforce_all_sourceids_viable = False
    else:
        enforce_all_sourceids_viable = True

    group_df_dr2 = given_source_ids_get_gaia_data(
        group_source_ids, groupname, overwrite=overwrite,
        enforce_all_sourceids_viable=enforce_all_sourceids_viable
    )

    target_d = objectid_search(
        source_id,
        columns=('source_id', 'ra','dec', 'phot_g_mean_mag',
                 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'l','b',
                 'parallax, parallax_error', 'pmra','pmra_error',
                 'pmdec','pmdec_error', 'radial_velocity'),
        forcefetch=True
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
        bounds[param+'_upper'] = (
            group_df_dr2[param].mean() + n_std*group_df_dr2[param].std()
        )
        bounds[param+'_lower'] = (
            group_df_dr2[param].mean() - n_std*group_df_dr2[param].std()
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
                                 is_k13_group=group_in_k13)

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

    return (targetname, groupname, group_df_dr2, target_df, nbhd_df,
            cutoff_probability, pmdec_min, pmdec_max, pmra_min, pmra_max,
            group_in_k13, group_in_cg18, group_in_kc19)
