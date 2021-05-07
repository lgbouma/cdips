"""
Given OC_MG_FINAL_GaiaRp_lt_16_v{versionnum}.csv, made from
catalogbuild/homogenize_cluster_lists.py, match what you can against
Kharchenko2013 to get cluster names and ages.

(Except for KC19, which gets its original ages)

Contents:
    construct_unique_cluster_name_column
    get_unique_cluster_name
    get_k13_df
    get_k13_name_match
"""

###########
# imports #
###########

from glob import glob
import os, re, requests, time, socket
import numpy as np, pandas as pd
from numpy import array as nparr

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as const

from itertools import repeat

from cdips.utils import collect_cdips_lightcurves as ccl
from astropy.io.votable import from_table, writeto, parse

from cdips.paths import DATADIR
clusterdatadir = os.path.join(DATADIR, 'cluster_data')

########
# code #
########

def construct_unique_cluster_name_column(cdips_cat_vnum=0.4):
    """
    We already have a catalog with the following columns:

        source_id;cluster;reference;ext_catalog_name;ra;dec;pmra;pmdec;parallax;
        phot_g_mean_mag;phot_bp_mean_mag;phot_rp_mean_mag;

    We want to supplement it with:

        unique_cluster_name;k13_name_match;how_match;not_in_k13;
        comment;logt;e_logt;logt_provenance

    ----------
    The need for a "unique_cluster_name" is mainly in order to (a) connect with
    other literature, and (b) know roughly how many "unique clusters" exist and
    have been searched. The concept of "unique cluster" in substuctured regions
    like Orion or Sco-OB2 is not defined, so whatever we arrive at for the
    unique name should not be taken as gospel.

    For "comment", we want any relevant information about the cluster (e.g., if
    its existence is dubious, if it overlaps with other clusters, etc.)

    For "logt", "e_logt", and "logt_provenance", we want both the K13 age if
    it's available, and the KC19 age, if that's available.
    ----------
    """

    cdips_df = ccl.get_cdips_catalog(ver=cdips_cat_vnum)
    k13 = get_k13_df()

    sourceid = nparr(cdips_df['source_id'])
    clusterarr = nparr(cdips_df['cluster'])
    ras = nparr(cdips_df['ra'])
    decs = nparr(cdips_df['dec'])
    referencearr = nparr(cdips_df['reference'])

    #
    # Attempt to get Kharchenko+2013 matches from a mix of cluster names and
    # star positions (i.e., Appendix B of CDIPS-I).
    #
    inds = ~pd.isnull(clusterarr)
    sourceid = sourceid[inds]
    clusterarr = clusterarr[inds]
    ras = ras[inds]
    decs = decs[inds]
    referencearr = referencearr[inds]
    uarr, inds = np.unique(clusterarr, return_index=True)
    sourceid = sourceid[inds]
    clusterarr = clusterarr[inds]
    ras = ras[inds]
    decs = decs[inds]
    referencearr = referencearr[inds]

    namematchpath = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/OC_MG_FINAL_v{}_with_K13_name_match.csv'.
        format(cdips_cat_vnum)
    )

    if not os.path.exists(namematchpath):
        res = list(map(get_k13_name_match,
                   zip(clusterarr, ras, decs, referencearr, repeat(k13))))

        resdf = pd.DataFrame(res,
            columns=['k13_name_match', 'how_match', 'have_name_match',
                     'have_mwsc_id_match', 'is_known_asterism', 'not_in_k13',
                     'why_not_in_k13']
        )
        resdf['source_id'] = sourceid
        mdf = resdf.merge(cdips_df, how='left', on='source_id')
        mdf.to_csv(namematchpath, index=False, sep=';')
        print('made {}'.format(namematchpath))

    else:
        mdf = pd.read_csv(namematchpath, sep=';')

    uniqpath = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/OC_MG_FINAL_v{}_uniq.csv'.
        format(cdips_cat_vnum)
    )

    if not os.path.exists(uniqpath):
        mdf['unique_cluster_name'] = list(map(
            get_unique_cluster_name, zip(mdf.iterrows()))
        )

        cols = ['unique_cluster_name', 'k13_name_match', 'how_match',
                'have_name_match', 'have_mwsc_id_match', 'is_known_asterism',
                'not_in_k13', 'why_not_in_k13', 'source_id', 'cluster', 'dec',
                'dist', 'ext_catalog_name', 'parallax', 'phot_bp_mean_mag',
                'phot_g_mean_mag', 'phot_rp_mean_mag', 'pmdec', 'pmra', 'ra',
                'reference']

        mdf[cols].to_csv(uniqpath, index=False, sep=';')
        print('made {}'.format(uniqpath))

    else:
        mdf = pd.read_csv(uniqpath, sep=';')

    #
    # Merge the unique cluster names against the whole cdips dataframe, using
    # mdf as a lookup table. This is slightly wrong, b/c some spatial matches
    # would give different results than using this string match. However they
    # are a small subset, this is computationally cheaper, and assigning a
    # unique name has inherent limitations.
    #
    subdf = mdf[['unique_cluster_name', 'k13_name_match', 'how_match',
                 'have_name_match', 'have_mwsc_id_match', 'is_known_asterism',
                 'not_in_k13', 'why_not_in_k13', 'cluster']]

    print('beginning big merge...')

    fdf = cdips_df.merge(subdf, on='cluster', how='left')

    assert len(fdf) == len(cdips_df)

    print(42*'#')
    print('# unique cluster names: {}'.format(
        len(np.unique(
            fdf[~pd.isnull(fdf['unique_cluster_name'])]['unique_cluster_name'])
        ))
    )
    print('fraction with k13 name match: {:.5f}'.format(
        len(fdf[~pd.isnull(fdf['k13_name_match'])])/len(fdf)
    ))
    print('fraction with any unique name: {:.5f}'.format(
        len(fdf[~pd.isnull(fdf['unique_cluster_name'])])/len(fdf)
    ))

    #
    # Merge fdf k13_name_match against K13 index, and get updated comments.
    # Join "why_not_in_k13" with "Source object type" "SType" from K13 index.
    # This gives the "comments" column.
    #
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/A+A/558/A53')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    k13_index = catalogs[1].to_pandas()
    for c in k13_index.columns:
        if c != 'N':
            k13_index[c] = k13_index[c].str.decode('utf-8')

    k13_df = k13

    styped = {
        "ass":"stellar association",
        "ast":"Dias: possible asterism/dust hole/star cloud",
        "dub":"Dias: dubious, objects considered doubtful by the DSS images inspection",
        "emb":"embedded open cluster/cluster associated with nebulosity",
        "glo":"globular cluster/possible globular cluster",
        "irc":"infrared cluster",
        "irg":"infrared stellar group",
        "mog":"Dias: possible moving group",
        "non":"Dias: non-existent NGC/ objects not found in DSS images inspection",
        "rem":"Possible cluster remnant",
        "var":"clusters with variable extinction"
    }

    k13_index['STypeComment'] = k13_index['SType'].map(styped)

    # in order to do the desired name join, must remove duplicates from k13
    # index. since we only care about the comments, keep "last" works from
    # inspection.
    ids = k13_index['Name']
    print('removing duplicates from K13 index...\n{}'.format(repr(
        k13_index[ids.isin(ids[ids.duplicated()])].sort_values(by='Name'))))

    k13_index = k13_index.drop_duplicates(subset=['Name'], keep='last')

    _df = fdf.merge(k13_index, left_on='k13_name_match', right_on='Name',
                    how='left')

    assert len(_df) == len(fdf)

    comment = _df['why_not_in_k13']
    stypecomment = _df['STypeComment']

    _df['comment'] = comment.map(str) + ". " + stypecomment.map(str)
    _df['comment'] = _df['comment'].map(lambda x: x.replace('nan',''))
    _df['comment'] = _df['comment'].map(lambda x: x.lstrip('. ').rstrip('.  '))

    # unique comments include:
    # array(['',
    #    'Dias: dubious, objects considered doubtful by the DSS images inspection',
    #    'Dias: non-existent NGC/ objects not found in DSS images inspection',
    #    'Dias: possible asterism/dust hole/star cloud',
    #    'K13index: duplicated/coincides with other cluster',
    #    'K13index: duplicated/coincides with other cluster. Dias: dubious, objects considered doubtful by the DSS images inspection',
    #    'K13index: duplicated/coincides with other cluster. Dias: non-existent NGC/ objects not found in DSS images inspection',
    #    'K13index: duplicated/coincides with other cluster. clusters with variable extinction',
    #    'K13index: duplicated/coincides with other cluster. embedded open cluster/cluster associated with nebulosity',
    #    'K13index: duplicated/coincides with other cluster. infrared cluster',
    #    'K13index: duplicated/coincides with other cluster. stellar association',
    #    'K13index: possibly this is a cluster, but parameters are not determined',
    #    'K13index: possibly this is a cluster, but parameters are not determined. Dias: dubious, objects considered doubtful by the DSS images inspection',
    #    'K13index: possibly this is a cluster, but parameters are not determined. Dias: non-existent NGC/ objects not found in DSS images inspection',
    #    'K13index: possibly this is a cluster, but parameters are not determined. infrared cluster',
    #    'K13index: possibly this is a cluster, but parameters are not determined. stellar association',
    #    'K13index: this is not a cluster',
    #    'K13index: this is not a cluster. Dias: dubious, objects considered doubtful by the DSS images inspection',
    #    'K13index: this is not a cluster. Dias: non-existent NGC/ objects not found in DSS images inspection',
    #    'K13index: this is not a cluster. Dias: possible asterism/dust hole/star cloud',
    #    'K13index: this is not a cluster. Possible cluster remt',
    #    'K13index: this is not a cluster. infrared cluster',
    #    'Majaess IR cluster match missing in K13', 'Possible cluster remt',
    #    'clusters with variable extinction',
    #    'embedded open cluster/cluster associated with nebulosity',
    #    'infrared cluster', 'is_bell_mg', 'is_gagne_mg',
    #    'is_gagne_mg. infrared cluster',
    #    'is_gagne_mg. stellar association', 'is_gaia_member',
    #    'is_kraus_mg', 'is_oh_mg', 'is_oh_mg. stellar association',
    #    'is_rizzuto_mg. stellar association', 'known missing from K13',
    #    'stellar association']
    fdf['comment'] = _df['comment']
    del _df

    #
    # Assign ages. If a Kharchenko+2013 name match was found, report that age.
    # In addition, if the star is in the Kounkel & Covey 2019 table, report
    # their age as well.
    #
    _df = fdf.merge(k13_df,
                    left_on='k13_name_match',
                    right_on='Name',
                    how='left')
    assert len(_df) == len(fdf)

    k13_logt = np.round(nparr(_df['logt']),2).astype(str)
    k13_e_logt = np.round(nparr(_df['e_logt']),2).astype(str)

    k13_logt_prov = np.repeat('',len(fdf)).astype('>U20')
    k13_logt_prov[k13_logt != 'nan'] = 'Kharchenko2013'

    # kounkel & covey 2019 age matches are only for the groups that were not
    # already known.  (otherwise, use the kharchenko age, which is assumed to
    # exist). do it by searching the cluster name. extract the groupid from the
    # name.

    kc19_df1 = pd.read_csv(os.path.join(clusterdatadir,'KC19_string_table1.csv'))
    kc19_df2 = pd.read_csv(os.path.join(clusterdatadir,'KC19_string_table2.csv'))
    kc19_sdf2 = kc19_df2[['group_id','name','age']]
    kc19_mdf = kc19_df1.merge(kc19_sdf2, on='group_id', how='left')

    # if the source_id is in the Kounkel & Covey table, then also report the
    # KC19 age.
    kc19_merge_cdips_df = fdf.merge(kc19_mdf, how='left', on='source_id')
    assert len(kc19_merge_cdips_df) == len(fdf)

    kc19_logt = np.round(nparr(kc19_merge_cdips_df['age']),2).astype(str)

    kc19_e_logt = np.repeat('',len(fdf)).astype('>U10')
    kc19_e_logt[kc19_logt != 'nan'] = '0.15' # Kounkel&Covey 2019 abstract precision

    kc19_logt_prov = np.repeat('',len(fdf)).astype('>U20')
    kc19_logt_prov[kc19_logt != 'nan'] = 'Kounkel_2019'

    #
    # Concatenate and separate by ",". Then remove all "nans". NOTE: if you add
    # extra age provenances (i.e., more than K13 and KC19), this nan-stripping
    # will require adjustments.
    #
    logt = list(map(','.join, zip(k13_logt, kc19_logt)))
    e_logt = list(map(','.join, zip(k13_e_logt, kc19_e_logt)))
    logt_prov = list(map(','.join, zip(k13_logt_prov, kc19_logt_prov)))

    logt = [_.lstrip('nan,').rstrip(',nan') for _ in logt]
    e_logt = [_.lstrip('nan,').rstrip(',nan') for _ in e_logt]
    logt_prov = [_.lstrip(',').rstrip(',') for _ in logt_prov]

    fdf['logt'] = logt
    fdf['e_logt'] = e_logt
    fdf['logt_provenance'] = logt_prov

    # reformat for table to publish
    scols = ['source_id', 'cluster', 'reference', 'ext_catalog_name', 'ra',
             'dec', 'pmra', 'pmdec', 'parallax', 'phot_g_mean_mag',
             'phot_bp_mean_mag', 'phot_rp_mean_mag', 'k13_name_match',
             'unique_cluster_name', 'how_match', 'not_in_k13', 'comment',
             'logt', 'e_logt', 'logt_provenance']
    _df = fdf[scols]

    pubpath = (
        '/nfs/phtess1/ar1/TESS/PROJ/lbouma/OC_MG_FINAL_v{}_publishable.csv'.
        format(cdips_cat_vnum)
    )
    _df.to_csv(pubpath, index=False, sep=';')
    print('made {}'.format(pubpath))

    import IPython; IPython.embed()


def get_unique_cluster_name(task):
    # ix = index
    # mdfr = row of merged data frame
    ix, mdfr = task[0][0], task[0][1]

    if not pd.isnull(mdfr['k13_name_match']):
        return mdfr['k13_name_match']

    if 'is_zari' in mdfr['why_not_in_k13']:
        return np.nan

    new_cluster_groups = ['Gulliver', 'RSG']
    for _n in new_cluster_groups:
        if _n in mdfr['cluster']:
            cluster = str(mdfr['cluster'])
            for c in cluster.split(','):
                if _n in c:
                    return c

    cluster = str(mdfr['cluster'])
    # hyades, and other moving group specifics
    if 'HYA' in cluster or 'Hyades' in cluster:
        return 'Hyades'
    if 'Tuc-Hor' in cluster:
        return 'THA'
    if '{beta}PMG' in cluster:
        return 'BPMG'
    if '32Ori' in cluster:
        return 'THOR'

    # for these groups, keep original naming scheme, and assign it override
    # priority.
    special_group_keys = [
        'cg19velaOB2_pop', 'k18orion_', 'kc19group_'
    ]
    for k in special_group_keys:
        if k in cluster:
            bar = pd.Series(cluster.split(','))
            uname = bar[bar.str.contains(k)].iloc[0]
            return uname

    if (
        pd.isnull(mdfr['k13_name_match']) and
        mdfr['why_not_in_k13'] in
        ['is_gagne_mg','is_oh_mg','is_rizzuto_mg','is_bell_mg','is_kraus_mg']
    ):
        for c in cluster.split(','):
            if not pd.isnull(c) and not c=='N/A':
                return c

    if (
        pd.isnull(mdfr['k13_name_match']) and
        'Majaess IR cluster' in mdfr['why_not_in_k13']
    ):
        for c in cluster.split(','):
            if not pd.isnull(c) and not c=='N/A':
                return c


def get_k13_df():

    # nb. this is the 3006 row table with determined parameters. full
    # kharchenko catalog is like 3700, which is the index.
    getfile = os.path.join(clusterdatadir,'Kharchenko_2013_MWSC.vot')
    vot = parse(getfile)
    tab = vot.get_first_table().to_table()
    k13 = tab.to_pandas()
    del tab
    k13['Name'] = k13['Name'].str.decode('utf-8')
    k13['Type'] = k13['Type'].str.decode('utf-8')
    k13['MWSC'] = k13['MWSC'].str.decode('utf-8')

    return k13


def get_k13_name_match(task):
    """
    call in for loop, or in map, once you have cdips_df['cluster'] type column

    args:
        originalname : name from arbitrary reference of cluster
        target_ra : in decimal deg
        target_dec
        reference: source of cluster membership
        k13: dataframe of Kharchenko+2013 result

    returns:
        (matchname, have_name_match, have_mwsc_id_match, is_known_asterism,
        not_in_k13, why_not_in_k13)
    """

    originalname, target_ra, target_dec, reference, k13 = task

    cluster = originalname
    print(42*'#')
    print(cluster)

    have_name_match = False
    have_mwsc_id_match = False
    is_known_asterism = False
    how_match = np.nan
    why_not_in_k13 = ''

    if pd.isnull(originalname):
        return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, True, 'no_cluster_name')

    if reference in ['Zari_2018_PMS','Zari_2018_UMS']:

        not_in_k13 = True
        is_zari_ums = False

        if 'Zari_2018_UMS' in reference:
            is_zari_ums = True
        is_zari_pms = False
        if 'Zari_2018_PMS' in reference:
            is_zari_pms = True

        if is_zari_pms:
            why_not_in_k13 = 'is_zari_pms'
        elif is_zari_ums:
            why_not_in_k13 = 'is_zari_ums'

        return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, not_in_k13, why_not_in_k13)

    special_references = ['Velez_2018_scoOB2', 'Kounkel_2018_Ori',
                          'CantatGaudin_2019_velaOB2']
    for r in special_references:
        if r in reference:
            # None of these works have meaningful matches in Kharchenko+2013.
            not_in_k13 = True
            why_not_in_k13 = 'gaia_supersedes'
            return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                    is_known_asterism, not_in_k13, why_not_in_k13)

    #
    # lowest name match priority: whatever KC2019 said!
    #
    if reference in ['Kounkel_2019'] and reference.startswith('kc19group_'):

        # Don't attempt Kharchenko matches for the new KC2019 groups.
        not_in_k13 = True
        why_not_in_k13 = 'kc19_gaia_supersedes'
        return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, not_in_k13, why_not_in_k13)


    is_gaia_member = False
    if 'Gulliver' in cluster and 'CantatGaudin_2018' in reference:
        is_gaia_member = True
        not_in_k13 = True
        why_not_in_k13 = 'is_gaia_member'
        return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, not_in_k13, why_not_in_k13)

    # Hyades: I checked in a distance sort that it's not in K13. Probably b/c
    #  it's very extended.
    # RSG7,8: Soubiran+2018 note "RSG 7 and RSG 8, two OCs recently found by
    #  Röser et al. (2016), are confirmed as two separate OCs very close in
    #  space and motion." So they were found after K13.
    known_missing_clusters = ['Hyades', 'RSG_7', 'RSG_8', 'RSG_1', "RSG_5"]

    for c in cluster.split(','):
        if c in known_missing_clusters:
            not_in_k13 = True
            why_not_in_k13 = 'known missing from K13'
            how_match = 'manual_check'
            return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                    is_known_asterism, not_in_k13, why_not_in_k13)

    #
    # special formatting cases:
    # * Pozzo 1 is Vel OB2
    # * Majaess50 is 10' from FSR0721, from D. Majaess (2012).
    # * Majaess58 is 6' from FSR0775, from D. Majaess (2012).
    # * Ditto for a few from CG2018...
    # * "Collinder 34" is a sub-cluster of the larger Collinder 33 nebula, not
    #   appreciably different from SAI-24 either. (Saurin+ 2015,
    #   https://arxiv.org/pdf/1502.00373.pdf)
    # * Some catalogs, eg., Cantat-Gaudin's, replaced "vdBergh-Hagen" with "BH"
    # * Coma Star Cluster in Coma Berenices is Melotte 111
    #
    if 'Pozzo_1' in cluster:
        cluster = 'Vel_OB2'
        how_match = 'manual_override'
    if 'Majaess 50' in cluster:
        cluster = 'FSR_0721'
        how_match = 'manual_override'
    if 'Majaess 58' in cluster:
        cluster = 'FSR_0775'
        how_match = 'manual_override'
    if 'Alessi_Teutsch_5' in cluster:
        # underscores mess up the duplicate name list
        cluster = "ASCC 118"
        how_match = 'manual_override'
    if 'Collinder 34' in cluster:
        cluster = 'Collinder_33'
        how_match = 'manual_override'
    if 'BH' in cluster:
        cluster = cluster.replace("BH","vdBergh-Hagen")
    if 'ComaBer' in cluster:
        cluster = cluster.replace('ComaBer','Melotte_111')
        how_match = 'manual_override'
    if 'CBER' in cluster:
        cluster = cluster.replace('CBER','Melotte_111')
        how_match = 'manual_override'
    if 'PLE' in cluster:
        cluster = cluster.replace('PLE','Melotte_22')
        how_match = 'manual_override'
    if 'ScoOB2' in cluster:
        cluster = cluster.replace('ScoOB2','Sco_OB2')
        how_match = 'string_match'
    if 'USCO' in cluster:
        cluster = cluster.replace('USCO','Sco_OB2')
        how_match = 'string_match'
    if "BDSB" in cluster:
        if ' ' not in cluster and "_" not in cluster:
            cluster = cluster.replace("BDSB","BDSB_")
            how_match = 'string_match'
    if "Casado Alessi 1" in cluster:
        cluster = "Alessi_1"
        how_match = 'manual_override'
    if "Coll140" in cluster:
        cluster = "Collinder_140"
        how_match = 'manual_override'
    if "alphaPer" in cluster:
        cluster = "Melotte_20"
        how_match = 'manual_override'
    if "Trump10" in cluster:
        cluster = "Trumpler_10"
        how_match = 'manual_override'
    if "Trump10" in cluster:
        cluster = cluster.replace("Trump10","Trumpler_10")
        how_match = 'manual_override'
    if "Trump02" in cluster:
        cluster = cluster.replace("Trump02","Trumpler_2")
        how_match = 'manual_override'
    if "PL8" in cluster:
        cluster = cluster.replace("PL8","Platais_8")
        how_match = 'manual_override'
    if cluster == "NGC2451":
        # Gaia Collaboration 2018 table 1a forgot the "A" (nb. from their
        # paper, it's definitely the one they considered).
        cluster = "NGC_2451A"
        how_match = 'manual_override'

    # types like: 'Aveni_Hunter_42,nan,Aveni_Hunter_42,nan' need to be recast
    # to e.g., 'Aveni-Hunter_42,nan,Aveni-Hunter_42,nan' 
    for m in re.finditer('[a-zA-Z]+_[a-zA-Z]+_[0-9]+', cluster):
        cluster = cluster.replace(
            m.group(0),
            m.group(0).split('_')[0]+"-"+'_'.join(m.group(0).split('_')[1:])
        )

    # types like: "Aveni Hunter 1,nan,Aveni Hunter 1,nan", or 'Alessi Teutsch
    # 10' need to follow the same "Alessi-Teutsch_10" syntax
    for m in re.finditer('[a-zA-Z]+\ [a-zA-Z]+\ [0-9]+', cluster):
        cluster = cluster.replace(
            m.group(0),
            m.group(0).split(' ')[0]+"-"+'_'.join(m.group(0).split(' ')[1:])
        )

    # reformat ESO clusters like "ESO 129 32" (D14) to "ESO_129-32" (K13 format)
    for m in re.finditer('ESO\ [0-9]+\ [0-9]+', cluster):
        cluster = cluster.replace(
            m.group(0),
            "ESO_"+'-'.join(m.group(0).split(' ')[1:])
        )

    # reformat ESO clusters like "ESO_129_32" (CG18) to "ESO_129-32" (K13 format)
    for m in re.finditer('ESO_[0-9]+_[0-9]+', cluster):
        cluster = cluster.replace(
            m.group(0),
            "ESO_"+'-'.join(m.group(0).split('_')[1:])
        )

    #
    # initial normal match: try matching against replacing spaces with
    # underscores
    #
    clustersplt = cluster.split(',')
    trystrs = []
    for c in clustersplt:
        trystrs.append(c)
        trystrs.append(c.replace(' ','_'))

    for trystr in trystrs:
        if trystr in nparr(k13['Name']):
            have_name_match = True
            name_match = trystr
            how_match = 'string_match'
            not_in_k13 = False
            why_not_in_k13 = ''
            break


    #
    # try if SIMBAD's name matcher has anything.
    #
    if not have_name_match:
        for c in clustersplt:
            try:
                res = Simbad.query_objectids(c)
            except requests.exceptions.ConnectionError:
                time.sleep(15)
                res = Simbad.query_objectids(c)

            try:
                resdf = res.to_pandas()
            except AttributeError:
                print('{}: simbad no matches'.format(c))
                continue

            resdf['ID'] = resdf['ID'].str.decode('utf-8')
            smatches = nparr(resdf['ID'])

            # some names have format 'Name M 42'
            clean_smatches = [s.lstrip('NAME ') for s in smatches]
            # some names have format '[KPS2012] MWSC 0531'
            for ix, s in enumerate(clean_smatches):
                strm = re.search("\[.*\]\ ", s)
                if strm is not None:
                    clean_smatches[ix] = s.lstrip(strm.group())
            # some names have variable length whitespace... e.g., 'NGC  2224'

            # first set of attempts: everything in clean matches (irrespective if
            # MWSC number exists)
            trystrs = []
            for _c in clean_smatches:
                trystrs.append(_c)
                trystrs.append(_c.replace(' ','_'))

            for trystr in trystrs:
                if trystr in nparr(k13['Name']):
                    have_name_match = True
                    how_match = 'SIMBAD_name_match'
                    name_match = trystr
                    break

            # only then: check if you have MWSC identifier.
            inds = ['MWSC' in _c for _c in clean_smatches]
            mwsc_match = nparr(clean_smatches)[inds]
            if len(mwsc_match) > 1:
                pass
            if len(mwsc_match) == 1:
                have_mwsc_id_match = True
                mwsc_id_match = mwsc_match[0].replace('MWSC ','')
            if len(mwsc_match) == 0:
                pass

            if have_mwsc_id_match:
                break

    #
    # if you got mwsc id above, use the index table to convert to a name
    #
    if have_mwsc_id_match and not have_name_match:

        Vizier.ROW_LIMIT = -1
        catalog_list = Vizier.find_catalogs('J/A+A/558/A53')
        catalogs = Vizier.get_catalogs(catalog_list.keys())
        t = catalogs[1]
        df_index = t.to_pandas()
        del t

        # columns are MWSC,Name,flag,Type,n_Type,Src,SType,N
        for c in df_index.columns:
            if c != 'N':
                df_index[c] = df_index[c].str.decode('utf-8')

        mapd = {
            "=": "cluster parameters are determined",
            ":": ("possibly this is a cluster, but parameters are not "
                  "determined"),
            "-": "this is not a cluster",
            "&": "duplicated/coincides with other cluster"
        }

        _k13 = df_index[df_index['MWSC'] == mwsc_id_match]

        have_name_match=True
        how_match = 'SIMBAD_MWSCID_match'
        name_match = _k13['Name'].iloc[0]

        flag = str(_k13['flag'].iloc[0])
        if flag in [":","-","&"]:
            is_in_index = True
            why_not_in_k13 = "K13index: "+mapd[flag]
        else:
            pass

    #
    # check against the larger kharchenko index, which includes the clusters
    # that they found dubious, or didn't report parameters for.
    #
    is_in_index = False
    if not have_name_match and not have_mwsc_id_match:

        Vizier.ROW_LIMIT = -1
        catalog_list = Vizier.find_catalogs('J/A+A/558/A53')
        catalogs = Vizier.get_catalogs(catalog_list.keys())
        t = catalogs[1]
        df_index = t.to_pandas()
        del t

        # columns are MWSC,Name,flag,Type,n_Type,Src,SType,N
        for c in df_index.columns:
            if c != 'N':
                df_index[c] = df_index[c].str.decode('utf-8')

        mapd = {
            "=": "cluster parameters are determined",
            ":": ("possibly this is a cluster, but parameters are not "
                  "determined"),
            "-": "this is not a cluster",
            "&": "duplicated/coincides with other cluster"
        }

        clustersplt = cluster.split(',')
        trystrs = []
        for c in clustersplt:
            trystrs.append(c)
            trystrs.append(c.replace(' ','_'))
        for trystr in trystrs:
            if trystr in nparr(df_index['Name']):
                have_name_match=True
                how_match = 'K13_index_table_string_match'
                name_match = trystr
                is_in_index = True

                _k13 = df_index.loc[df_index['Name'] == name_match]
                flag = str(_k13['flag'].iloc[0])

                if flag not in [":","-","&"]:
                    raise NotImplementedError('why do u match here, not '
                                              'earlier?')
                else:
                    why_not_in_k13 = "K13index: "+mapd[flag]

                break


    #
    # try searching K13 within circles of 5,10 arcminutes of the quoted
    # position. if more than 1 match, omit (to not get false name matches).  if
    # only 1 match, use the name.  (this introduces some false matches,
    # probably...)
    #
    if not have_name_match and not have_mwsc_id_match:

        ra,dec = target_ra, target_dec

        c = SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs')

        k13_c = SkyCoord(nparr(k13['RAJ2000']), nparr(k13['DEJ2000']),
                         frame='icrs', unit=(u.deg,u.deg))

        seps = k13_c.separation(c)

        CUTOFFS = [5*u.arcmin, 10*u.arcmin]

        for CUTOFF in CUTOFFS:
            cseps = seps < CUTOFF

            if len(cseps[cseps]) == 1:
                have_name_match=True
                how_match = 'K13_spatial_match_lt_{}arcmin'.format(CUTOFF.value)
                name_match = k13.loc[cseps, 'Name'].iloc[0]

            elif len(cseps[cseps]) > 1:
                print('got too many matches within {} arcminutes!'.
                      format(CUTOFF))
                pass

            else:
                pass

            if have_name_match:
                break

    #
    # if after all this, no match, check double cluster names list, and recall
    # this function
    #
    if not have_name_match and not have_mwsc_id_match:
        ddf = pd.read_csv(
            os.path.join(clusterdatadir, 'double_names_WEBDA_20190606.csv'),
            sep=';')
        for c in clustersplt:
            if len(ddf[ddf['cluster_name'] == c.replace('_',' ')]) == 1:
                adopted_name = (
                    ddf[ddf['cluster_name'] ==
                        c.replace('_',' ')].iloc[0]['adopted_name']
                )
                return get_k13_name_match((adopted_name, target_ra, target_dec,
                                           reference, k13))

    #
    # Check against known asterisms
    #
    is_known_asterism = False
    for c in clustersplt:
        # Baumgardt 1998.
        if c in ['Collinder 399', 'Upgren 1', 'NGC 1252', 'Melotte 227',
                 'NGC 1746']:
            is_known_asterism = True
            break

        if 'NGC' in c:
            getfile = '/nfs/phn12/ar0/H/PROJ/lbouma/cdips/data/cluster_data/Sulentic_1973_NGC_known_asterisms.vot'
            vot = parse(getfile)
            tab = vot.get_first_table().to_table()
            ddf = tab.to_pandas()
            del tab

            ngc_asterisms = nparr(ddf['NGC'])

            if c.startswith('NGC '):
                c = c.lstrip('NGC ')
            elif c.startswith('NGC_'):
                c = c.lstrip('NGC_')
            elif c.startswith('NGC'):
                c = c.lstrip('NGC')

            # https://en.wikipedia.org/wiki/NGC_2451
            if c.endswith('A'):
                c = c.rstrip('A')
            if c.endswith('B'):
                c = c.rstrip('B')

            try:
                if int(c) in ngc_asterisms:
                    is_known_asterism = True
                    break
            except ValueError:
                # e.g., "NGC 2467-east", or similiar bs
                pass

            # NGC 1252 was also by Baumgardt 1998.
            # identified by Kos+2018, MNRAS 480 5242-5259 as asterisms
            try:
                if int(c) in [1252, 6994, 7772, 7826]:
                    is_known_asterism = True
                    break
            except ValueError:
                # e.g., "NGC 2467-east"
                pass

    is_gagne_mg = False
    if 'Gagne' in reference:
        is_gagne_mg = True
        why_not_in_k13 = 'is_gagne_mg'

    is_oh_mg = False
    if 'Oh' in reference:
        is_oh_mg = True
        why_not_in_k13 = 'is_oh_mg'

    is_rizzuto_mg = False
    if 'Rizzuto' in reference:
        is_rizzuto_mg = True
        why_not_in_k13 = 'is_rizzuto_mg'

    is_bell32ori_mg = False
    if 'Bell_2017_32Ori' in reference:
        is_bell32ori_mg = True
        why_not_in_k13 = 'is_bell_mg'

    is_kraus_mg = False
    if 'Kraus' in reference:
        is_kraus_mg = True
        why_not_in_k13 = 'is_kraus_mg'

    not_in_k13 = False
    if (
        (is_gagne_mg or is_oh_mg or is_rizzuto_mg or is_bell32ori_mg or
         is_kraus_mg or is_in_index)
        and
        not have_name_match
    ):
        not_in_k13 = True
        name_match = np.nan

    #
    # finally, if we failed to get matches above, (e.g., for some of the IR
    # Majaess clusters), skip
    #
    for c in cluster.split(','):
        if "Majaess" in c:
            not_in_k13 = True
            why_not_in_k13 = 'Majaess IR cluster match missing in K13'
            how_match = 'majaess_flag'
            return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                    is_known_asterism, not_in_k13, why_not_in_k13)

    if have_name_match:

        return (name_match, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, not_in_k13, why_not_in_k13)

    #
    # lowest name match priority: whatever KC2019 said (even if there are
    # others)!
    #
    if 'Kounkel_2019' in reference:

        not_in_k13 = True
        why_not_in_k13 = 'kc19_gaia_lastmatch'
        return (np.nan, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, not_in_k13, why_not_in_k13)


    try:
        return (name_match, how_match, have_name_match, have_mwsc_id_match,
                is_known_asterism, not_in_k13, why_not_in_k13)
    except UnboundLocalError:
        print('got UnboundLocalError for {} ({})'.format(
            repr(cluster), repr(reference)))
        import IPython; IPython.embed()


if __name__ == "__main__":
    construct_unique_cluster_name_column()
