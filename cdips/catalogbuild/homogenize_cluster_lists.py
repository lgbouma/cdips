import matplotlib.pyplot as plt, pandas as pd, numpy as np
"""
goal: get Gaia DR2 IDs of stars in "clusters". these lists are then used to
define the CDIPS stellar sample.

see ../doc/list_of_cluster_member_lists.ods for an organized spreadsheet of the
different member lists
"""

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

from cdips.catalogbuild.open_cluster_xmatch_utils import (
    GaiaCollaboration2018_clusters_to_csv,
    Kharchenko2013_position_mag_match_Gaia,
    Dias2014_nbhr_gaia_to_nearestnbhr,
    KounkelCovey2019_clusters_to_csv,
    Kounkel2018_orion_to_csv
)
from cdips.catalogbuild.moving_group_xmatch_utils import (
    make_Gagne18_BANYAN_XI_GaiaDR2_crossmatch,
    make_Gagne18_BANYAN_XII_GaiaDR2_crossmatch,
    make_Gagne18_BANYAN_XIII_GaiaDR2_crossmatch,
    make_Kraus14_GaiaDR2_crossmatch,
    make_Rizzuto11_GaiaDR2_crossmatch,
    make_Luhman12_GaiaDR2_crossmatch,
    make_Bell17_GaiaDR2_crossmatch,
    make_Roser11_GaiaDR2_crossmatch,
    make_Oh17_GaiaDR2_crossmatch
)
from cdips.catalogbuild.star_forming_rgn_xmatch_utils import (
    Zari18_stars_to_csv,
    VillaVelez18_check,
    CantatGaudin2019_velaOB2_to_csv
)

if socket.gethostname() == 'phtess2':
    clusterdatadir = '/home/lbouma/proj/cdips/data/cluster_data/'
elif socket.gethostname() == 'brik':
    clusterdatadir = '/home/luke/Dropbox/proj/cdips/data/cluster_data/'
else:
    raise NotImplementedError

def main():

    # OCs
    GaiaCollab18 = 0
    K13 = 0
    D14 = 0
    KC19 = 0
    K18 = 0
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

    do_merge_MG_catalogs = 0 #NOTE: YA, U NEED TO DO THIS.
    do_merge_OC_catalogs = 0
    do_merge_OC_MG_catalogs = 0
    do_final_merge = 0

    if GaiaCollab18:
        GaiaCollaboration2018_clusters_to_csv()
    if K13:
        Kharchenko2013_position_mag_match_Gaia()
    if D14:
        Dias2014_nbhr_gaia_to_nearestnbhr()
    if KC19:
        KounkelCovey2019_clusters_to_csv()
    if K18:
        Kounkel2018_orion_to_csv()

    if do_BANYAN_XI:
        make_Gagne18_BANYAN_XI_GaiaDR2_crossmatch()
    if do_BANYAN_XII:
        make_Gagne18_BANYAN_XII_GaiaDR2_crossmatch()
    if do_BANYAN_XIII:
        make_Gagne18_BANYAN_XIII_GaiaDR2_crossmatch()
    if do_Kraus_14:
        make_Kraus14_GaiaDR2_crossmatch()
    if do_Roser_11:
        make_Roser11_GaiaDR2_crossmatch()
    if do_Bell_17:
        make_Bell17_GaiaDR2_crossmatch()
    if do_Oh17:
        make_Oh17_GaiaDR2_crossmatch()
    if do_Rizzuto11:
        make_Rizzuto11_GaiaDR2_crossmatch()
    if do_Zari18:
        Zari18_stars_to_csv()
    if do_VillaVelez18:
        VillaVelez18_check()
    if do_CG19_vela:
        CantatGaudin2019_velaOB2_to_csv()

    if do_merge_MG_catalogs:
        merge_MG_catalogs()
    if do_merge_OC_catalogs:
        merge_OC_catalogs()
    if do_merge_OC_MG_catalogs:
        merge_OC_MG_catalogs()
    if do_final_merge:
        final_merge()


def fname_to_reference(fname):

    #FIXME: need to add the new ones here!!!
    d = {
        'MATCHED_Gagne_2018_BANYAN_XI_GaiaDR2_crossmatched.csv':'Gagne_2018_BANYAN_XI',
        'MATCHED_Gagne_2018_BANYAN_XII_GaiaDR2_crossmatched.csv':'Gagne_2018_BANYAN_XII',
        'MATCHED_Gagne_2018_BANYAN_XIII_GaiaDR2_crossmatched.csv':'Gagne_2018_BANYAN_XIII',
        'MATCHED_Oh_2017_clustering_GaiaDR2_crossmatched.csv':'Oh_2017_clustering',
        'MATCHED_Rizzuto_11_table_1_ScoOB2_members.csv':'Rizzuto_2011_ScoOB2',
        'Bell17_table_32Ori_MATCHES_GaiaDR2.csv':'Bell_2017_32Ori',
        'Kraus_14_table_2_TucanaHorologiumMG_members_MATCHES_GaiaDR2.csv':'Kraus_2014_TucHor',
        'Roser11_table_1_Hyades_members_MATCHES_GaiaDR2.csv':'Roser_2011_Hyades',
        'CantatGaudin_2018_table2_cut_only_source_cluster.csv':'CantatGaudin_2018',
        'GaiaCollaboration2018_616_A10_tablea1a_within_250pc_cut_only_source_cluster.csv':'GaiaCollaboration2018_tab1a',
        'GaiaCollaboration2018_616_A10_tablea1b_beyond_250pc_cut_only_source_cluster.csv':'GaiaCollaboration2018_tab1b',
        'MWSC_Gaia_matched_concatenated.csv':'Kharchenko2013',
        'Dias14_seplt5arcsec_Gdifflt2.csv':'Dias2014',
        'Zari_2018_ums_tab_cut_only_source_cluster_MATCH.csv':'Zari_2018_UMS',
        'Zari_2018_pms_tab_cut_only_source_cluster_MATCH.csv':'Zari_2018_PMS',
        'VillaVelez_2018_DR2_PreMainSequence_MATCH.csv':'VillaVelez_2018'
    }

    return d[fname]


def combine_it(rows):
    # given set of rows w/ duplicate source_ids, give a single row.

    source_id = rows['source_id']

    assocs, references, ext_catalog_names, dists = [],[],[],[]
    for ix, r in rows.iterrows():
        assocs.append(r['assoc'])
        references.append(r['reference'])
        ext_catalog_names.append(r['ext_catalog_name'])
        dists.append(r['dist'])

    outrow = pd.DataFrame( {
        'source_id':int(np.unique(source_id)),
        'assoc':','.join(list(map(str,assocs))),
        'reference':','.join(list(map(str,references))),
        'ext_catalog_name':','.join(list(map(str,ext_catalog_names))),
        'dist':','.join(list(map(str,dists))) },
        index=[0]
    )

    return outrow


def combine_it_OC(rows):
    # given set of rows w/ duplicate source_ids, give a single row.

    source_id = rows['source_id']

    assocs, references, ext_catalog_names, dists = [],[],[],[]
    for ix, r in rows.iterrows():
        assocs.append(r['cluster'])
        references.append(r['reference'])
        ext_catalog_names.append(r['ext_catalog_name'])
        dists.append(r['dist'])

    outrow = pd.DataFrame( {
        'source_id':int(np.unique(source_id)),
        'cluster':','.join(list(map(str,assocs))),
        'reference':','.join(list(map(str,references))),
        'ext_catalog_name':','.join(list(map(str,ext_catalog_names))),
        'dist':','.join(list(map(str,dists))) },
        index=[0]
    )

    return outrow


def combine_it_final(rows):
    # given set of rows w/ duplicate source_ids, give a single row.

    source_id = rows['source_id']

    assocs, references, ext_catalog_names, dists = [],[],[],[]
    for ix, r in rows.iterrows():
        assocs.append(r['cluster'])
        references.append(r['reference'])
        ext_catalog_names.append(r['ext_catalog_name'])
        dists.append(r['dist'])

    outrow = pd.DataFrame( {
        'source_id':int(np.unique(source_id)),
        'ra':np.float64(rows['ra'].iloc[0]),
        'dec':np.float64(rows['dec'].iloc[0]),
        'pmra':np.float64(rows['pmra'].iloc[0]),
        'pmdec':np.float64(rows['pmdec'].iloc[0]),
        'parallax':np.float64(rows['parallax'].iloc[0]),
        'phot_g_mean_mag':np.float64(rows['phot_g_mean_mag'].iloc[0]),
        'phot_bp_mean_mag':np.float64(rows['phot_bp_mean_mag'].iloc[0]),
        'phot_rp_mean_mag':np.float64(rows['phot_rp_mean_mag'].iloc[0]),
        'cluster':','.join(list(map(str,assocs))),
        'reference':','.join(list(map(str,references))),
        'ext_catalog_name':','.join(list(map(str,ext_catalog_names))),
        'dist':','.join(list(map(str,dists))) },
        index=[0]
    )

    return outrow

def merge_MG_catalogs():
    """
    merged MG member catalog will have:

        source_id: GAIA-DR2 source id
        assoc: association name
        reference: reference(s) from which I got the membership.
            comma-separated string.
        ext_catalog_name: comma-separated string with identifer appropriate to
            the external catalog, in "reference" column.
        dist: comma-separated string giving spatial distance of the match, if
            available, in degrees.
    """
    datadir = os.path.join(clusterdatadir, 'moving_groups')
    mgfiles = glob(os.path.join(datadir,'*MATCH*.csv'))
    outpath = os.path.join(datadir,'MOVING_GROUPS_MERGED.csv')

    dfl = []
    for mgfile in mgfiles:

        df = pd.read_csv(mgfile)

        sdf = pd.DataFrame()

        #
        # assign source_id
        #
        if 'Zari_2018' in mgfile:
            sdf['source_id'] = df['source']
        else:
            sdf['source_id'] = df['source_id']

        #
        # assign assoc
        #
        if ('assoc' not in df.columns and
            not 'Rizzuto_11' in mgfile and
            not 'Zari_2018' in mgfile and
            not 'VillaVelez_2018' in mgfile
           ):
            raise AssertionError
        elif 'assoc' not in df.columns and (
            'Rizzuto_11' in mgfile or 'VillaVelez_2018' in mgfile
        ):
            sdf['assoc'] = 'ScoOB2'
        elif 'assoc' not in df.columns and 'Zari_2018' in mgfile:
            sdf['assoc'] = 'N/A'
        elif 'Oh_2017' in mgfile:
            sdf['assoc'] = np.array(['Oh_'+str(a) for a in df['assoc']])
        else:
            sdf['assoc'] = df['assoc'].astype(str)

        #
        # assign reference
        #
        sdf['reference'] = fname_to_reference(os.path.basename(mgfile))

        #
        # assign external catalog name
        #
        if 'Zari_2018' in mgfile:
            sdf['ext_catalog_name'] = df['source']
        elif 'VillaVelez_2018' in mgfile:
            sdf['ext_catalog_name'] = df['source_id']
        else:
            sdf['ext_catalog_name'] = df['name']

        #
        # assign distance if xmatched
        #
        if 'Zari_2018' in mgfile or 'VillaVelez_2018' in mgfile:
            sdf['dist'] = 0
        else:
            sdf['dist'] = df['dist']

        dfl.append(sdf)

    # now need to merge the assoc, reference, ext_catalog_name, and dist
    # columns.
    bdf = pd.concat(dfl)

    bdf = bdf.sort_values('source_id')

    theserows = []
    for usource in np.unique(bdf['source_id']):
        rows = bdf[bdf['source_id']==usource]
        theserows.append(combine_it(rows))
    bdf_agg = pd.concat(theserows)

    # below pattern should work, but it did not!
    # bdf = bdf.groupby('source_id')
    # bdf_agg = bdf.agg(combine_it)

    bdf_agg.to_csv(outpath, index=False, sep=';')
    print('made {}'.format(outpath))


def merge_OC_catalogs():
    """
    merged OC member catalog will have:

        source_id: GAIA-DR2 source id
        cluster: cluster name
        reference: reference(s) from which I got the membership.
            comma-separated string.
        ext_catalog_name: comma-separated string with identifer appropriate to
            the external catalog, in "reference" column (if gaia catalog, it's
            a repeat of the source_id).
        dist: comma-separated string giving spatial distance of the match, if
            available, in degrees.
    """
    fnames = [
        'CantatGaudin_2018_table2_cut_only_source_cluster.csv',
        'GaiaCollaboration2018_616_A10_tablea1a_within_250pc_cut_only_source_cluster.csv',
        'GaiaCollaboration2018_616_A10_tablea1b_beyond_250pc_cut_only_source_cluster.csv',
        'MWSC_Gaia_matched_concatenated.csv',
        'Dias14_seplt5arcsec_Gdifflt2.csv'
    ]

    ocfiles = [os.path.join(clusterdatadir, f) for f in fnames]

    dfl = []
    for ocfile in ocfiles:

        df = pd.read_csv(ocfile)

        if 'Dias14' in ocfile:
            # need to get names b/c i messed up earlier
            getfile = '/home/luke/local/cdips/catalogs/Dias_2014_prob_gt_50_pct_vizier.vot'
            vot = parse(getfile)
            tab = vot.get_first_table().to_table()
            ddf = tab.to_pandas()
            del tab
            ddf['Cluster'] = ddf['Cluster'].str.decode('utf-8')
            ddf['UCAC4'] = ddf['UCAC4'].str.decode('utf-8')

            df = df.merge(ddf, how='left', left_on='ucac_id', right_on='UCAC4')

        sdf = pd.DataFrame()

        if 'CantatGaudin' in ocfile or 'GaiaCollab' in ocfile:
            sdf['source_id'] = df['source']
        elif 'MWSC' in ocfile:
            sdf['source_id'] = df['gaia_dr2_match_id']
        elif 'Dias14' in ocfile:
            sdf['source_id'] = df['source_id']

        if 'CantatGaudin' in ocfile or 'GaiaCollab' in ocfile:
            # saved binary strings that are then read as strings...
            sdf['cluster'] = df['cluster'].apply(
                lambda x: x.rstrip('\''))
            sdf['cluster'] = sdf['cluster'].apply(
                lambda x: x.lstrip('b\''))
        elif 'MWSC' in ocfile:
            sdf['cluster'] = df['cname']
        elif 'Dias14' in ocfile:
            sdf['cluster'] = df['Cluster']

        sdf['reference'] = fname_to_reference(os.path.basename(ocfile))

        if 'CantatGaudin' in ocfile or 'GaiaCollab' in ocfile:
            sdf['ext_catalog_name'] = df['source']
        elif 'MWSC' in ocfile:
            sdf['ext_catalog_name'] = df['k13_tmass_oids']
        elif 'Dias14' in ocfile:
            sdf['ext_catalog_name'] = df['ucac_id']

        if 'CantatGaudin' in ocfile or 'GaiaCollab' in ocfile:
            sdf['dist'] = 0
        elif 'MWSC' in ocfile:
            sdf['dist'] = df['dist_deg']
        elif 'Dias14' in ocfile:
            sdf['dist'] = df['dist']

        dfl.append(sdf)
        del df, sdf

    # now need to merge the assoc, reference, ext_catalog_name, and dist
    # columns.
    bdf = pd.concat(dfl)

    bdf = bdf.sort_values('source_id')

    bdf = bdf[bdf['source_id'] != -1]

    theserows = []
    usources, inv, cnts = np.unique(bdf['source_id'], return_inverse=True,
                                    return_counts=True)
    n_usources = len(usources)
    cnts_bdf = cnts[inv]

    ind_sings = cnts_bdf == 1
    ind_mults = cnts_bdf > 1

    df_sings = bdf.ix[ind_sings]

    sing_path = os.path.join(datadir,'OPEN_CLUSTERS_SINGLES.csv')
    df_sings.to_csv(sing_path, index=False, sep=';')
    print('made {}'.format(sing_path))

    df_mults = bdf.ix[ind_mults]
    del bdf, df_sings, dfl
    print('beginning aggregation...')
    df_mults_agg = df_mults.groupby('source_id').apply(combine_it_OC)
    mult_path = os.path.join(datadir,'OPEN_CLUSTERS_MULTS.csv')
    df_mults_agg.to_csv(mult_path, index=False, sep=';')

    # for each unique gaia source id, make the row.
    df_sings = pd.read_csv(sing_path, sep=';')
    outdf = pd.concat((df_sings,df_mults_agg))
    outdf = outdf.sort_values('source_id')

    outpath = os.path.join(datadir,'OPEN_CLUSTERS_MERGED.csv')
    outdf.to_csv(outpath, index=False, sep=';')
    print('made {}'.format(outpath))


def merge_OC_MG_catalogs():

    datadir = '/home/luke/Dropbox/proj/cdips/data/cluster_data/'

    ocs = pd.read_csv(datadir+'OPEN_CLUSTERS_MERGED.csv',
                      sep=';')
    mgs = pd.read_csv(datadir+'moving_groups/MOVING_GROUPS_MERGED.csv',
                      sep=';')
    mgs = mgs.rename(index=str, columns={"assoc":"cluster"})

    outdf = pd.concat((ocs, mgs))

    outpath = '/home/luke/local/cdips/catalogs/OC_MG_MERGED.csv'
    outdf.to_csv(outpath, sep=';', index=False)
    print('made {}'.format(outpath))

    # upload this csv file to gaia archive, then use it to run a crossmatch
    # that gets things like G_Rp band photometry.
    outpath = '/home/luke/local/cdips/catalogs/oc_mg_sources.csv'
    outdf['source_id'].to_csv(outpath, sep=',', index=False)
    print('made {}'.format(outpath))


def final_merge(vnum='0.3'):
    """
    merge Gaia-DR2 info and source/reference info into one file.
    NOTE: for some reason ~half of the sources in OC_MG_MERGED.csv are lost
    in this step. scary, and unclear why...!
    """

    datadir = '/home/luke/local/cdips/catalogs/'
    #datadir = '/home/lbouma/local/cdips/catalogs/'

    ocmg_df = pd.read_csv(datadir+'OC_MG_MERGED.csv', sep=';')

    vot = parse(datadir+'cdips_v{}-result.vot.gz'.format(vnum))
    tab = vot.get_first_table().to_table()
    df = tab.to_pandas()

    outdf = df.merge(ocmg_df, how='left', on='source_id')
    outdf['source_id'] = outdf['source_id'].astype(np.int64)

    # there are overlaps from the OC/MG concatenation. sources that only show
    # up once can be staged to FINAL_SINGLES.csv. those that show up twice or
    # more need to be combined ("combine_it_final").
    usources, inv, cnts = np.unique(outdf['source_id'], return_inverse=True,
                                    return_counts=True)
    n_usources = len(usources)
    cnts_outdf = cnts[inv]

    ind_sings = cnts_outdf == 1
    ind_mults = cnts_outdf > 1

    df_sings = outdf.iloc[ind_sings]

    sing_path = os.path.join(datadir,'FINAL_SINGLES.csv')
    df_sings.to_csv(sing_path, index=False, sep=';')
    print('made {}'.format(sing_path))

    df_mults = outdf.iloc[ind_mults]
    del outdf, df_sings

    print('beginning aggregation...')
    df_mults_agg = df_mults.groupby('source_id').apply(combine_it_final)
    mult_path = os.path.join(datadir,'FINAL_MULTS.csv')
    df_mults_agg.to_csv(mult_path, index=False, sep=';')

    # for each unique gaia source id, make the row.
    df_sings = pd.read_csv(sing_path, sep=';')
    outdf = pd.concat((df_sings,df_mults_agg))
    outdf = outdf.sort_values('source_id')

    outpath = datadir+'OC_MG_FINAL.csv'
    outdf.to_csv(outpath, sep=';', index=False)
    print('made {}'.format(outpath))

    outpath = datadir+'OC_MG_FINAL_GaiaRp_lt_16_v{}.csv'.format(vnum)
    outdf[outdf['phot_rp_mean_mag']<16].to_csv(outpath, sep=';', index=False)
    print('made {}'.format(outpath))


if __name__ == "__main__":
    main()
