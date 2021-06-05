"""
Once the CSV files of source_ids, ages, and references are assembled,
concatenate and merge them.

Date: May 2021.

Background: Created for v0.5 target catalog merge, to simplify life.

Contents:
    AGE_LOOKUP: manual lookupdictionary of common cluster ages.
    get_target_catalog
    assemble_initial_source_list
    verify_target_catalog
"""

import numpy as np, pandas as pd
import os
from glob import glob

from cdips.utils.gaiaqueries import (
    given_source_ids_get_gaia_data, given_votable_get_df
)

from cdips.paths import DATADIR, LOCALDIR
clusterdatadir = os.path.join(DATADIR, 'cluster_data')
localdir = LOCALDIR

agefmt = lambda x: np.round(np.log10(x),2)

AGE_LOOKUP = {
    # Rizzuto+17 clusters. Hyades&Praesepe age from Brandt and Huang 2015,
    # following Rebull+17's logic.
    'Hyades': agefmt(8e8),
    'Praesepe': agefmt(8e8),
    'Pleiades': agefmt(1.25e8),
    'PLE': agefmt(1.25e8),
    'Upper Sco': agefmt(1.1e7),
    'Upper Scorpius': agefmt(1.1e7),
    'Upper Sco Lit.': agefmt(1.1e7),
    # Furnkranz+19
    'ComaBer': agefmt(4e8),
    'ComaBerNeighborGroup': agefmt(7e8),
    # EsplinLuhman, and other
    'Taurus': agefmt(5e6),
    'TAU': agefmt(5e6),
    # assorted
    'PscEri': agefmt(1.25e8),
    'LCC': agefmt(1.1e7),
    'ScoOB2': agefmt(1.1e7),
    'ScoOB2_PMS': agefmt(1.1e7),
    'ScoOB2_UMS': agefmt(1.1e7),
    # Meingast 2021
    'Blanco 1': agefmt(140e6),
    'IC 2391': agefmt(36e6),
    'IC 2602': agefmt(40e6),
    'Melotte 20': agefmt(87e6),
    'Melotte 22': agefmt(125e6),
    'NGC 2451A': agefmt(44e6),
    'NGC 2516': agefmt(170e6),
    'NGC 2547': agefmt(30e6),
    'NGC 7092': agefmt(310e6),
    'Platais 9': agefmt(100e6),
    # Gagne2018 moving groups
    '118TAU': agefmt(10e6),
    'ABDMG': agefmt(149e6),
    'CAR': agefmt(45e6),
    'CARN': agefmt(200e6),
    'CBER': agefmt(400e6),
    'COL': agefmt(42e6),
    'EPSC': agefmt(4e6),
    'ETAC': agefmt(8e6),
    'HYA': agefmt(750e6),
    'IC2391': agefmt(50e6),
    'IC2602': agefmt(40e6),
    'LCC': agefmt(15e6),
    'OCT': agefmt(35e6),
    'PL8': agefmt(60e6),
    'ROPH': agefmt(2e6),
    'THA': agefmt(45e6),
    'THOR': agefmt(22e6),
    'Tuc-Hor': agefmt(22e6),
    'TWA': agefmt(10e6),
    'UCL': agefmt(16e6),
    'CRA': agefmt(10e6),
    'UCRA': agefmt(10e6),
    'UMA': agefmt(414e6),
    'USCO': agefmt(10e6),
    'XFOR': agefmt(500e6),
    '{beta}PMG': agefmt(24e6),
    'BPMG': agefmt(24e6),
    # CantatGaudin2019 vela
    'cg19velaOB2_pop1': agefmt(46e6),
    'cg19velaOB2_pop2': agefmt(44e6),
    'cg19velaOB2_pop3': agefmt(40e6),
    'cg19velaOB2_pop4': agefmt(35e6),
    'cg19velaOB2_pop5': agefmt(25e6),
    'cg19velaOB2_pop6': agefmt(20e6),
    'cg19velaOB2_pop7': agefmt(11e6),
}

def assemble_initial_source_list(catalog_vnum):
    """
    Given LIST_OF_LISTS_STARTER_v0.5.csv , exported from
    /doc/list_of_cluster_member_lists.ods, clean and concatenate the cluster
    members. Flatten the resulting list on source_ids, joining the cluster,
    age, and bibcode columns into comma-separated strings.
    """

    metadf = pd.read_csv(
        os.path.join(clusterdatadir, 'LIST_OF_LISTS_STARTER_V0.6.csv')
    )
    metadf['bibcode'] = metadf.ads_link.str.extract("abs\/(.*)\/")

    N_stars_in_lists = []
    Nstars_with_age_in_lists = []
    dfs = []

    # for each table, concatenate into a dataframe of source_id, cluster,
    # log10age ("age").
    for ix, r in metadf.iterrows():

        print(79*'-')
        print(f'Beginning {r.reference_id}...')

        csvpath = os.path.join(clusterdatadir, r.csv_path)
        assert os.path.exists(csvpath)

        df = pd.read_csv(csvpath)

        df['reference_id'] = r.reference_id
        df['reference_bibcode'] = r.bibcode
        if 'HATSandHATNcandidates' in r.reference_id:
            df['reference_bibcode'] = 'JoelHartmanPrivComm'

        colnames = df.columns

        #
        # every CSV file needs a Gaia DR2 "source_id" column
        #
        if "source" in colnames:
            df = df.rename(
                columns={"source":"source_id"}
            )

        #
        # every CSV file needs a "cluster name" name column
        #
        if "assoc" in colnames:
            df = df.rename(
                columns={"assoc":"cluster"} # moving groups
            )

        colnames = df.columns

        if "cluster" not in colnames:
            msg = (
                f'WRN! for {r.reference_id} did not find "cluster" column. '+
                f'Appending the reference_id ({r.reference_id}) as the cluster ID.'
            )
            print(msg)

            df['cluster'] = r.reference_id

        #
        # every CSV file needs an "age" column, which can be null, but
        # preferably is populated.
        #
        if "age" not in colnames:

            if r.reference_id in [
                'CantatGaudin2018a', 'CantatGaudin2020a', 'CastroGinard2020',
                'GaiaCollaboration2018lt250', 'GaiaCollaboration2018gt250'
            ]:

                # get clusters and ages from CG20b; use them as the reference
                cg20bpath = os.path.join(
                    clusterdatadir,
                    "v05/CantatGaudin20b_cut_cluster_source_age.csv"
                )
                df_cg20b = pd.read_csv(cg20bpath)
                cdf_cg20b = df_cg20b.drop_duplicates(subset=['cluster','age'])[
                    ['cluster', 'age']
                ]

                # cleaning steps
                if r.reference_id == 'CastroGinard2020':
                    df['cluster'] = df.cluster.str.replace('UBC', 'UBC_')

                elif r.reference_id in [
                    'GaiaCollaboration2018lt250',
                    'GaiaCollaboration2018gt250'
                ]:
                    df['cluster'] = df.cluster.str.replace('NGC0', 'NGC_')
                    df['cluster'] = df.cluster.str.replace('NGC', 'NGC_')
                    df['cluster'] = df.cluster.str.replace('IC', 'IC_')
                    df['cluster'] = df.cluster.str.replace('Stock', 'Stock_')
                    df['cluster'] = df.cluster.str.replace('Coll', 'Collinder_')
                    df['cluster'] = df.cluster.str.replace('Trump02', 'Trumpler_2')
                    df['cluster'] = df.cluster.str.replace('Trump', 'Trumpler_')

                _df = df.merge(cdf_cg20b, how='left', on=['cluster'])
                assert len(_df) == len(df)

                df['age'] = _df['age']
                print(
                    f'For {r.reference_id} got {len(df[~pd.isnull(df.age)])}/{len(df)} finite ages via CantatGaudin2020b crossmatch on cluster ID.'
                )

                del _df

            elif (
                ('Zari2018' in r.reference_id)
                or
                ('Oh2017' in r.reference_id)
                or
                ('Ujjwal2020' in r.reference_id)
                or
                ('CottenSong' in r.reference_id)
                or
                ('HATSandHATNcandidates' in r.reference_id)
                or
                ('SIMBAD' in r.reference_id)
                or
                ('Gagne2018' in r.reference_id)
            ):
                age = np.ones(len(df))*np.nan
                df['age'] = age

            else:
                age_mapper = lambda k: AGE_LOOKUP[k]
                age = df.cluster.apply(age_mapper)
                df['age'] = age

        N_stars_in_lists.append(len(df))
        Nstars_with_age_in_lists.append(len(df[~pd.isnull(df.age)]))
        dfs.append(df)

        assert (
            'source_id' in df.columns
            and
            'cluster' in df.columns
            and
            'age' in df.columns
        )

    metadf["Nstars"] = N_stars_in_lists
    metadf["Nstars_with_age"] = Nstars_with_age_in_lists

    # concatenation.
    nomagcut_df = pd.concat(dfs)
    assert np.sum(metadf.Nstars) == len(nomagcut_df)

    # clean ages
    sel = (nomagcut_df.age == -np.inf)
    nomagcut_df.loc[sel,'age'] = np.nan
    nomagcut_df['age'] = np.round(nomagcut_df.age,2)

    #
    # merge duplicates, and ','-join the cluster id strings, age values
    #
    scols = ['source_id', 'cluster', 'age', 'reference_id', 'reference_bibcode']
    nomagcut_df = nomagcut_df[scols].sort_values(by='source_id')

    for c in nomagcut_df.columns:
        nomagcut_df[c] = nomagcut_df[c].astype(str)

    print(79*'-')
    print('Beginning aggregation (takes ~2-3 minutes for v0.5)...')
    _ = nomagcut_df.groupby('source_id')
    df_agg = _.agg({
        "cluster": list,
        "age": list,
        "reference_id": list,
        "reference_bibcode": list
    })

    u_sourceids = np.unique(nomagcut_df.source_id)
    N_sourceids = len(u_sourceids)
    assert len(df_agg) == N_sourceids

    df_agg["source_id"] = df_agg.index

    # turn the lists to comma separated strings.
    outdf = pd.DataFrame({
        "source_id": df_agg.source_id,
        "cluster": [','.join(map(str, l)) for l in df_agg['cluster']],
        "age": [','.join(map(str, l)) for l in df_agg['age']],
        "mean_age": [np.round(np.nanmean(np.array(l).astype(float)),2) for l in df_agg['age']],
        "reference_id": [','.join(map(str, l)) for l in df_agg['reference_id']],
        "reference_bibcode": [','.join(map(str, l)) for l in df_agg['reference_bibcode']],
    })

    outpath = os.path.join(
        clusterdatadir, f'list_of_lists_keys_paths_assembled_v{catalog_vnum}.csv'
    )
    metadf.to_csv(outpath, index=False)
    print(f'Made {outpath}')

    outpath = os.path.join(
        clusterdatadir, f'cdips_targets_v{catalog_vnum}_nomagcut.csv'
    )
    outdf.to_csv(outpath, index=False)
    print(f'Made {outpath}')


def verify_target_catalog(df, metadf):
    """
    Check that each entry in the (pre magnitude cut) target catalog has
    a source_id that matches the original catalog. (i.e., ensure that no
    int/int64/str lossy conversion bugs have happened).
    """

    print(79*'-')
    print('Beginning verification...')
    print(79*'-')

    for ix, r in metadf.sort_values('Nstars').iterrows():

        print(f'{r.reference_id} (Nstars={r.Nstars})...')

        sel = df.reference_id.str.contains(r.reference_id)
        df_source_ids = np.array(df.loc[sel, 'source_id']).astype(np.int64)

        csvpath = os.path.join(clusterdatadir, r.csv_path)
        df_true = pd.read_csv(csvpath)
        if 'source_id' not in df_true.columns:
            df_true = df_true.rename(columns={"source":"source_id"})

        true_source_ids = (
            np.unique(np.array(df_true.source_id).astype(np.int64))
        )

        np.testing.assert_array_equal(
            np.sort(df_source_ids), np.sort(true_source_ids)
        )

    print('Verified that the pre-mag cut target catalog has source_ids that '
          'correctly match the original. ')
    print(79*'-')



def verify_gaia_xmatch(df, gdf, metadf):
    """
    Check that each entry in the target catalog has a Gaia xmatch source_id
    that matches the original catalog. For any that do not, understand why not.
    """

    print(79*'-')
    print('Beginning Gaia xmatch verification...')
    print(79*'-')

    gdf_source_ids = np.unique(np.array(gdf.source_id).astype(np.int64))

    for ix, r in metadf.sort_values('Nstars').iterrows():

        print(f'{r.reference_id} (Nstars={r.Nstars})...')

        sel = df.reference_id.str.contains(r.reference_id)
        df_source_ids = np.array(df.loc[sel, 'source_id']).astype(np.int64)

        int1d = np.intersect1d(df_source_ids, gdf_source_ids)
        if not len(int1d) == len(df_source_ids):
            msg = f'\tWRN! {r.reference_id} only got {len(int1d)} Gaia xmatches.'
            print(msg)

            if 'NASAExoArchive' in r.reference_id:
                csvpath = os.path.join(clusterdatadir, r.csv_path)
                df_true = pd.read_csv(csvpath)
                missing = df_source_ids[
                    ~np.in1d(df_source_ids, gdf_source_ids)
                ]
                # NOTE: should not be raised.

    print('Verified that the pre-mag cut target catalog has source_ids that '
          'match the original (or close enough). ')
    print(79*'-')



def get_target_catalog(catalog_vnum, VERIFY=1):
    """
    1. Assemble the target catalog (down to arbitrary brightness; i.e, just
    clean and concatenate).
    2. Manually async query the Gaia database based on those source_ids.
    3. Verify the result, and merge and write it.
    """

    csvpath = os.path.join(
        clusterdatadir, f'cdips_targets_v{catalog_vnum}_nomagcut.csv'
    )
    if not os.path.exists(csvpath):
        assemble_initial_source_list(catalog_vnum)

    df = pd.read_csv(csvpath)

    # made by assemble_initial_source_list above.
    metapath = os.path.join(
        clusterdatadir, f'list_of_lists_keys_paths_assembled_v{catalog_vnum}.csv'
    )
    metadf = pd.read_csv(metapath)

    if VERIFY:
        # one-time verification
        verify_target_catalog(df, metadf)

    # e.g., cdips_v05_1-result.vot.gz
    votablepath = os.path.join(
        clusterdatadir, f'cdips_v{str(catalog_vnum).replace(".","")}_1-result.vot.gz'
    )
    if not os.path.exists(votablepath):
        temppath = os.path.join(clusterdatadir, f'v{str(catalog_vnum).replace(".","")}_sourceids.csv')
        print(f'Wrote {temppath}')
        df['source_id'].to_csv(
            temppath,
            index=False
        )
        querystr = (
            "SELECT top 2000000 g.source_id, g.ra, g.dec, g.parallax, "+
            "g.parallax_error, g.pmra, g.pmdec, g.phot_g_mean_mag, "+
            "g.phot_rp_mean_mag, g.phot_bp_mean_mag FROM "+
            f"user_lbouma.v{str(catalog_vnum).replace('.','')}_sourceids as u, gaiadr2.gaia_source AS g WHERE "+
            "u.source_id=g.source_id "
        )
        print('Now you must go to https://gea.esac.esa.int/archive/, login, and run')
        print(querystr)
        assert 0
        # # NOTE: the naive implementation below doesn't work, probably because of a
        # # sync/async issue. given_source_ids_get_gaia_data now raises an
        # # error # if n_max exceeds 5e4, because the ~70k items that WERE
        # # returned are duds.
        # cols = (
        #     'g.source_id, g.ra, g.dec, g.parallax, g.parallax_error, g.pmra, '
        #     'g.pmdec, g.phot_g_mean_mag, g.phot_rp_mean_mag, g.phot_bp_mean_mag'
        # )
        # gdf = given_source_ids_get_gaia_data(
        #     np.array(df.source_id.astype(np.int64)),
        #     f'cdips_targets_v{catalog_vnum}',
        #     n_max=int(2e6), overwrite=False,
        #     enforce_all_sourceids_viable=True, whichcolumns=cols,
        #     gaia_datarelease='gaiadr2'
        # )

    gdf = given_votable_get_df(votablepath, assert_equal='source_id')

    if not len(gdf) == len(df):
        print(79*"*")
        print('WRN!')
        print(f'Expected {len(df)} matches in Gaia DR2')
        print(f'Got {len(gdf)} matches in Gaia DR2')
        print(79*"*")
        verify_gaia_xmatch(df, gdf, metadf)

    # every queried source_id should have a result. the two that do not are
    # EsplinLuhman2019, 377 matches to 443 stars, and Gagne2018c, 914 matches
    # to 916 stars. this is 60 missing stars out of 1.5 million. we'll be okay.
    # so, do the merge using the GAIA xmatch results as the base.
    mdf = gdf.merge(df, on='source_id', how='left')


    #
    # update metadf with new info.
    #
    N_stars_in_lists = []
    Nstars_with_age_in_lists = []
    N_sel0 = []
    N_sel1 = []
    N_sel2 = []
    for ix, r in metadf.iterrows():

        csvpath = os.path.join(clusterdatadir, r.csv_path)
        assert os.path.exists(csvpath)
        _df = pd.read_csv(csvpath)
        if 'source_id' not in _df.columns:
            _df = _df.rename(columns={"source":"source_id"})

        _sel = mdf.source_id.isin(_df.source_id)
        N_stars_in_lists.append(len(mdf[_sel]))
        _selage =  (~pd.isnull(mdf.age)) & _sel
        Nstars_with_age_in_lists.append(len(mdf[_selage]))

        _sel0 = (
            _sel
            &
            (mdf.phot_rp_mean_mag < 16)
        )

        _sel1 =  (
            _sel
            &
            ( (mdf.phot_rp_mean_mag < 16)
             |
            (
              (mdf.parallax/mdf.parallax_error > 5) & (mdf.parallax > 10)
            )
            )
        )

        _sel2 = _sel1 & (mdf.mean_age > -1)

        N_sel0.append(len(mdf[_sel0]))
        N_sel1.append(len(mdf[_sel1]))
        N_sel2.append(len(mdf[_sel2]))

    metadf["N_gaia"] = N_stars_in_lists
    metadf["N_gaia_withage"] = Nstars_with_age_in_lists
    metadf["N_Rplt16"] = N_sel0
    metadf["N_Rplt16_orclose"] = N_sel1
    metadf["N_Rplt16_orclose_withage"] = N_sel2
    metadf['Nstars_m_Ngaia'] = metadf.Nstars - metadf.N_gaia

    #
    # save the output
    #
    csvpath = os.path.join(
        clusterdatadir, f'cdips_targets_v{catalog_vnum}_nomagcut_gaiasources.csv'
    )
    if not os.path.exists(csvpath):
        mdf.to_csv(csvpath, index=False)
        print(f'Wrote {csvpath}')
    else:
        print(f'Found {csvpath}')

    metapath = os.path.join(
        clusterdatadir,
        f'list_of_lists_keys_paths_assembled_v{catalog_vnum}_gaiasources.csv'
    )
    if not os.path.exists(metapath):
        metadf.sort_values(by='Nstars', ascending=False).to_csv(metapath, index=False)
        print(f'Wrote {metapath}')
    else:
        print(f'Found {metapath}')

    # Rp<16
    csvpath = os.path.join(
        clusterdatadir, f'cdips_targets_v{catalog_vnum}_gaiasources_Rplt16.csv'
    )
    if not os.path.exists(csvpath):
        sel = (mdf.phot_rp_mean_mag < 16)
        smdf = mdf[sel]
        smdf.to_csv(csvpath, index=False)
        print(f'Wrote {csvpath}')
    else:
        print(f'Found {csvpath}')

    # Rp<16 or close
    csvpath = os.path.join(
        clusterdatadir, f'cdips_targets_v{catalog_vnum}_gaiasources_Rplt16_orclose.csv'
    )
    if not os.path.exists(csvpath):
        sel =  (
            (mdf.phot_rp_mean_mag < 16)
            |
            (
              (mdf.parallax/mdf.parallax_error > 5) & (mdf.parallax > 10)
            )
        )
        smdf = mdf[sel]
        smdf.to_csv(csvpath, index=False)
        print(f'Wrote {csvpath}')
    else:
        print(f'Found {csvpath}')
