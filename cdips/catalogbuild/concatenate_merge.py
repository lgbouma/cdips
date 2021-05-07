"""
Once the CSV files of source_ids, ages, and references are assembled,
concatenate and merge them.

Date: May 2021.

Background: Created for v0.5 target catalog merge, to simplify life.
"""

import numpy as np, pandas as pd
import os
from glob import glob

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

def get_target_catalog_latex_table():
    """
    authorname/year, title, number of DR2 sources that I collected, number of DR2
    sources with Rp<16
    """

    # https://ads.readthedocs.io/en/latest/#getting-started
    import ads
    # NOTE: it takes lists too. best to just do that! ignoring the nan for
    # Joel's list.
    r = ads.ExportQuery("2019AJ....158..122K").execute()


def assemble_target_catalog(catalog_vnum):

    metadf = pd.read_csv(
        os.path.join(clusterdatadir, 'V05_LIST_OF_LISTS_KEYS_PATHS.csv')
    )
    metadf['bibcode'] = metadf.ads_link.str.extract("abs\/(.*)\/")

    N_stars_in_lists = []
    Nstars_with_age_in_lists = []

    # for each table, concatenate into a dataframe of source_id, cluster,
    # log10age ("age").
    for ix, r in metadf.iterrows():

        print(79*'-')
        print(f'Beginning {r.reference_id}...')

        csvpath = os.path.join(clusterdatadir, r.csv_path)
        assert os.path.exists(csvpath)

        df = pd.read_csv(csvpath)

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
                'Appending the reference_id as the cluster ID.'
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
            ):
                age = np.ones(len(df))*np.nan
                df['age'] = age

            else:
                age_mapper = lambda k: AGE_LOOKUP[k]
                age = df.cluster.apply(age_mapper)
                df['age'] = age

        N_stars_in_lists.append(len(df))
        Nstars_with_age_in_lists.append(len(df[~pd.isnull(df.age)]))

    metadf["Nstars"] = N_stars_in_lists
    metadf["Nstars_with_age"] = Nstars_with_age_in_lists
    #FIXME FIXME FIXME
    #FIXME FIXME FIXME
    #FIXME FIXME FIXME


    outpath = os.path.join(
        clusterdatadir, 'V05_LIST_OF_LISTS_KEYS_PATHS_ASSEMBLED.csv'
    )

    import IPython; IPython.embed()
    assert 0
