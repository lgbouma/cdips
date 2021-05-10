# -*- coding: utf-8 -*-
'''
functions to download and wrangle catalogs of cluster members that I want to
crossmatch against Gaia-DR2.

called from `homogenize_cluster_lists.py`

contents:
Utilities:
    given_votable_get_df
Gaia groups:
    CantatGaudin20b_to_csv
    Kounkel2018_orion_to_csv
    KounkelCovey2019_clusters_to_csv
    GaiaCollaboration2018_clusters_to_csv

DEPRECATED:
    Kharchenko+2013:
        Kharchenko2013_position_mag_match_Gaia
        estimate_Gaia_G_given_2mass_J_Ks
        failed_Kharchenko2013_match_Gaia
    Dias+2014:
        Dias2014_position_mag_match_Gaia
        make_Dias2014_cut_csv
        make_Dias2014_cut_vot
        Dias2014_nbhr_gaia_to_nearestnbhr
'''
import os, pickle, subprocess, itertools, socket
from glob import glob

import numpy as np, pandas as pd
from numpy import array as nparr

from astropy.table import Table
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.io.votable import from_table, writeto, parse

from astroquery.gaia import Gaia

from astrobase.timeutils import precess_coordinates
from datetime import datetime

from cdips.paths import DATADIR
clusterdatadir = os.path.join(DATADIR, 'cluster_data')

from cdips.catalogbuild.vizier_xmatch_utils import (
    get_vizier_table_as_dataframe
)

def given_votable_get_df(votablepath, assert_equal='source_id'):
    """
    Given a single votable, convert to pandas DataFrame.

    If the votable has Gaia source-ids, then auto-verify the 19-character
    gaia-id doesn't suffer a too-few-bit integer truncation error.
    """
    vot = parse(votablepath)
    tab = vot.get_first_table().to_table()
    df = tab.to_pandas()

    if isinstance(assert_equal, str):
        np.testing.assert_array_equal(tab[assert_equal], df[assert_equal])

    return df


def Kounkel2018_orion_to_csv():

    inpath = os.path.join(clusterdatadir, 'Kounkel_2018_orion_table2.vot')
    df = given_votable_get_df(inpath, assert_equal='Gaia')

    inpath2 = os.path.join(clusterdatadir, 'Kounkel_2018_orion_table3.fits')
    df2 = Table(fits.open(inpath2)[1].data).to_pandas()

    sel = ~(
        (pd.isnull(df['Group']))
        |
        (df['Group'] == 'field')
        |
        (df['Group'] == '')
    )

    sdf = df[sel]

    # take HRD age over CMD age (spectroscopic Teffs, but more
    # uncertain)
    sdf2 = df2[['Group','Age-HR']]
    mdf = sdf.merge(sdf2, on='Group', how='left')
    mdf["age"] = np.round(np.log10(mdf["Age-HR"]),3)

    outdf = mdf[['Gaia','Group','age']]

    outdf = outdf.rename(columns={"Gaia": "source_id", "Group": "cluster"})

    outdf['cluster'] = np.core.defchararray.add(
        np.repeat('k18orion_', len(outdf)), nparr(outdf['cluster'])
    )

    outpath = inpath.replace('.vot','_cut_only_source_cluster_age.csv')
    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def KounkelCovey2019_clusters_to_csv():
    #
    # make "cluster,source" header dataframe with "cluster" being whatever
    # Kounkel & Covey gave as their crossmatched names, else groupids (if their
    # crossmatched name was NaN). we are not including the ages now, because we
    # are getting them in construct_unique_cluster_name_column when the age
    # column is assigned.
    #

    df1 = pd.read_csv(os.path.join(clusterdatadir,'KC19_string_table1.csv'))
    df2 = pd.read_csv(os.path.join(clusterdatadir,'KC19_string_table2.csv'))

    sdf2 = df2[['group_id','name','age']]

    mdf = df1.merge(sdf2, on='group_id', how='left')

    names = nparr(mdf['name'])
    groupids = nparr(mdf['group_id'])
    ages = nparr(mdf['age'])

    nullgroupids = groupids[pd.isnull(names)].astype(str)
    baz = np.core.defchararray.add(
        np.repeat('kc19group_', len(nullgroupids)), nullgroupids
    )
    names[pd.isnull(names)] = baz

    mdf['cluster'] = names

    scols = ['source_id','cluster','age']
    out_df = mdf[scols]

    outpath = os.path.join(
        clusterdatadir, 'KounkelCovey2019_cut_cluster_source_age.csv'
    )
    out_df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def CantatGaudin20b_to_csv():

    # open clusters and ages
    srccolumns = "Cluster|AgeNN"
    dstcolumns = "cluster|age"
    df1 = get_vizier_table_as_dataframe(
        "J/A+A/640/A1", srccolumns, dstcolumns, table_num=0,
        whichcataloglist="J/A+A/640/A1"
    )
    assert len(df1) == 2017

    # cluster members and source_ids
    srccolumns = "GaiaDR2|Cluster"
    dstcolumns = "source_id|cluster"
    df2 = get_vizier_table_as_dataframe(
        "J/A+A/640/A1", srccolumns, dstcolumns, table_num=1,
        whichcataloglist="J/A+A/640/A1"
    )
    assert len(df2) == 234128

    mdf = df2.merge(df1, how='left', on='cluster')
    assert len(mdf) == len(df2)

    outpath = os.path.join(
        clusterdatadir, 'v05', 'CantatGaudin20b_cut_cluster_source_age.csv'
    )
    mdf.to_csv(outpath, index=False)
    print(f'made {outpath}')


def Kounkel2020_to_csv():
    """
    Paper II of the "strings" series, Kounkel, Covey, and Stassun 2020.
    """
    #
    # make "cluster,source" header dataframe with "cluster" being whatever
    # KCS20 gave as their crossmatched names, else groupids (if their
    # crossmatched name was NaN). we are not including the ages now, because we
    # are getting them in construct_unique_cluster_name_column when the age
    # column is assigned.
    #

    # 987376 rows (stars)
    t1path = os.path.join(clusterdatadir,'v05','J_AJ_160_279_table1.dat.gz.fits')
    # 8293 groups
    t2path = os.path.join(clusterdatadir,'v05','J_AJ_160_279_table2.dat.gz.fits')

    df1 = Table(fits.open(t1path)[1].data).to_pandas()
    df2 = Table(fits.open(t2path)[1].data).to_pandas()

    sdf2 = df2[['Group','OName','logAge']]

    mdf = df1.merge(sdf2, on='Group', how='left')

    names = nparr(mdf['OName'])
    names = np.char.strip(names.astype(str))
    groupids = nparr(mdf['Group'])
    ages = nparr(mdf['logAge'])

    noothername = (pd.isnull(names))|(names=='')
    nullgroupids = groupids[noothername].astype(str)
    baz = np.core.defchararray.add(
        np.repeat('kcs20group_', len(nullgroupids)), nullgroupids
    )
    names[noothername] = baz

    mdf['cluster'] = names

    # # remove trailing whitespace [nvm; done above]
    # mdf.cluster = mdf.cluster.apply(
    #     lambda x: x.strip() if isinstance(x, str) else x
    # )

    mdf = mdf.rename(columns={"Gaia":"source_id", 'logAge':'age'})
    scols = ['source_id','cluster','age']
    out_df = mdf[scols]

    outpath = os.path.join(
        clusterdatadir, 'v05', 'Kounkel2020_cut_cluster_source_age.csv'
    )
    out_df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))


def GaiaCollaboration2018_clusters_to_csv():

    inpaths = [
        os.path.join(clusterdatadir,'GaiaCollaboration2018_616_A10_tablea1b_beyond_250pc.vot'),
        os.path.join(clusterdatadir,'GaiaCollaboration2018_616_A10_tablea1a_within_250pc.vot')
    ]

    for inpath in inpaths:
        tab = parse(inpath)

        t = tab.get_first_table().to_table()

        df = pd.DataFrame({'source':t['Source'], 'cluster':t['Cluster']})

        cnames = np.array(df['cluster'].str.decode('utf-8'))

        # you don't want the globular clusters from this work!
        bad_ngcs = [104, 288, 362, 1851, 5272, 5904, 6205, 6218, 6341, 6397,
                    6656, 6752, 6809, 7099]
        bad_ngcs = ['NGC{:s}'.format(str(bn).zfill(4)) for bn in bad_ngcs]
        # however, I manually checked: none of the globular clusters are in the
        # these source catalogs.

        outpath = inpath.replace('.vot','_cut_only_source_cluster.csv')
        df.to_csv(outpath, index=False)
        print('made {}'.format(outpath))


def failed_Kharchenko2013_match_Gaia():
    #
    # attempt:
    # use the 2MASS object ID's, given in “pts_key” format from Kharchenko, to
    # match against the tmass_best_neighbour table (tmass_oid column).
    #
    # result:
    # fails, because I think these are different IDs. not sure if it ALWAYS
    # fails, but it seems to fail enough of the time that using POSITIONS and
    # MAGNITUDES directly is probably the safer algorithm.

    csvpaths = glob('../data/cluster_data/MWSC_1sigma_members/????_*.csv')

    for csvpath in np.sort(csvpaths):

        outpath = csvpath.replace(
            'MWSC_1sigma_members','MWSC_1sigma_members_Gaia_matched')
        if os.path.exists(outpath):
            print('found {}, skipping'.format(outpath))
            continue

        k13_tab = Table.read(csvpath, format='ascii')

        print('{} -- {} stars'.format(csvpath, len(k13_tab)))

        k13_tmass_oids = np.array(k13_tab['2MASS'])

        gaia_source_ids = []
        for k13_tmass_oid in k13_tmass_oids:
            querystr = (
                "select top 1 source_id, original_ext_source_id, "
                "angular_distance, tmass_oid from gaiadr2.tmass_best_neighbour "
                "where tmass_oid = {}".format(k13_tmass_oid)
            )
            job = Gaia.launch_job(
                )
            res = job.get_results()

            if len(res)==0:
                gaia_source_ids.append(-1)
            else:
                gaia_source_ids.append(res['source_id'])

        gaia_source_ids = np.array(gaia_source_ids)

        outdf = pd.DataFrame(
            {'k13_tmass_oids':k13_tmass_oids,
             'gaia_tmass_best_neighbour_DR2_ids':gaia_source_ids })
        outdf.to_csv(outpath, index=False)
        print('--> made {}'.format(outpath))


def estimate_Gaia_G_given_2mass_J_Ks(J, Ks):
    # https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/cu5/Figures/cu5pho_PhotTransf_GKvsJK.png
    # or
    # https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html#Ch5.F17
    #
    # CITE: Carrasco et al (2016). The scatter on this relation is 0.14274 mag

    G_minus_Ks = (
        0.23584 + 4.0548*(J-Ks) -2.5608*(J-Ks)**2
        + 2.2228*(J-Ks)**3 - 0.54944*(J-Ks)**4
    )

    return G_minus_Ks + Ks


def Kharchenko2013_position_mag_match_Gaia(sepmax=5*u.arcsec, Gmag_pm_bound=2):
    """
    use position, and estimated G mag, to crossmatch Kharchenko catalog to
    gaia. (precess the J2000 K+13 positions to J2015.5 using the Kharchenko
    proper motions).

    args:
        sepmax: maximum separation

        Gmag_pm_bound: +/- how many magnitudes for real and estimated Gmag to
        match?
    """

    csvpaths = glob('../data/cluster_data/MWSC_1sigma_members/????_*.csv')

    for csvpath in np.sort(csvpaths):

        outpath = csvpath.replace(
            'MWSC_1sigma_members','MWSC_1sigma_members_Gaia_matched')
        if os.path.exists(outpath):
            print('found {}, skipping'.format(outpath))
            continue

        k13_tab = Table.read(csvpath, format='ascii')

        print('{} -- {} stars'.format(os.path.basename(csvpath), len(k13_tab)))

        ra_deg = np.array(k13_tab['RAdeg'])
        dec_deg = np.array(k13_tab['DEdeg'])

        pm_ra = np.array(k13_tab['pmRA'])
        pm_dec = np.array(k13_tab['pmDE'])

        # # K13 is listed as epoch 2000. I would think that I would need to
        # # precess to J2015.5 to match DR2. So set jd to JD2015.5, because this
        # # accounts for precession of the observing frame for whenever ra/dec
        # # were "measured". (ND doing JD2000 doesn't change). WHEN I DO THIS, IT
        # # BREAKS THE MATCH. So omit this step (even though it's the obviously
        # # right thing to do)
        # JD2000 = 2451545.0
        # JD2015pt5 = 2457174.5
        # coords_j2015pt5 = np.atleast_2d([
        #     precess_coordinates(_r, _d, 2000.0, 2015.5, jd=JD2015pt5,
        #                         mu_ra=_pr, mu_dec=_pd, outscalar=True)
        #     for _r, _d, _pr, _pd in
        #     zip(ra_deg, dec_deg, pm_ra, pm_dec)
        # ])
        # ra_j2015pt5, dec_j2015pt5 = coords_j2015pt5[:,0], coords_j2015pt5[:,1]

        k13_tmass_oids = np.array(k13_tab['2MASS'])

        J_mag = np.array(k13_tab['Jmag'])
        Ks_mag = np.array(k13_tab['Ksmag'])

        G_mag_estimate = estimate_Gaia_G_given_2mass_J_Ks(J_mag, Ks_mag)
        Gmag_lower = G_mag_estimate - Gmag_pm_bound
        Gmag_upper = G_mag_estimate + Gmag_pm_bound

        # SAVE:
        # K+13 2MASS ID (as an internal identifier for the K+13 catalog).
        # predicted mean gaia mag
        # the gaia source ID that is obtained
        # the gaia mag that is obtained

        gaia_source_ids, gaia_mags, n_in_nbhd, dists = [], [], [], []
        for _ra, _dec, _Gmag_lo, _Gmag_hi in zip(
            ra_deg, dec_deg, Gmag_lower, Gmag_upper
        ):

            pointsepstr = (
                "POINT('ICRS',ra,dec), POINT('ICRS',{},{})".
                format(_ra, _dec)
            )

            querystr = (
                "SELECT ra, dec, source_id, phot_g_mean_mag, "
                "DISTANCE({}) AS dist ".format(pointsepstr) +
                "FROM gaiadr2.gaia_source "
                "WHERE CONTAINS(POINT('ICRS', ra, dec), "
                "CIRCLE('ICRS', {}, {}, {})) = 1 ".format(
                _ra, _dec,
                sepmax.to(u.degree).value) +
                "AND "
                "phot_g_mean_mag BETWEEN {} and {}".format(
                _Gmag_lo, _Gmag_hi)
            )

            job = Gaia.launch_job(querystr)
            res = job.get_results()

            if len(res)==0:
                gaia_source_ids.append(-1)
                gaia_mags.append(np.nan)
                n_in_nbhd.append(0)
                dists.append(np.nan)
            elif len(res)==1:
                gaia_source_ids.append(int(res['source_id']))
                gaia_mags.append(float(res['phot_g_mean_mag']))
                n_in_nbhd.append(1)
                dists.append(float(res['dist']))
            else:
                res.sort(keys='dist')
                gaia_source_ids.append(int(res[0]['source_id']))
                gaia_mags.append(float(res[0]['phot_g_mean_mag']))
                n_in_nbhd.append(len(res))
                dists.append(float(res[0]['dist']))

        gaia_source_ids = np.array(gaia_source_ids)
        gmag_true_minus_estimate = np.array(gaia_mags) - G_mag_estimate

        outdf = pd.DataFrame(
            {'k13_tmass_oids':k13_tmass_oids,
             'gaia_dr2_match_id':gaia_source_ids,
             'gmag_match_minus_estimate':gmag_true_minus_estimate,
             'gmag_match':gaia_mags,
             'dist_deg':dists,
             'n_in_nbhd':n_in_nbhd})
        outdf.to_csv(outpath, index=False)
        print('--> made {}'.format(outpath))


def Dias2014_position_mag_match_Gaia(sepmax=5*u.arcsec, Gmag_pm_bound=2):
    """
    position + Gmag xmatch of Dias2014 to Gaia DR2.

    does ~4.5 matches per second. by default, 2e6 "members"->123 hours.
    """
    # http://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=J/A%2bA/564/A79/pm_ucac4
    inpath = ( '/home/luke/local/tess-trex/catalogs/'
              'Dias_2014_prob_gt_50_pct_vizier.vot')

    outpath = ('../data/cluster_data/'
               'Dias_2014_prob_gt_50_pct_vizier_Gaia_matched_EVENS.csv') #FIXME
    if os.path.exists(outpath):
        print('found {}, skipping'.format(outpath))
        return

    tab = parse(inpath)
    t = tab.get_first_table().to_table()

    J_mag = np.array(t['Jmag'])
    Ks_mag = np.array(t['Kmag']) # is actually Ks
    G_mag_estimate = estimate_Gaia_G_given_2mass_J_Ks(J_mag, Ks_mag)
    del J_mag, Ks_mag
    okinds = np.isfinite(G_mag_estimate)
    d14_ucac4_ids = np.array(t['UCAC4'])[okinds]

    Gmag_lower = G_mag_estimate[okinds] - Gmag_pm_bound
    Gmag_upper = G_mag_estimate[okinds] + Gmag_pm_bound

    #TODO: check if J2015 transformation needed.
    ra_deg = np.array(t['RAJ2000'])[okinds]
    dec_deg = np.array(t['DEJ2000'])[okinds]
    pmra = np.array(t['pmRA'])[okinds]
    pmde = np.array(t['pmDE'])[okinds]
    del t

    gaia_source_ids, gaia_mags, n_in_nbhd, dists = [], [], [], []
    ix = 0
    sl = slice(0,len(ra_deg),2) #FIXME
    for _ra, _dec, _Gmag_lo, _Gmag_hi in zip(
        ra_deg[sl], dec_deg[sl], Gmag_lower[sl], Gmag_upper[sl]
    ):
        print('{}: {}/{}'.format(datetime.utcnow().isoformat(), ix,
                                 len(ra_deg[sl])))
        ix += 1

        pointsepstr = (
            "POINT('ICRS',ra,dec), POINT('ICRS',{},{})".
            format(_ra, _dec)
        )

        querystr = (
            "SELECT ra, dec, source_id, phot_g_mean_mag, "
            "DISTANCE({}) AS dist ".format(pointsepstr) +
            "FROM gaiadr2.gaia_source "
            "WHERE CONTAINS(POINT('ICRS', ra, dec), "
            "CIRCLE('ICRS', {}, {}, {})) = 1 ".format(
            _ra, _dec,
            sepmax.to(u.degree).value) +
            "AND "
            "phot_g_mean_mag BETWEEN {} and {}".format(
            _Gmag_lo, _Gmag_hi)
        )

        job = Gaia.launch_job(querystr)
        res = job.get_results()

        if len(res)==0:
            gaia_source_ids.append(-1)
            gaia_mags.append(np.nan)
            n_in_nbhd.append(0)
            dists.append(np.nan)
        elif len(res)==1:
            gaia_source_ids.append(int(res['source_id']))
            gaia_mags.append(float(res['phot_g_mean_mag']))
            n_in_nbhd.append(1)
            dists.append(float(res['dist']))
        else:
            res.sort(keys='dist')
            gaia_source_ids.append(int(res[0]['source_id']))
            gaia_mags.append(float(res[0]['phot_g_mean_mag']))
            n_in_nbhd.append(len(res))
            dists.append(float(res[0]['dist']))

    gaia_source_ids = np.array(gaia_source_ids)
    gmag_true_minus_estimate = np.array(gaia_mags) - G_mag_estimate[okinds][sl]

    outdf = pd.DataFrame(
        {'d14_ucac4_ids':d14_ucac4_ids[sl],
         'gaia_dr2_match_id':gaia_source_ids,
         'gmag_match_minus_estimate':gmag_true_minus_estimate,
         'gmag_match':gaia_mags,
         'dist_deg':dists,
         'n_in_nbhd':n_in_nbhd})
    outdf.to_csv(outpath, index=False)
    print('--> made {}'.format(outpath))


def make_Dias2014_cut_csv():

    inpath = ( '/home/luke/local/tess-trex/catalogs/'
              'Dias_2014_prob_gt_50_pct_vizier.vot')

    outpath = ('../data/cluster_data/'
               'Dias_2014_prob_gt_50_pct_to_gaia_archive.csv')
    if os.path.exists(outpath):
        print('found {}, skipping'.format(outpath))
        return

    tab = parse(inpath)
    t = tab.get_first_table().to_table()

    J_mag = np.array(t['Jmag'])
    Ks_mag = np.array(t['Kmag']) # is actually Ks
    G_mag_estimate = estimate_Gaia_G_given_2mass_J_Ks(J_mag, Ks_mag)
    okinds = np.isfinite(G_mag_estimate)

    d14_ucac4_ids = np.array(t['UCAC4'])[okinds]

    #TODO: check if J2015 transformation needed.
    ra_deg = np.array(t['RAJ2000'])[okinds]
    dec_deg = np.array(t['DEJ2000'])[okinds]

    outdf = pd.DataFrame(
        {'RA':ra_deg,
         'DEC':dec_deg,
         'gmag_estimate':G_mag_estimate[okinds],
         'ucac_id':d14_ucac4_ids})
    outdf.to_csv(outpath, index=False)
    print('--> made {}'.format(outpath))


def make_Dias2014_cut_vot():

    inpath = ( '/home/luke/local/tess-trex/catalogs/'
              'Dias_2014_prob_gt_50_pct_vizier.vot')

    outpath = ('../data/cluster_data/'
               'Dias_2014_prob_gt_50_pct_to_gaia_archive.vot')
    if os.path.exists(outpath):
        print('found {}, skipping'.format(outpath))
        return

    tab = parse(inpath)
    t = tab.get_first_table().to_table()

    J_mag = np.array(t['Jmag'])
    Ks_mag = np.array(t['Kmag']) # is actually Ks
    G_mag_estimate = estimate_Gaia_G_given_2mass_J_Ks(J_mag, Ks_mag)
    okinds = np.isfinite(G_mag_estimate)

    d14_ucac4_ids = np.array(t['UCAC4'])[okinds]

    #TODO: check if J2015 transformation needed.
    ra_deg = np.array(t['RAJ2000'])[okinds]
    dec_deg = np.array(t['DEJ2000'])[okinds]

    import IPython; IPython.embed()
    outtab = Table()
    outtab['ra'] = ra_deg*u.deg
    outtab['dec'] = dec_deg*u.deg
    outtab['gmag_estimate'] = G_mag_estimate[okinds]*u.mag
    outtab['ucac_id'] = d14_ucac4_ids
    v_outtab = from_table(outtab)
    writeto(v_outtab, outpath)
    print('--> made {}'.format(outpath))


def Dias2014_nbhr_gaia_to_nearestnbhr():
    """
    ran dias14_gaia_adql_match.sql on the gaia archive. gave almost what we
    wanted, but we want nearest neighbors.
    """

    inpath = ( '../data/cluster_data/'
              'dias14_nbhr_gaia-result.vot.gz')

    outpath = ('../data/cluster_data/'
               'Dias14_seplt5arcsec_Gdifflt2.csv')
    if os.path.exists(outpath):
        print('found {}, skipping'.format(outpath))
        return

    tab = parse(inpath)
    t = tab.get_first_table().to_table()

    # make multiplicity column. then sort by ucacid, then by distance. then drop
    # ucac-id duplicates, keeping the first (you have nearest neighbor saved!)
    df = t.to_pandas()

    _, inv, cnts = np.unique(df['ucac_id'], return_inverse=True,
                             return_counts=True)

    df['n_in_nbhd'] = cnts[inv]

    df['ucac_id'] = df['ucac_id'].str.decode('utf-8')

    df = df.sort_values(['ucac_id','dist'])

    df = df.drop_duplicates(subset='ucac_id', keep='first')

    df['source_id'] = df['source_id'].astype('int64')

    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))
