import matplotlib.pyplot as plt, pandas as pd, numpy as np
"""
goal: get Gaia DR2 IDs of stars in "clusters". these lists are then used to
define the CDIPS stellar sample.

see ../doc/list_of_cluster_member_lists.ods for an organized spreadsheet of the
different member lists
"""

import os, pickle, subprocess, itertools
from glob import glob

from numpy import array as nparr

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.io.votable import from_table, writeto, parse

from astroquery.gaia import Gaia

from astrobase.timeutils import precess_coordinates

def GaiaCollaboration2018_clusters_to_csv():

    inpaths = [
        '../data/cluster_data/GaiaCollaboration2018_616_A10_tablea1a_within_250pc.vot',
        '../data/cluster_data/GaiaCollaboration2018_616_A10_tablea1b_beyond_250pc.vot'
    ]

    for inpath in inpaths:
        tab = parse(inpath)

        t = tab.get_first_table().to_table()

        df = pd.DataFrame({'source':t['Source'], 'cluster':t['Cluster']})

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


if __name__ == "__main__":

    GaiaCollab18 = 0
    K13 = 1

    if GaiaCollab18:
        GaiaCollaboration2018_clusters_to_csv()

    if K13:
        Kharchenko2013_position_mag_match_Gaia()
