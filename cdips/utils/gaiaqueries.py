"""
Contents:

    | given_source_ids_get_gaia_data
    | given_source_ids_get_neighbor_counts
    | gaia2read_given_df
    | query_neighborhood
    | given_dr2_sourceids_get_edr3_xmatch
    | given_dr3_sourceids_get_dr2_xmatch

    | make_votable_given_source_ids
    | given_votable_get_df

    Auxiliary utilies:
    | edr3_propermotion_to_ICRF
    | parallax_to_distance_highsn
    | dr3_activityindex_espcs_to_RprimeIRT

    Photometric conversion:
    | dr3_bprp_to_gv
"""
###########
# imports #
###########
import os
import uuid
import numpy as np, matplotlib.pyplot as plt, pandas as pd

from astropy.io.votable import from_table, writeto, parse
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

from astroquery.gaia import Gaia

##########
# config #
##########

homedir = os.path.expanduser("~")
credentials_file = os.path.join(homedir, '.gaia_credentials')
if not os.path.exists(credentials_file):
    raise AssertionError(
        f'need gaia dr2 credentials file at {credentials_file}.\n'
        f'format should be two lines, with username and password.'
    )

gaiadir = os.path.join(homedir, '.gaia_cache')
if not os.path.exists(gaiadir):
    os.mkdir(gaiadir)


#############
# functions #
#############

def make_votable_given_source_ids(source_ids, outpath=None):

    t = Table()
    t['source_id'] = source_ids

    votable = from_table(t)

    writeto(votable, outpath)
    print(f'made {outpath}')

    return outpath


def make_votable_given_vector_dict(vectordict, outpath=None):

    t = Table()
    for k,v in vectordict.items():
        t[k] = v

    votable = from_table(t)

    writeto(votable, outpath)
    print(f'made {outpath}')

    return outpath


def given_votable_get_df(votablepath, assert_equal='source_id'):

    vot = parse(votablepath)
    tab = vot.get_first_table().to_table()
    df = tab.to_pandas()

    if isinstance(assert_equal, str):
        np.testing.assert_array_equal(tab[assert_equal], df[assert_equal])

    return df


def given_source_ids_get_gaia_data(source_ids, groupname, n_max=10000,
                                   overwrite=True,
                                   enforce_all_sourceids_viable=True,
                                   savstr='',
                                   which_columns='*',
                                   table_name='gaia_source',
                                   gaia_datarelease='gaiadr2',
                                   getdr2ruwe=False):
    """
    Args:

        source_ids (np.ndarray) of np.int64 Gaia DR2/EDR3 source_ids. (If EDR3,
        be sure to use the correct `gaia_datarelease` kwarg)

        groupname (str)

        overwrite: if True, and finds that this crossmatch has already run,
        deletes previous cached output and reruns anyway.

        enforce_all_sourceids_viable: if True, will raise an assertion error if
        every source id does not return a result. (Unless the query returns
        n_max entries, in which case only a warning will be raised).

        savstr (str); optional string that will be included in the path to
        the downloaded vizier table.

        which_columns (str): ADQL column getter string. For instance "*", or
        "g.ra, g.dec, g.parallax, g.pmra, g.pmdec, g.phot_g_mean_mag".

        gaia_datarelease (str): 'gaiadr2' or 'gaiaedr3'. Default is Gaia DR2.

        getdr2ruwe (bool): if True, queries gaiadr2.ruwe instead of
        gaiadr2.gaia_source

    Returns:

        dataframe with Gaia DR2 / EDR3 crossmatch info.
    """

    if n_max > int(5e4):
        raise NotImplementedError(
            'the gaia archive / astroquery seems to give invalid results past '
            '50000 source_ids in this implementation...'
        )

    if type(source_ids) != np.ndarray:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia source_ids'
        )
    if type(source_ids[0]) != np.int64:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia source_ids'
        )

    xmltouploadpath = os.path.join(
        gaiadir, f'toupload_{groupname}{savstr}_{gaia_datarelease}.xml'
    )
    dlpath = os.path.join(
        gaiadir, f'group{groupname}_matches{savstr}_{gaia_datarelease}.xml.gz'
    )

    if overwrite:
        if os.path.exists(xmltouploadpath):
            os.remove(xmltouploadpath)

    if not os.path.exists(xmltouploadpath):
        make_votable_given_source_ids(source_ids, outpath=xmltouploadpath)

    if os.path.exists(dlpath) and overwrite:
        os.remove(dlpath)

    if not getdr2ruwe:
        jobstr = (
        '''
        SELECT top {n_max:d} {which_columns}
        FROM tap_upload.foobar as u, {gaia_datarelease:s}.{table_name} AS g
        WHERE u.source_id=g.source_id
        '''
        ).format(
            which_columns=which_columns,
            n_max=n_max,
            gaia_datarelease=gaia_datarelease,
            table_name=table_name
        )
    else:
        assert gaia_datarelease == 'gaiadr2'
        jobstr = (
        '''
        SELECT top {n_max:d} *
        FROM tap_upload.foobar as u, gaiadr2.ruwe AS g
        WHERE u.source_id=g.source_id
        '''
        ).format(
            n_max=n_max
        )

    query = jobstr

    if not os.path.exists(dlpath):

        Gaia.login(credentials_file=credentials_file)

        # might do async if this times out. but it doesn't.
        j = Gaia.launch_job(query=query,
                            upload_resource=xmltouploadpath,
                            upload_table_name="foobar", verbose=True,
                            dump_to_file=True, output_file=dlpath)

        Gaia.logout()

    df = given_votable_get_df(dlpath, assert_equal='source_id')

    if len(df) != len(source_ids) and enforce_all_sourceids_viable:
        if len(df) == n_max:
            wrnmsg = (
                'WRN! got {} matches vs {} source id queries'.
                format(len(df), len(source_ids))
            )
            print(wrnmsg)
        else:
            errmsg = (
                f'ERROR! got {len(df)} matches vs {len(source_ids)} '
                f'source id queries. '
                f'\ngroupname was {groupname}; '
                f'\nSource ids were: '
                f'\n{source_ids}'
            )
            print(errmsg)
            raise AssertionError(errmsg)

    if len(df) != len(source_ids) and not enforce_all_sourceids_viable:
        wrnmsg = (
            'WRN! got {} matches vs {} source id queries'.
            format(len(df), len(source_ids))
        )
        print(wrnmsg)

    return df


def given_source_ids_get_neighbor_counts(
    source_ids, dGmag, sep_arcsec, runid, n_max=20000, overwrite=True,
    enforce_all_sourceids_viable=True,
    gaia_datarelease='gaiadr3'):
    """
    Given a list of Gaia source_ids, return a dataframe containing the count of
    the number of sources within `dGmag` magnitudes and `sep_arcsec` arcseconds
    away from each source.  This vectorizes the slow serial cone-search
    implementation available in astroquery.

    Args:

        source_ids (np.ndarray) of np.int64 Gaia DR2/EDR3 source_ids. Default
        assumed is (E)DR3.

        runid (str): identifyin string

        overwrite: if True, and finds that this crossmatch has already run,
        deletes previous cached output and reruns anyway.

        enforce_all_sourceids_viable: if True, will raise an assertion error if
        every source id does not return a result. (Unless the query returns
        n_max entries, in which case only a warning will be raised).

        gaia_datarelease (str): 'gaiadr2' or 'gaiaedr3'. Default is Gaia DR2.

    Returns:

        tuple of two dataframes: (count_df, df).  `count_df` has columns
        "source_id" (the input source_ids) and "nbhr_count".  `df` actually
        specifies the matches, including their source_id, ra, dec,
        phot_g_mean_mag, distance in arcseconds, and difference in g_mag
        relative to each target star.
    """

    if n_max > int(5e4):
        raise NotImplementedError(
            'the gaia archive / astroquery seems to give invalid results past '
            '50000 source_ids in this implementation...'
        )

    if type(source_ids) != np.ndarray:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia source_ids'
        )
    if type(source_ids[0]) != np.int64:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia source_ids'
        )

    gdf0 = given_source_ids_get_gaia_data(
        source_ids, runid+"_initial_query", n_max=n_max, overwrite=overwrite,
        enforce_all_sourceids_viable=enforce_all_sourceids_viable,
        which_columns='g.source_id, g.ra, g.dec, g.phot_g_mean_mag',
        table_name='gaia_source', gaia_datarelease=gaia_datarelease
    )

    xmltouploadpath = os.path.join(
        gaiadir, f'toupload_nbhrcount_{runid}_{gaia_datarelease}.xml'
    )
    dlpath = os.path.join(
        gaiadir, f'nbhrcount_group{runid}_matches_{gaia_datarelease}.xml.gz'
    )

    if overwrite:
        if os.path.exists(xmltouploadpath):
            os.remove(xmltouploadpath)

    if not os.path.exists(xmltouploadpath):
        d = {
            'source_id': np.array(gdf0.source_id).astype(np.int64),
            'ra': np.array(gdf0.ra),
            'dec': np.array(gdf0.dec),
            'phot_g_mean_mag': np.array(gdf0.phot_g_mean_mag)
        }
        make_votable_given_vector_dict(d, outpath=xmltouploadpath)

    if os.path.exists(dlpath) and overwrite:
        os.remove(dlpath)

    from astropy import units as u
    sep_deg = (sep_arcsec*u.arcsec).to(u.deg).value

    jobstr = (
    """
    SELECT top {n_max:d}
    u.source_id, g.source_id, g.ra, g.dec, g.phot_g_mean_mag,
    3600*DISTANCE(POINT('ICRS', u.ra, u.dec), POINT('ICRS', g.ra, g.dec)) as dist_arcsec,
    g.phot_g_mean_mag - u.phot_g_mean_mag as d_gmag
    FROM
    tap_upload.foobar as u, {gaia_datarelease:s}.gaia_source as g
    WHERE
    1 = CONTAINS(POINT('ICRS', u.ra, u.dec), CIRCLE('ICRS', g.ra, g.dec, {sep_deg}))
    AND
    g.phot_g_mean_mag - u.phot_g_mean_mag < {dGmag}
    AND
    u.source_id != g.source_id
    ORDER BY
    u.source_id, dist_arcsec ASC
    """
    ).format(
        n_max=n_max,
        gaia_datarelease=gaia_datarelease,
        sep_deg=sep_deg,
        dGmag=dGmag
    )

    query = jobstr

    if not os.path.exists(dlpath):

        Gaia.login(credentials_file=credentials_file)

        # might do async if this times out. but it doesn't.
        j = Gaia.launch_job(query=query,
                            upload_resource=xmltouploadpath,
                            upload_table_name="foobar", verbose=True,
                            dump_to_file=True, output_file=dlpath)

        Gaia.logout()

    df = given_votable_get_df(dlpath, assert_equal=None)
    df = df.rename({'source_id_2':'nbhr_source_id'})

    from collections import Counter

    r = Counter(df['source_id'])

    foo_df = pd.DataFrame(
        {"source_id":source_ids, "nbhr_count_0":np.zeros(len(source_ids))}
    )
    temp_df = pd.DataFrame(
        {"source_id":r.keys(), "nbhr_count_1":r.values()}
    )
    count_df = foo_df.merge(temp_df, how='left', on='source_id')
    count_df['nbhr_count'] = np.nanmax(
        [count_df.nbhr_count_0, count_df.nbhr_count_1], axis=0
    ).astype(int)
    count_df = count_df[['source_id', 'nbhr_count']]

    return count_df, df


def query_neighborhood(bounds, groupname, n_max=2000, overwrite=True,
                       manual_gmag_limit=16, mstr='',
                       use_bonus_quality_cuts=True):
    """
    Given the bounds in position and parallax corresponding to a group in the
    CDIPS target catalogs, get the DR2 stars from the group's neighborhood.

    The bounds are lower and upper in ra, dec, parallax, and there is a
    limiting G magnitude. A maximum number of stars, `n_max`, are selected from
    within these bounds.

    Args:
        bounds (dict): dict with keys parallax_lower, parallax_upper, ra_lower,
        ra_upper, dec_lower, dec_upper. (Each of which has a float value).

        groupname (str): string used when cacheing for files. if you are
        querying a field star, best to include sourceid.

        n_max (int): maximum number of stars in the neighborhood to acquire.

        mstr (str): string used in the cached neighborhood pickle file.

        use_bonus_quality_cuts (bool): default True. Imposes some things like
        "there need to be at least 8 Gaia transits", and similar requirements.

    Returns:
        dataframe of DR2 stars within the bounds given. This is useful for
        querying stars that are in the neighborhood of some group.
    """

    if manual_gmag_limit is not None:
        g_mag_limit = manual_gmag_limit
    else:
        g_mag_limit = 16
        LOGINFO(f'Using default g_mag_limit of {g_mag_limit}')

    dlpath = os.path.join(
        gaiadir,'nbhd_group{}_matches{}.xml.gz'.format(groupname, mstr)
    )

    if os.path.exists(dlpath) and overwrite:
        os.remove(dlpath)

    if not os.path.exists(dlpath):

        Gaia.login(credentials_file=credentials_file)

        jobstr = (
        """
        select top {n_max:d}
            g.source_id, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
            g.phot_g_mean_mag, g.parallax, g.ra, g.dec, g.pmra, g.pmdec,
            g.radial_velocity, g.radial_velocity_error
        from gaiadr2.gaia_source as g
        where
            g.parallax > {parallax_lower:.2f}
        and
            g.parallax < {parallax_upper:.2f}
        and
            g.dec < {dec_upper:.2f}
        and
            g.dec > {dec_lower:.2f}
        and
            g.ra > {ra_lower:.2f}
        and
            g.ra < {ra_upper:.2f}
        and
            g.phot_g_mean_mag < {g_mag_limit:.1f}
        order by
            random_index
        """
        )
        query = jobstr.format(
            n_max=n_max,
            parallax_lower=bounds['parallax_lower'],
            parallax_upper=bounds['parallax_upper'],
            ra_lower=bounds['ra_lower'],
            ra_upper=bounds['ra_upper'],
            dec_lower=bounds['dec_lower'],
            dec_upper=bounds['dec_upper'],
            g_mag_limit=g_mag_limit
        )

        if use_bonus_quality_cuts:
            # Impose some extra quality cuts, originally from Kounkel & Covey
            # 2019, but generally applicable.
            jobstr = (
            """
            select top {n_max:d}
                g.source_id, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
                g.phot_g_mean_mag, g.parallax, g.ra, g.dec, g.pmra, g.pmdec,
                g.radial_velocity, g.radial_velocity_error
            from gaiadr2.gaia_source as g
            where
                g.parallax > {parallax_lower:.2f}
            and
                g.parallax < {parallax_upper:.2f}
            and
                g.dec < {dec_upper:.2f}
            and
                g.dec > {dec_lower:.2f}
            and
                g.ra > {ra_lower:.2f}
            and
                g.ra < {ra_upper:.2f}
            and
                g.phot_g_mean_mag < {g_mag_limit:.1f}
            and
                parallax > 1
            and
                parallax_error < 0.1
            and
                1.0857/phot_g_mean_flux_over_error < 0.03
            and
                astrometric_sigma5d_max < 0.3
            and
                visibility_periods_used > 8
            and (
                    (astrometric_excess_noise < 1)
                    or
                    (astrometric_excess_noise > 1 and astrometric_excess_noise_sig < 2)
            )
            order by
                random_index
            """
            )
            query = jobstr.format(
                n_max=n_max,
                parallax_lower=bounds['parallax_lower'],
                parallax_upper=bounds['parallax_upper'],
                ra_lower=bounds['ra_lower'],
                ra_upper=bounds['ra_upper'],
                dec_lower=bounds['dec_lower'],
                dec_upper=bounds['dec_upper'],
                g_mag_limit=g_mag_limit
            )

        # async jobs can avoid timeout
        j = Gaia.launch_job_async(query=query, verbose=True, dump_to_file=True,
                                  output_file=dlpath)
        #j = Gaia.launch_job(query=query, verbose=True, dump_to_file=True,
        #                    output_file=dlpath)

        Gaia.logout()

    df = given_votable_get_df(dlpath, assert_equal='source_id')

    return df


def given_dr2_sourceids_get_edr3_xmatch(
    dr2_source_ids, runid, overwrite=True,
    enforce_all_sourceids_viable=True):
    """
    Use the dr2_neighborhood table to look up the (E)DR3 source_ids given DR2
    source_ids.

    "The only safe way to compare source records between different Data
    Releases in general is to check the records of proximal source(s) in the
    same small part of the sky. This table provides the means to do this via a
    precomputed crossmatch of such sources, taking into account the proper
    motions available at E/DR3."

    "Within the neighbourhood of a given E/DR3 source there may be none, one or
    (rarely) several possible counterparts in DR2 indicated by rows in this
    table. This occasional source confusion is an inevitable consequence of the
    merging, splitting and deletion of identifiers introduced in previous
    releases during the DR3 processing and results in no guaranteed one–to–one
    correspondence in source identifiers between the releases."

    See:
    https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_auxiliary_tables/ssec_dm_dr2_neighbourhood.html

    Args:

        dr2_source_ids (np.ndarray) of np.int64 Gaia DR2 source_ids

        runid (str): identifier used to identify and cache jobs.

        overwrite: if True, and finds that this crossmatch has already run,
        deletes previous cached output and reruns anyway.

        enforce_all_sourceids_viable: if True, will raise an assertion error if
        every source id does not return a result. (Unless the query returns
        n_max entries, in which case only a warning will be raised).

    Returns:

        dr2_x_edr3_df (pd.DataFrame), containing:
            ['source_id', 'dr2_source_id', 'dr3_source_id', 'angular_distance',
            'magnitude_difference', 'proper_motion_propagation']

        where "source_id" is the requested source_id, and the remaining columns
        are matches from the dr2_neighborhood table.

        This DataFrame should then be used to ensure e.g., that every REQUESTED
        source_id provides only one MATCHED star.
    """

    if type(dr2_source_ids) != np.ndarray:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia DR2 source_ids'
        )
    if type(dr2_source_ids[0]) != np.int64:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia DR2 source_ids'
        )
    if not isinstance(runid, str):
        raise TypeError(
            'Expect runid to be a (preferentially unique among jobs) string.'
        )

    xmltouploadpath = os.path.join(
        gaiadir, f'toupload_{runid}.xml'
    )
    dlpath = os.path.join(
        gaiadir,f'{runid}_matches.xml.gz'
    )

    if overwrite:
        if os.path.exists(xmltouploadpath):
            os.remove(xmltouploadpath)

    if not os.path.exists(xmltouploadpath):
        make_votable_given_source_ids(dr2_source_ids, outpath=xmltouploadpath)

    if os.path.exists(dlpath) and overwrite:
        os.remove(dlpath)

    if not os.path.exists(dlpath):

        n_max = 2*len(dr2_source_ids)
        print(f"Setting n_max = 2 * (number of dr2_source_ids) = {n_max}")

        Gaia.login(credentials_file=credentials_file)

        jobstr = (
        '''
        SELECT top {n_max:d} *
        FROM tap_upload.foobar as u, gaiaedr3.dr2_neighbourhood AS g
        WHERE u.source_id=g.dr2_source_id
        '''
        ).format(
            n_max=n_max
        )
        query = jobstr

        # might do async if this times out. but it doesn't.
        j = Gaia.launch_job(query=query,
                            upload_resource=xmltouploadpath,
                            upload_table_name="foobar", verbose=True,
                            dump_to_file=True, output_file=dlpath)

        Gaia.logout()

    df = given_votable_get_df(dlpath, assert_equal=None)

    if len(df) > len(dr2_source_ids):
        wrnmsg = (
            'WRN! got {} matches vs {} source id queries. Fix via angular_distance or magnitude_difference'.
            format(len(df), len(dr2_source_ids))
        )
        print(wrnmsg)

    if len(df) < len(dr2_source_ids) and enforce_all_sourceids_viable:
        errmsg = (
            'ERROR! got {} matches vs {} dr2 source id queries'.
            format(len(df), len(dr2_source_ids))
        )
        print(errmsg)
        raise AssertionError(errmsg)

    df['abs_magnitude_difference'] = np.abs(df['magnitude_difference'])

    return df


def given_dr3_sourceids_get_dr2_xmatch(
    dr3_source_ids, runid, overwrite=True,
    enforce_all_sourceids_viable=True):
    """
    Use the dr2_neighborhood table to look up the DR2 source_ids given
    the (E)DR3 source_ids.

    See docstring for given_dr2_sourceids_get_edr3_xmatch.

    Returns:

        dr2_x_edr3_df (pd.DataFrame), containing:
            ['source_id', 'dr2_source_id', 'dr3_source_id', 'angular_distance',
            'magnitude_difference', 'proper_motion_propagation']

        where "source_id" is the requested source_id, and the remaining columns
        are matches from the dr2_neighborhood table.

        This DataFrame should then be used to ensure e.g., that every REQUESTED
        source_id provides only one MATCHED star.
    """

    if type(dr3_source_ids) != np.ndarray:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia DR2 source_ids'
        )
    if type(dr3_source_ids[0]) != np.int64:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia DR2 source_ids'
        )
    if not isinstance(runid, str):
        raise TypeError(
            'Expect runid to be a (preferentially unique among jobs) string.'
        )

    xmltouploadpath = os.path.join(
        gaiadir, f'toupload_{runid}.xml'
    )
    dlpath = os.path.join(
        gaiadir,f'{runid}_matches.xml.gz'
    )

    if overwrite:
        if os.path.exists(xmltouploadpath):
            os.remove(xmltouploadpath)

    if not os.path.exists(xmltouploadpath):
        make_votable_given_source_ids(dr3_source_ids, outpath=xmltouploadpath)

    if os.path.exists(dlpath) and overwrite:
        os.remove(dlpath)

    if not os.path.exists(dlpath):

        n_max = 2*len(dr3_source_ids)
        print(f"Setting n_max = 2 * (number of dr3_source_ids) = {n_max}")

        Gaia.login(credentials_file=credentials_file)

        jobstr = (
        '''
        SELECT top {n_max:d} *
        FROM tap_upload.foobar as u, gaiaedr3.dr2_neighbourhood AS g
        WHERE u.source_id=g.dr3_source_id
        '''
        ).format(
            n_max=n_max
        )
        query = jobstr

        # might do async if this times out. but it doesn't.
        j = Gaia.launch_job(query=query,
                            upload_resource=xmltouploadpath,
                            upload_table_name="foobar", verbose=True,
                            dump_to_file=True, output_file=dlpath)

        Gaia.logout()

    df = given_votable_get_df(dlpath, assert_equal=None)

    if len(df) > len(dr3_source_ids):
        wrnmsg = (
            'WRN! got {} matches vs {} source id queries. Fix via angular_distance or magnitude_difference'.
            format(len(df), len(dr3_source_ids))
        )
        print(wrnmsg)

    if len(df) < len(dr3_source_ids) and enforce_all_sourceids_viable:
        errmsg = (
            'ERROR! got {} matches vs {} dr2 source id queries'.
            format(len(df), len(dr3_source_ids))
        )
        print(errmsg)
        raise AssertionError(errmsg)

    df['abs_magnitude_difference'] = np.abs(df['magnitude_difference'])

    return df


def gaia2read_given_df(df, cachedir, cache_id=None):
    """
    Given a dataframe of Gaia DR2 sources with key "dr2_source_id", run
    `gaia2read` to get all Gaia columns for the source list.   This assumes
    `gaia2read` is installed and working.

    The output is cached at either {cachedir}/{cache_id}_gaia2read.csv or if
    cache_id is None, at {cachedir}/{randomstring}_gaia2read.csv
    """

    if cache_id is None:
        idstr = str(uuid.uuid4())
    else:
        idstr = cache_id
    srcpath = os.path.join(cachedir, f'{idstr}_sources_only.csv')
    dstpath = os.path.join(cachedir, f'{idstr}_gaia2read.csv')

    if not os.path.exists(srcpath):
        assert 'dr2_source_id' in df
        df['dr2_source_id'].to_csv(srcpath, index=False, header=False)

    if not os.path.exists(dstpath):
        gaia2readcmd = f"gaia2read --header --extra --idfile {srcpath} --out {dstpath}"
        print(f'Beginning {gaia2readcmd}')
        returncode = os.system(gaia2readcmd)
        if returncode != 0: raise AssertionError('gaia2read cmd failed!!')
        print(f'Ran {gaia2readcmd}')
    else:
        print(f'Found {dstpath}')

    gdf = pd.read_csv(dstpath, delim_whitespace=True)
    gdf = gdf.rename({
        '#Gaia-ID[1]':'dr2_source_id',
        'RA[deg][2]':'ra',
        'Dec[deg][3]':'dec',
        'phot_g_mean_mag[20]':'phot_g_mean_mag',
        'phot_bp_mean_mag[25]':'phot_bp_mean_mag',
        'phot_rp_mean_mag[30]':'phot_rp_mean_mag',
    }, axis='columns')

    return gdf


def edr3_propermotion_to_ICRF(pmra, pmdec, ra, dec, G):
    """
    Cantat-Gaudin & Brandt 2021 correction of EDR3 to ICRF proper motions.
    This applies only to bright sources (G<13).  Correction is of order ~0.1
    mas/yr, depending on the source brightness and on-sky location.

    Input: source position, coordinates, and G magnitude from Gaia EDR3.
    Output: corrected proper motion.
    """
    if G>=13:
        return pmra, pmdec

    def sind(x):
        return np.sin(np.radians(x))

    def cosd(x):
        return np.cos(np.radians(x))

    table1=""" 0.0 9.0 18.4 33.8 -11.3
               9.0 9.5 14.0 30.7 -19.4
               9.5 10.0 12.8 31.4 -11.8
               10.0 10.5 13.6 35.7 -10.5
               10.5 11.0 16.2 50.0 2.1
               11.0 11.5 19.4 59.9 0.2
               11.5 11.75 21.8 64.2 1.0
               11.75 12.0 17.7 65.6 -1.9
               12.0 12.25 21.3 74.8 2.1
               12.25 12.5 25.7 73.6 1.0
               12.5 12.75 27.3 76.6 0.5
               12.75 13.0 34.9 68.9 -2.9 """
    table1 = np.fromstring(table1 ,sep=' ').reshape((12,5)).T

    Gmin = table1[0]
    Gmax = table1[1]

    #pick the appropriate omegaXYZ for the source's magnitude:
    omegaX = table1[2][(Gmin <=G)&(Gmax >G)][0]
    omegaY = table1[3][(Gmin <=G)&(Gmax >G)][0]
    omegaZ = table1[4][(Gmin <=G)&(Gmax >G)][0]

    pmraCorr = -1*sind(dec)*cosd(ra)*omegaX -sind(dec)*sind(ra)*omegaY + cosd(dec)*omegaZ
    pmdecCorr = sind(ra)*omegaX -cosd(ra)*omegaY

    return pmra -pmraCorr/1000., pmdec -pmdecCorr /1000.


def parallax_to_distance_highsn(parallax_mas, e_parallax_mas=0,
                                gaia_datarelease='gaia_edr3'):
    """
    Given a Gaia parallax in mas, get a zero-point corrected trigonometric
    parallax in parsecs.

    gaia_datarelease must be in ['gaia_edr3', 'gaia_dr3', 'gaia_dr2', 'none'],
    where "none" means the offset correction is not applied.
    """

    assert gaia_datarelease in ['gaia_edr3', 'gaia_dr3', 'gaia_dr2', 'none']

    if gaia_datarelease in ['gaia_edr3', 'gaiadr3']:
        # Applicable to 5-parameter solutions (and more or less to 6-parameter
        # too) e.g., https://arxiv.org/abs/2103.16096,
        # https://arxiv.org/abs/2101.09691
        offset = -0.026 # mas
    elif gaia_datarelease == 'gaia_dr2':
        # Lindegren+2018, as used by e.g., Gagne+2020 ApJ 903 96
        offset = -0.029 # mas
    elif gaia_datarelease == 'none':
        offset = 0 # mas
    else:
        raise NotImplementedError

    dist_trig_pc = 1e3 * ( 1 / (parallax_mas - offset) )

    if e_parallax_mas:
        upper_dist_trig_pc = 1e3 * ( 1 / (parallax_mas - e_parallax_mas - offset) )
        lower_dist_trig_pc = 1e3 * ( 1 / (parallax_mas + e_parallax_mas - offset) )
        upper_unc = upper_dist_trig_pc - dist_trig_pc
        lower_unc = dist_trig_pc - lower_dist_trig_pc
        return dist_trig_pc, upper_unc, lower_unc

    return dist_trig_pc


def dr3_activityindex_espcs_to_RprimeIRT(alpha, Teff, M_H=0.):
    """
    Use the coefficients from Table 1 of Lanzafame+2022 (Gaia DR3 Ca IRT
    validation paper) to convert the activity index ("activityindex_espcs") for
    Ca IRT to a R'_IRT value.

    Args:
        alpha: activityindex_espcs np.ndarray
        Teff: effective temperature np.nddary (calibrated on teff_gspphot)

    Kwargs:
        M_H: the weakly dependent metallicity term.  Not obvious that including
        this does much useful, so by default this function adopts the solar
        metallicity.  A "todo" would be to add an interpolation step here.

    Returns:
        log10_RprimeIRT
    """

    θ = np.log10(Teff)
    log10_alpha = np.log10(alpha)

    # Table 1: key [M/H], vals C0, C1, C2, C3
    c_dict = {
        -0.5: [-3.3391, -0.1564, -0.1046, 0.0311],
         0.0: [-3.3467, -0.1989, -0.1020, 0.0349],
         0.25: [-3.3501, -0.2137, -0.1029, 0.0357],
         0.5: [-3.3527, -0.2219, -0.1056, 0.0353],
    }

    if isinstance(M_H, (int, float)):

        vals = c_dict[float(M_H)]
        C0, C1, C2, C3 = vals

    else:

        errmsg = (
            'Need to add an interpolator over metallicity. '
            'This is not very hard!'
        )
        raise NotImplementedError(errmsg)


    print('WARNING!  In RprimeIRT it would be best to interpolate over metallcity, rather '
          'than assuming solar!!!')
    log10_RprimeIRT = (
        (C0 + C1*θ + C2*θ**2 + C3*θ**3)
        +
        log10_alpha
    )

    return log10_RprimeIRT


def dr3_bprp_to_gv(bp_rp):
    """
    Given Gaia DR3 BP-RP, return Johnson-Cousins G-V
    (and therefore, you can get V).
    https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5/Figures/cu5pho_PhotTransf_GVvsBPRP_Stetson.png
    """
    x = bprp
    c0 = -0.02704
    c1 = 0.01424
    c2 = -0.2156
    c3 = 0.01426
    y = c0 + c1*x + c2*x**2 + c3*x**3
    return y
