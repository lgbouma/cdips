###########
# imports #
###########
import os
import numpy as np, matplotlib.pyplot as plt, pandas as pd

from astropy.io.votable import from_table, writeto, parse
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

from astroquery.gaia import Gaia

from astrobase.services.gaia import objectid_search

##########
# config #
##########

homedir = os.path.expanduser("~")
credentials_file = os.path.join(homedir, '.gaia_credentials')
if not os.path.exists(credentials_file):
    raise AssertionError(
        'need gaia dr2 credentials file at {}'.format(credentials_file)
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
    print('made {}'.format(outpath))

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
                                   savstr=''):
    """
    Args:

        source_ids (np.ndarray) of np.int64 Gaia DR2 source_ids

        groupname (str)

        overwrite: if True, and finds that this crossmatch has already run,
        deletes previous cached output and reruns anyway.

        enforce_all_sourceids_viable: if True, will raise an assertion error if
        every source id does not return a result.

        savstr (str); optional string that will be included in the path to
        the downloaded vizier table.

    Returns:

        dataframe with Gaia DR2 crossmatch info.
    """

    if type(source_ids) != np.ndarray:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia DR2 source_ids'
        )
    if type(source_ids[0]) != np.int64:
        raise TypeError(
            'source_ids must be np.ndarray of np.int64 Gaia DR2 source_ids'
        )

    xmltouploadpath = os.path.join(
        gaiadir, 'toupload_{}{}.xml'.format(groupname, savstr)
    )
    dlpath = os.path.join(
        gaiadir,'group{}_matches{}.xml.gz'.format(groupname, savstr)
    )

    if overwrite:
        if os.path.exists(xmltouploadpath):
            os.remove(xmltouploadpath)

    if not os.path.exists(xmltouploadpath):
        make_votable_given_source_ids(source_ids, outpath=xmltouploadpath)

    if os.path.exists(dlpath) and overwrite:
        os.remove(dlpath)

    if not os.path.exists(dlpath):

        Gaia.login(credentials_file=credentials_file)

        jobstr = (
        '''
        SELECT top {n_max:d} *
        FROM tap_upload.foobar as u, gaiadr2.gaia_source AS g
        WHERE u.source_id=g.source_id
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

    df = given_votable_get_df(dlpath, assert_equal='source_id')

    if len(df) != len(source_ids) and enforce_all_sourceids_viable:
        errmsg = (
            'ERROR! got {} matches vs {} source id queries'.
            format(len(df), len(source_ids))
        )
        print(errmsg)
        import IPython; IPython.embed()
        raise AssertionError(errmsg)

    if len(df) != len(source_ids) and not enforce_all_sourceids_viable:
        wrnmsg = (
            'WRN! got {} matches vs {} source id queries'.
            format(len(df), len(source_ids))
        )
        print(wrnmsg)

    return df


def query_neighborhood(bounds, groupname, n_max=2000, overwrite=True,
                       is_cg18_group=True, is_kc19_group=False,
                       is_k13_group=False):
    """
    Given the bounds in position and parallx corresponding to some group (e.g.,
    from Cantat-Gaudin+2018 or Kounkel & Covey 2019), get the DR2 stars from
    the group's neighborhood.

    The bounds are lower and upper in ra, dec, parallax, and there is a
    limiting G magnitude. A maximum number of stars, `n_max`, are selected from
    within these bounds.

    Args:
        bounds (dict): with parallax, ra, dec bounds.

        groupname (str)

        n_max (int): maximum number of stars in the neighborhood to acquire.

        is_kc19_group (bool): if the group is from Kounkel & Covey 2019,
        slightly different query is required.

    Returns:
        dataframe of DR2 stars within the bounds given. This is useful for
        querying stars that are in the neighborhood of some group.
    """

    if is_cg18_group:
        g_mag_limit=18
    elif is_kc19_group:
        g_mag_limit=14
    elif is_k13_group:
        g_mag_limit=16

    dlpath = os.path.join(
        gaiadir,'nbhd_group{}_matches.xml.gz'.format(groupname)
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
            g.phot_g_mean_mag < {g_mag_limit:d}
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

        if is_kc19_group:
            # Kounkel & Covey impose some extra quality cuts.
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
                g.phot_g_mean_mag < {g_mag_limit:d}
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
