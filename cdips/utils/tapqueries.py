"""
Contents:

    get_exoplanetarchive_planetarysystems
    given_source_ids_get_tic8_data
"""
import os
import numpy as np
from astroquery.utils.tap.core import TapPlus
from cdips.utils.gaiaqueries import (
    make_votable_given_source_ids, given_votable_get_df
)

CACHEDIR = os.path.join(
    os.path.expanduser('~'), '.tapqueries_cache'
)
if not os.path.exists(CACHEDIR):
    os.mkdir(CACHEDIR)

def get_exoplanetarchive_planetarysystems(tabletype="ps", overwrite=1,
                                          n_max=int(3e5), verbose=1):
    """
    Args:
        overwrite: if true, will download the LATEST tables from the NASA
        exoplanet archive. otherwise, uses the cache.

        tabletype: "ps" or "pscomppars"

    Returns:
        dataframe with the Planetary Systems, "ps" table. (Nb. there is a
        composite parameters table too, "pscomppars").
        https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
    """

    dlpath = os.path.join(
        CACHEDIR, f'{tabletype}_exoplanetarchive.xml.gz'
    )

    if os.path.exists(dlpath) and overwrite:
        os.remove(dlpath)

    if not os.path.exists(dlpath):

        tap = TapPlus(url="https://exoplanetarchive.ipac.caltech.edu/TAP/")

        jobstr = (
        f'''
        SELECT top {n_max:d} *
        FROM {tabletype}
        '''
        )
        query = jobstr

        if verbose:
            print(20*'.')
            print('Executing:')
            print(query)
            print(20*'.')

        j = tap.launch_job(query=query, verbose=True, dump_to_file=True,
                           output_file=dlpath)

    df = given_votable_get_df(dlpath, assert_equal=None)

    return df



def given_source_ids_get_tic8_data(source_ids, queryname, n_max=10000,
                                   overwrite=True,
                                   enforce_all_sourceids_viable=True):
    """
    Args:

        source_ids (np.ndarray) of np.int64 Gaia DR2 source_ids

        queryname (str): used for files

        overwrite: if True, and finds that this crossmatch has already run,
        deletes previous cached output and reruns anyway.

        enforce_all_sourceids_viable: if True, will raise an assertion error if
        every source id does not return a result. (Unless the query returns
        n_max entries, in which case only a warning will be raised).

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
        CACHEDIR, f'toupload_{queryname}_tic8.xml'
    )
    dlpath = os.path.join(
        CACHEDIR, f'{queryname}_matches_tic8.xml.gz'
    )

    if overwrite:
        if os.path.exists(xmltouploadpath):
            os.remove(xmltouploadpath)

    if not os.path.exists(xmltouploadpath):
        make_votable_given_source_ids(source_ids, outpath=xmltouploadpath)

    if os.path.exists(dlpath) and overwrite:
        os.remove(dlpath)

    if not os.path.exists(dlpath):

        tap = TapPlus(url="http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap")

        jobstr = (
        '''
        SELECT top {n_max:d} *
        FROM TAP_UPLOAD.foobar as u, "IV/38/tic" as t
        WHERE u.source_id=t.GAIA
        '''
        ).format(
            n_max=n_max
        )
        query = jobstr

        # might do async if this times out. but it doesn't.
        j = tap.launch_job(query=query, upload_resource=xmltouploadpath,
                           upload_table_name="foobar", verbose=True,
                           dump_to_file=True, output_file=dlpath)

    df = given_votable_get_df(dlpath, assert_equal=None)

    import IPython; IPython.embed()

    if len(df) != len(source_ids) and enforce_all_sourceids_viable:
        if len(df) == n_max:
            wrnmsg = (
                'WRN! got {} matches vs {} source id queries'.
                format(len(df), len(source_ids))
            )
            print(wrnmsg)
        else:
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
