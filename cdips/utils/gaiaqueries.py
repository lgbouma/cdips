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

homedir = os.path.basedir(os.path.abspath('~'))
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

def given_source_ids_get_gaia_data(source_ids, groupname, overwrite=True,
                                   enforce_all_sourceids_viable=True):
    """
    Args:

        source_ids (np.ndarray) of np.int64 Gaia DR2 source_ids

        groupname (str)

        overwrite: if True, and finds that this crossmatch has already run,
        deletes previous cached output and reruns anyway.

        enforce_all_sourceids_viable: if True, will raise an assertion error if
        every source id does not return a result.

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

    xmltouploadpath = os.path.join(gaiadir,'toupload_{}.xml'.format(groupname))
    dlpath = os.path.join(gaiadir,'group{}_matches.xml.gz'.format(groupname))

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
        SELECT *
        FROM tap_upload.foobar as u, gaiadr2.gaia_source AS g
        WHERE u.source_id=g.source_id
        '''
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
        raise AssertionError(errmsg)

    if len(df) != len(source_ids) and not enforce_all_sourceids_viable:
        wrnmsg = (
            'WRN! got {} matches vs {} source id queries'.
            format(len(df), len(source_ids))
        )
        print(wrnmsg)

    return df



