"""
Get all the Gaia DR2 source identifiers of the known planets.
"""
import os
import numpy as np, pandas as pd
from astroquery.utils.tap.core import TapPlus

from cdips.paths import DATADIR, LOCALDIR
clusterdatadir = os.path.join(DATADIR, 'cluster_data')

from cdips.utils import today_YYYYMMDD

def NASAExoArchive_to_csv(N_max=int(1e5)):

    outpath = os.path.join(
        LOCALDIR, f'NASAExoArchive_ps_table_20210506.csv'
    )

    if not os.path.exists(outpath):

        tap = TapPlus(url="https://exoplanetarchive.ipac.caltech.edu/TAP/")
        query = (
            f'select top {N_max} '+
            'pl_name, gaia_id, st_age, st_ageerr1, st_ageerr2 from ps'
        )
        print(query)
        j = tap.launch_job(query=query)
        r = j.get_results()

        assert len(r) != N_max

        df = r.to_pandas()

        source_ids = np.array(
            pd.DataFrame({'gaia_id':r['gaia_id'].astype(str)}).
            gaia_id.str.extract('Gaia DR2 (.*)')
        ).astype(str)

        # NOTE this is nasty. NaN's, if written to the CSV file, will then be
        # READ in with read_csv as an object column. int64/string
        # representation doesnt change it. (int64 actually fails, because it
        # can't convert the NaN)
        source_ids = source_ids.flatten()
        sel = ~(source_ids == 'nan')

        sdf = df[sel]

        sdf['source_id'] = source_ids[sel].astype(str)

        sdf.to_csv(outpath, index=False)
        print(f'Made {outpath}')

    df = pd.read_csv(outpath)
    sel = ~(pd.isnull(df.source_id))

    outdf = pd.DataFrame({
        'source_id':df.loc[sel,'source_id'].astype(np.int64),
        'st_age':df.loc[sel,'st_age'],
        'st_ageerr1':df.loc[sel,'st_ageerr1'],
        'st_ageerr2':df.loc[sel,'st_ageerr2']
    })
    outdf['cluster'] = f'NASAExoArchive_ps_20210506'

    has_age_value = ~pd.isnull(outdf.st_age)
    has_age_errs = (~pd.isnull(outdf.st_ageerr1)) & (~pd.isnull(outdf.st_ageerr2))

    outdf['age'] = np.ones(len(outdf))*np.nan
    sel = has_age_value & has_age_errs

    # assign ages
    outdf.loc[sel, 'age'] = np.round(np.log10(outdf.loc[sel, 'st_age']*1e9),3)

    # drop duplicates
    outdf = outdf.drop_duplicates('source_id', keep='first')

    outcols = ['source_id','cluster','age']

    outpath = os.path.join(
        clusterdatadir, 'v05', f'NASAExoArchive_ps_20210506.csv'
    )

    outdf[outcols].to_csv(outpath, index=False)
    print(f'Made {outpath}')
