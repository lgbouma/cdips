"""
Get all the Gaia DR2 source identifiers of the known planets.
"""
import os
import numpy as np, pandas as pd
from astroquery.utils.tap.core import TapPlus

from cdips.paths import DATADIR, LOCALDIR
clusterdatadir = os.path.join(DATADIR, 'cluster_data')

from cdips.utils import today_YYYYMMDD

def NASAExoArchive_to_csv(N_max=int(3e4)):

    outpath = os.path.join(
        LOCALDIR, f'NASAExoArchive_ps_table_{today_YYYYMMDD()}.csv'
    )

    if not os.path.exists(outpath):

        tap = TapPlus(url="https://exoplanetarchive.ipac.caltech.edu/TAP/")
        query = (
            f'select top 100000 '+
            'pl_name, gaia_id, st_age, st_ageerr1, st_ageerr2 from ps'
        )
        print(query)
        j = tap.launch_job(query=query)
        r = j.get_results()

        assert len(r) != N_max

        source_ids = np.array(
            pd.DataFrame({'gaia_id':r['gaia_id']}).
            gaia_id.str.extract('Gaia DR2 (.*)')
        )

        df = r.to_pandas()
        df['source_id'] = source_ids.astype(str)

        df.to_csv(outpath, index=False)
        print(f'Made {outpath}')

    df = pd.read_csv(outpath)

    sel = ~(pd.isnull(df.source_id))

    outdf = pd.DataFrame({
        'source_id':df.loc[sel,'source_id'].astype(np.int64),
        'st_age':df.loc[sel,'st_age'],
        'st_ageerr1':df.loc[sel,'st_ageerr1'],
        'st_ageerr2':df.loc[sel,'st_ageerr2']
    })
    outdf['cluster'] = f'NASAExoArchive_ps_{today_YYYYMMDD()}'

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
        clusterdatadir, 'v05', f'NASAExoArchive_ps_{today_YYYYMMDD()}.csv'
    )

    outdf[outcols].to_csv(outpath, index=False)
    print(f'Made {outpath}')
