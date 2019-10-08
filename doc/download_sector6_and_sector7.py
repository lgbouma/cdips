"""
Script to download all the sector 6 and 7 CDIPS light curves via astroquery.

On a wired internet connection, I got ~150 LCs per minute. So this is 1e3
minutes ~= 16 hours to download the 1.5e5 light curves from CDIPS sectors 6 and
7.

Sector 6 is 21 Gb.  Sector 7 is 31 Gb.

So on average this is a bit less than 1Mb/sec download speed.
"""
from astroquery.mast import Observations
import os

lightcurve_directory = '/home/luke/temp' # CHANGE THIS! LCs are saved here.

if not os.path.exists(lightcurve_directory):
    os.mkdir(lightcurve_directory)

with open('s6s7_lc_list.txt', 'r') as f:
    lc_list = f.readlines()

for ix, lc in enumerate(lc_list):

    print('{}/{}'.format(ix, len(lc_list)))

    obs_id = lc.replace('.fits\n','')

    outdir = os.path.join(lightcurve_directory, obs_id)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    obs_table = Observations.query_criteria(
        provenance_name='CDIPS', obs_id=obs_id
    )

    _table = Observations.get_product_list(
        obs_table[0]
    )

    Observations.download_products(
        _table, download_dir=outdir
    )
