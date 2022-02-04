import os
import pandas as pd
from cdips.utils.catalogs import get_cdips_catalog
from cdips.paths import LOCALDIR

lc_list_path = os.path.join(LOCALDIR, "catalogs", "lc_list_20220131.txt")

df = pd.read_csv(lc_list_path, names=['lcpath'])

df['name'] = df['lcpath'].apply(
    lambda x: os.path.basename(x)
)

df['source_id'] = df['name'].apply(
    lambda x: x.split("gaiatwo")[1].split('-')[0].lstrip('0')
)
df['sector'] = df['name'].apply(
    lambda x: x.split("gaiatwo")[1].split('-')[1].lstrip('0')
)
df['cam'] = df['lcpath'].apply(
    lambda x: x.split('cam')[1][0]
)
df['ccd'] = df['lcpath'].apply(
    lambda x: x.split('ccd')[1][0]
)

cdf = get_cdips_catalog(ver=0.6)
cdf['source_id'] = cdf.source_id.astype(str)

mdf = df.merge(cdf, how='left', on='source_id')

assert len(mdf) == len(df)

cols = (
    'source_id,ra,dec,parallax,parallax_error,pmra,pmdec,phot_g_mean_mag,'+
    'phot_rp_mean_mag,phot_bp_mean_mag'
).split(',')
coldict = {c:'dr2_'+c for c in cols}

mdf = mdf.rename(columns=coldict)

mdf = mdf.drop(columns='lcpath')

outpath = os.path.join(LOCALDIR, "catalogs",
             "hlsp_cdips_tess_ffi_s0001-s0026_tess_v01_catalog.csv")
mdf.to_csv(
    outpath, index=False, sep=';'
)
print(f"Wrote {outpath}")
