import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data
from cdips.paths import TESTDATADIR
import os

source_ids = np.array([
    '2128840912955018368'
]).astype(np.int64)

r0 = given_source_ids_get_gaia_data(
    source_ids,
    'test0',
    gaia_datarelease='gaiadr3',
    table_name='gaia_source',
    overwrite=False
)

r1 = given_source_ids_get_gaia_data(
    source_ids,
    'test1',
    gaia_datarelease='gaiadr3',
    table_name='vari_summary',
    overwrite=False
)

df = pd.read_csv(os.path.join(TESTDATADIR, 'tab_supp_RSG5_CH2_Prot.csv'))
df = df[(df.cluster=='RSG-5') & (~pd.isnull(df['Prot_Adopted']))]
print(f'Querying for len(df) stars...')

OVERWRITE = 1

r2 = given_source_ids_get_gaia_data(
    np.array(df.dr3_source_id.astype(np.int64)),
    'test2',
    gaia_datarelease='gaiadr3',
    table_name='vari_summary',
    enforce_all_sourceids_viable=False,
    overwrite=OVERWRITE
)

r3 = given_source_ids_get_gaia_data(
    np.array(df.dr3_source_id.astype(np.int64)),
    'test3',
    gaia_datarelease='gaiadr3',
    table_name='gaia_source',
    enforce_all_sourceids_viable=True,
    overwrite=OVERWRITE
)

df['dr3_source_id'] = df.dr3_source_id.astype(str)
r3['source_id'] = r3.source_id.astype(str)
mdf = df.merge(
    r3, left_on='dr3_source_id', right_on='source_id', how='left'
)

sel0 = ~pd.isnull(mdf.vbroad)

mdf0 = mdf[sel0]

plt.close('all')
fig, ax = plt.subplots(figsize=(4,3))
ax.scatter(mdf0.vbroad, mdf0['(BP-RP)0'])
ax.update({'xlabel':'vbroad', 'ylabel':'(BP-RP)0'})
fig.savefig('rsg5_DR3vbroad_vs_BPmRP0.png', dpi=300, bbox_inches='tight')

plt.close('all')
fig, ax = plt.subplots(figsize=(4,3))
_p = ax.scatter(mdf0.Prot_Adopted, mdf0.vbroad, c=mdf0['(BP-RP)0'])

axins1 = inset_axes(ax, width="3%", height="20%", loc='lower right',
                    borderpad=0.7)
cb = fig.colorbar(_p, cax=axins1, orientation="vertical", extend="neither")
cb.ax.tick_params(labelsize='x-small')
cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_label_position('left')
cb.set_label('(BP-RP)0', fontsize='x-small')

ax.update({'xlabel':'Prot [d]', 'ylabel':'vbroad'})
fig.savefig('rsg5_TESS-Prot_vs_DR3vbroad.png', dpi=300, bbox_inches='tight')


r4 = given_source_ids_get_gaia_data(
    np.array(df.dr3_source_id.astype(np.int64)),
    'test4',
    gaia_datarelease='gaiadr3',
    table_name='vari_classifier_result',
    enforce_all_sourceids_viable=False,
    overwrite=OVERWRITE
)

# zero matches! lol!
r5 = given_source_ids_get_gaia_data(
    np.array(df.dr3_source_id.astype(np.int64)),
    'test5',
    gaia_datarelease='gaiadr3',
    table_name='vari_rotation_modulation',
    enforce_all_sourceids_viable=False,
    overwrite=OVERWRITE
)



import IPython; IPython.embed()
