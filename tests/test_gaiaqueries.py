import numpy as np
from cdips.utils.gaiaqueries import given_dr2_sourceids_get_edr3_sourceids

dr2_source_ids = np.array([
    np.int64(5489726768531119616),
    np.int64(5489726768531118848)
])

df = given_dr2_sourceids_get_edr3_xmatch(dr2_source_ids, 'test_toi1937')

print(df)

