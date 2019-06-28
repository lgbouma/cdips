"""
analyze stats of cdips stars.

* how many are there total?
* what fraction come from which sources?
* how many are single-source claims?
* how many are multi-source claims?

python paper_get_CDIPS_star_stats.py &> ../logs/paper_get_CDIPS_star_stats.log
"""

import pandas as pd, numpy as np
from collections import Counter
from cdips.utils import collect_cdips_lightcurves as ccl

df = ccl.get_cdips_catalog(ver=0.3)

n_stars = len(df)

refs = np.array(df['reference'])
cnt = Counter(refs)
mostcommon = cnt.most_common(n=10)

ismult = np.array([',' in r for r in refs])
issing = ~ismult

n_single = len(refs[issing])
n_mult = len(refs[ismult])

print(
"""
From {cdipspath}

Total number of stars with G_Rp < 16: {n_stars}

Most common sources are:
{mostcommon}

There are {n_single} single-source claims.

There are {n_mult} claims from multiple sources.
""".format(
    cdipspath=cdipspath,
    n_stars=n_stars,
    mostcommon=repr(mostcommon),
    n_single=n_single,
    n_mult=n_mult
)
)
