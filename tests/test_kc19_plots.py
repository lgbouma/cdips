import pandas as pd
from test_nbhd_plot import test_nbhd_plot

df = pd.read_csv('data/Mann_20200730_Theia_assoc_supp.csv')

for ix, r in df.iterrows():

    source_id = str(r.GAIA)
    sector = 42
    force_references = None
    force_groupname = None

    try:
        test_nbhd_plot(source_id, sector, force_references=force_references,
                       force_groupname=force_groupname)

    except:
        pass
