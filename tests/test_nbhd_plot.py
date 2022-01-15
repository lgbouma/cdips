"""
Given a source_id and group information, make the neighborhood plot.
(Even if the target star has not been labelled as a member of the group)
"""
import os, pickle
from glob import glob
import numpy as np, matplotlib.pyplot as plt

from cdips.vetting import (
    initialize_vetting_report_information as ivri,
    make_all_vetting_reports as mavp,
    initialize_neighborhood_information as ini
)
from cdips.plotting import vetting_pdf as vp

DEBUG = 1

def test_nbhd_plot(source_id, sector, cdips_cat_vnum=0.6,
                   force_references=None, force_groupname=None,
                   manual_gmag_limit=None):

    outdir = 'test_nbhd_plot'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outpath = os.path.join(outdir, f'{source_id}_nbhd_plot.png')

    if os.path.exists(outpath):
        print(f'Found {outpath}! continue.')
        return 1

    (lc_paths, lcbasedir, resultsdir, cdips_df, supplementstatsdf, pfdf, toidf,
     sector) = (
            ivri.initialize_vetting_report_information(
                sector, cdips_cat_vnum, baseresultsdir=outdir
            )
    )

    info = (
         ini.get_group_and_neighborhood_information(
             source_id, overwrite=0, force_references=force_references,
             force_groupname=force_groupname
         )
    )

    if isinstance(info, tuple):

        if DEBUG:
            picklepath = 'test_nbhd_plot/nbhd_info_{}.pkl'.format(source_id)
            with open(picklepath , 'wb') as f:
                pickle.dump(info, f)
                print('made {}'.format(picklepath))

        (targetname, groupname, referencename, group_df_dr2, target_df,
         nbhd_df, pmdec_min, pmdec_max, pmra_min, pmra_max
        ) = info

        fig = vp.plot_group_neighborhood(
            targetname, groupname, referencename, group_df_dr2, target_df,
            nbhd_df, pmdec_min=pmdec_min, pmdec_max=pmdec_max,
            pmra_min=pmra_min, pmra_max=pmra_max, source_id=source_id,
            figsize=(30,20), show_rvs=True
        )

    elif info is None:

        info = ini.get_neighborhood_information(
            source_id, overwrite=0, manual_gmag_limit=manual_gmag_limit
        )

        if DEBUG:
            picklepath = 'test_nbhd_plot/nbhd_info_{}.pkl'.format(source_id)
            with open(picklepath , 'wb') as f:
                pickle.dump(info, f)
                print('made {}'.format(picklepath))

        (targetname, groupname, target_df, nbhd_df, pmdec_min, pmdec_max,
         pmra_min, pmra_max) = info

        fig = vp.plot_neighborhood_only(
            targetname, groupname, target_df, nbhd_df,
            pmdec_min=pmdec_min, pmdec_max=pmdec_max,
            pmra_min=pmra_min, pmra_max=pmra_max,
            source_id=source_id, figsize=(30,20),
        )

    else:

        raise NotImplementedError

    fig.savefig(outpath)
    print('made {}'.format(outpath))

    plt.close()


if __name__ == "__main__":

    # source_id = '3222255959210123904'
    # sector = 42
    # force_references = "Kounkel_2018_Ori" # can be none
    # force_groupname = "k18orion_25Ori-1"

    source_id =  '4516549232971506560'
    sector = 14
    force_references = None
    force_groupname = None

    # source_id = '565706429072383232'
    # sector = 42
    # force_references = None
    # force_groupname = None

    # source_id = '5245968236116294016'
    # sector = 9
    # force_references = "Kounkel_2019" # can be none
    # force_groupname = "kc19group_1091"

    manual_gmag_limit=18

    test_nbhd_plot(source_id, sector, force_references=force_references,
                   force_groupname=force_groupname,
                   manual_gmag_limit=manual_gmag_limit)
