"""
Given a source_id and group information, make the neighborhood plot.
(Even if the target star has not been labelled as a member of the group)
"""
import os
from glob import glob
import numpy as np, matplotlib.pyplot as plt

from cdips.vetting import (
    initialize_vetting_report_information as ivri,
    make_all_vetting_reports as mavp,
    initialize_neighborhood_information as ini
)
from cdips.plotting import vetting_pdf as vp

def test_nbhd_plot(source_id, sector, cdips_cat_vnum=0.4,
                   force_references=None, force_groupname=None):

    outdir = 'test_nbhd_plot'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    (tfa_sr_paths, lcbasedir, resultsdir,
    cddf, supplementstatsdf, pfdf, toidf,
    k13_notes_df, sector) = (
            ivri.initialize_vetting_report_information(sector, cdips_cat_vnum,
                                                       baseresultsdir=outdir)
    )

    info = (
         ini.get_neighborhood_information(source_id, mmbr_dict=None,
                                          k13_notes_df=k13_notes_df,
                                          overwrite=0,
                                          force_references=force_references,
                                          force_groupname=force_groupname)
    )

    (targetname, groupname, group_df_dr2, target_df, nbhd_df,
     cutoff_probability, pmdec_min, pmdec_max, pmra_min, pmra_max,
     group_in_k13, group_in_cg18, group_in_kc19
    ) = info

    fig = vp.plot_group_neighborhood(
        targetname, groupname, group_df_dr2, target_df, nbhd_df,
        cutoff_probability, pmdec_min=pmdec_min, pmdec_max=pmdec_max,
        pmra_min=pmra_min, pmra_max=pmra_max,
        group_in_k13=group_in_k13, group_in_cg18=group_in_cg18,
        group_in_kc19=group_in_kc19, source_id=source_id,
        figsize=(30,20), show_rvs=True
    )

    outpath = os.path.join(outdir, '{}_nbhd_plot.png'.format(source_id))
    fig.savefig(outpath)
    print('made {}'.format(outpath))

    plt.close()


if __name__ == "__main__":

    source_id = '5245968236116294016'
    sector = 9
    force_references = "Kounkel_2019" # can be none
    force_groupname = "kc19group_1091"

    test_nbhd_plot(source_id, sector, force_references=force_references,
                   force_groupname=force_groupname)