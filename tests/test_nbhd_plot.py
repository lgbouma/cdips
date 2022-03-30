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

    source_id = None #'4516549232971506560'
    sector = 99
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

    manual_gmag_limit=17

    source_ids = ["2020089283003866624", "218545876102500096", "6199429191749977344",
                  "5376067159192314240", "6083912679070170496", "420394942288749440",
                  "604988891451223936", "5617412190575256448", "4469298957694642816",
                  "2942380703200115200", "220489572141949568", "1972288603420863104",
                  "5878638324948917760", "5420033090134127872", "1984191744484890240",
                  "2899361864087217152", "3082564690530707328", "3433405450655641984",
                  "5599919823877882496", "1957513636725425920", "222756009202731648",
                  "4112191977761736448", "3291455819447952768", "659285833649720064",
                  "5877202397180600832", "5606474390644345216", "139900347689740800",
                  "3080104185367102592", "978017379613172864", "1810700285772906880",
                  "344346876950576768", "3290593798035412480", "6658373007402886400",
                  "6770698256306535296", "893550942158776832", "604915739569261184"
             ]
    source_ids = ["5236556416614488576", "5288535107223500928"]
    source_ids = ["181104928193233408"]

    for source_id in source_ids:
        test_nbhd_plot(source_id, sector, force_references=force_references,
                       force_groupname=force_groupname,
                       manual_gmag_limit=manual_gmag_limit)
