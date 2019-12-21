import os
from glob import glob

from cdips.vetting import (
    initialize_vetting_report_information as ivri,
    make_all_vetting_reports as mavp
)


def test_vetting_report(sector=None, cdips_cat_vnum=None):

    outdir = 'test_vetting_reports'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    (tfa_sr_paths, lcbasedir, resultsdir,
    cddf, supplementstatsdf, pfdf, toidf,
    k13_notes_df, sector) = (
            ivri.initialize_vetting_report_information(sector, cdips_cat_vnum,
                                                       baseresultsdir=outdir)
    )

    if sector == 6:
        ids_to_test = [
            '5579734916388215808' # PC
        ]

    if sector == 7:
        ids_to_test = [
            '5596735638203997824' # PC
        ]

    if sector == 9:
        ids_to_test = [
            '5256717749007641344', # PC near TOI 684
            '5523104093269203712', # PC
            '5489726768531119616', # PC
            '5432321060287733888', # PC
            '5256502966294604800', # PC
            '5524965566445719168', # EB offtarget
            '4666329826478795008', # contact binary
            '5436252604630719488', # non-CM from parallax
        ]

    for id_to_test in ids_to_test:

        existing_report_files = glob(
            'test_vetting_reports/*/*/*{}*pdf'.format(id_to_test)
        )
        if len(existing_report_files) > 0:
            for f in existing_report_files:
                os.remove(f)
                print('removing {} to make for test_vetting_report'.format(f))

        paths = [f for f in tfa_sr_paths if id_to_test in f]

        mavp.make_all_vetting_reports(
            paths, lcbasedir, resultsdir, cddf, supplementstatsdf, pfdf,
            toidf, k13_notes_df, sector=sector
        )


if __name__ == "__main__":

    test_vetting_report(sector=6, cdips_cat_vnum=0.4)
