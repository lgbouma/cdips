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

    id_to_test = '5256502966294604800'

    existing_report_files = glob(
        'test_vetting_reports/*/*/*{}*pdf'.format(id_to_test)
    )
    if len(existing_report_files) > 0:
        for f in existing_report_files:
            os.remove(f)
            print('removing {} to make for test_vetting_report'.format(f))

    tfa_sr_paths = [f for f in tfa_sr_paths if id_to_test in f]

    mavp.make_all_vetting_reports(
        tfa_sr_paths, lcbasedir, resultsdir, cddf, supplementstatsdf, pfdf,
        toidf, k13_notes_df, sector=sector
    )


if __name__ == "__main__":

    test_vetting_report(sector=9, cdips_cat_vnum=0.4)

