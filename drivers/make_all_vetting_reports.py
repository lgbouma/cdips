"""
Make multipage PDFs needed to vet CDIPS objects of interest.

$ python -u make_all_vetting_reports.py &> logs/vetting_pdf.log &
"""

from cdips.vetting import (
    initialize_vetting_report_information as ivri,
    make_all_vetting_reports as mavp
)

def main(sector=None, cdips_cat_vnum=None):

    (tfa_sr_paths, lcbasedir, resultsdir,
    cddf, supplementstatsdf, pfdf, toidf,
    k13_notes_df, sector) = (
            ivri.initialize_vetting_report_information(sector, cdips_cat_vnum)
    )

    mavp.make_all_vetting_reports(
        tfa_sr_paths, lcbasedir, resultsdir, cddf, supplementstatsdf, pfdf,
        toidf, k13_notes_df, sector=sector
    )


if __name__ == "__main__":

    main(sector=9, cdips_cat_vnum=0.4)
