"""
Make multipage PDFs needed to vet CDIPS objects of interest.

$ python -u make_all_vetting_reports.py &> logs/vetting_pdf.log &
"""

from cdips.vetting import (
    initialize_vetting_report_information as ivri,
    make_all_vetting_reports as mavp
)

import os
from glob import glob

def main(sector=None, cdips_cat_vnum=None):

    (lc_paths, lcbasedir, resultsdir, cdips_df, supplementstatsdf, pfdf, toidf,
     sector) = (
            ivri.initialize_vetting_report_information(sector, cdips_cat_vnum)
    )

    mavp.make_all_vetting_reports(
        lc_paths, lcbasedir, resultsdir, cdips_df, supplementstatsdf, pfdf,
        toidf, sector=sector, show_rvs=True
    )

if __name__ == "__main__":
    main(sector=14, cdips_cat_vnum=0.6)
