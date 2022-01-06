"""
Make multipage PDFs needed to vet CDIPS objects of interest.

$ python -u make_all_vetting_reports.py &> logs/vetting_pdf.log &
"""

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############

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

    LOGINFO(f'Completed sector {sector}!')

if __name__ == "__main__":
    main(sector=24, cdips_cat_vnum=0.6)
    #for s in range(14,25):
    #    main(sector=s, cdips_cat_vnum=0.6)
