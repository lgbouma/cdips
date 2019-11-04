"""
Make multipage PDFs needed to vet CDIPS objects of interest.

$ python -u make_all_vetting_reports.py &> logs/vetting_pdf.log &
"""

from glob import glob
import datetime, os, pickle, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cdips.plotting import vetting_pdf as vp
from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.vetting import (
    initialize_vetting_report_information as ivri,
    make_all_vetting_reports as mavp
)

from numpy import array as nparr
from astropy.io import fits
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier


def main(sector=None, cdips_cat_vnum=None):

    (tfa_sr_paths, lcbasedir, resultsdir,
    cddf, supplementstatsdf, pfdf, toidf,
    k13_notes_df, sector) = (
            ivri.initialize_vetting_report_information(sector, cdips_cat_vnum)
    )

    mavp.make_all_vetting_reports(tfa_sr_paths, lcbasedir, resultsdir, cddf,
                                  supplementstatsdf, pfdf, toidf, k13_notes_df,
                                  sector=sector)


if __name__ == "__main__":

    main(sector=9, cdips_cat_vnum=0.4)
