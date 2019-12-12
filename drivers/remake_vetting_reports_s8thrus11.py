"""
ONE-TIME to remake the vetting reports for PCs identified from sectors 8
through 11.  (After adding pg 7 and TLS fit overplot).

$ python -u make_all_vetting_reports.py &> logs/vetting_pdf.log &
"""

from cdips.vetting import (
    initialize_vetting_report_information as ivri,
    make_all_vetting_reports as mavp
)

import os
from glob import glob
import pandas as pd, numpy as np

def main(sector=None, cdips_cat_vnum=None):

    #
    # remake the vetting reports for already identified planet candidtaes
    #
    (tfa_sr_paths, lcbasedir, resultsdir,
    cddf, supplementstatsdf, pfdf, toidf,
    k13_notes_df, sector) = (
            ivri.initialize_vetting_report_information(sector, cdips_cat_vnum)
    )

    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/vetting/'
        'sector-{}_remake/'.format(sector)
    )
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)
    dirs = [resultsdir, os.path.join(resultsdir,'pdfs'),
            os.path.join(resultsdir,'pkls'),
            os.path.join(resultsdir,'nottransitpdfs'),
            os.path.join(resultsdir,'nottransitpkls')]
    for _d in dirs:
        if not os.path.exists(_d):
            os.mkdir(_d)

    classifxn_path = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'vetting_classifications/'
        'sector-{}_PCs_MERGED_SUBCLASSIFICATIONS.csv'.format(sector)
    )
    classifxn_df = pd.read_csv(classifxn_path, sep=';')

    classifxn_df['match_id'] = classifxn_df.Name.apply(
        lambda x: x.rstrip('.pdf').lstrip('vet_')
    )

    all_path_ids = np.array([
        os.path.basename(t).rstrip('.fits') for t in tfa_sr_paths
    ])

    sel = np.in1d(all_path_ids, np.array(classifxn_df.match_id))
    make_paths = np.array(tfa_sr_paths)[sel]
    assert len(make_paths) == len(classifxn_df)

    mavp.make_all_vetting_reports(
        make_paths, lcbasedir, resultsdir, cddf, supplementstatsdf, pfdf,
        toidf, k13_notes_df, sector=sector
    )


if __name__ == "__main__":

    for sector in [8,9,10,11]:
        main(sector=sector, cdips_cat_vnum=0.4)
