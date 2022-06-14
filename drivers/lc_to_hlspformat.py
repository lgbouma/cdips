"""
TO PRODUCE CANDIDATES FROM CDIPS LCS
----------

Merges steps 2-4 of "HOWTO.md". Goes from cdips-pipeline light curves to
HLSP-formatted light curves.

After running, period-finding can be run on the light curves for which it is
expected to be worthwhile, in `do_initial_period_finding`.

USAGE:
    python -u lc_to_hlspformat.py &> logs/s14_to_hlsp.log &
"""

import os, shutil
from glob import glob
import multiprocessing as mp

from cdips.lcproc import trex_lc_to_mast_lc as tlml
import get_cdips_lc_stats as get_cdips_lc_stats
from how_many_cdips_stars_on_silicon import how_many_cdips_stars_on_silicon

def main():

    ##########################################
    sector = 15
    outdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/'
    overwrite = 0
    cams = [1,2,3,4]
    ccds = [1,2,3,4]
    OC_MG_CAT_ver = 0.6
    cdipsvnum = 1
    ##########################################

    nworkers = mp.cpu_count()

    lcpaths = glob(os.path.join(outdir, 'sector-{}'.format(sector),
                                'cam?_ccd?', 'hlsp*.fits'))

    # turn cdips-pipeline light curves to HLSP light curves
    if len(lcpaths) == 0 or overwrite:
        tlml.trex_lc_to_mast_lc(sectors=[sector], cams=cams, ccds=ccds,
                                make_symlinks=1, reformat_lcs=1,
                                OC_MG_CAT_ver=OC_MG_CAT_ver,
                                cdipsvnum=cdipsvnum, outdir=outdir)
    else:
        print('found {} HLSP LCs; wont reformat'.format(len(lcpaths)))

    # get stats, make the supp data file, print out the metadata, and move
    # allnan light curves
    statsfile = os.path.join(
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_stats',
        'sector-{}'.format(sector),
        'cdips_lc_statistics.txt'
    )
    suppstatsfile = statsfile.replace('cdips_lc_statistics',
                                      'supplemented_cdips_lc_statistics')
    if not os.path.exists(suppstatsfile) and not overwrite:
        get_cdips_lc_stats.main(sector, OC_MG_CAT_ver, cdipsvnum, overwrite)
    else:
        print('found {}'.format(suppstatsfile))

    # see how many LCs were expected
    outpath = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/star_catalog/'+
        'how_many_cdips_stars_on_silicon_sector{}.txt'.
        format(sector)
    )
    if not os.path.exists(outpath) and not overwrite:
        how_many_cdips_stars_on_silicon(sector=sector, ver=OC_MG_CAT_ver)
    else:
        print('found {}'.format(outpath))


if __name__=="__main__":
    main()
