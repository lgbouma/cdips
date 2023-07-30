"""
TO PRODUCE CANDIDATES FROM CDIPS LCS
----------

Merges steps 2-4 of "HOWTO.md". Goes from cdips-pipeline light curves to
HLSP-formatted light curves.

After running, period-finding can be run on the light curves for which it is
expected to be worthwhile, in `do_initial_period_finding`.

USAGE:
    (trex_37 environment) python -u lc_to_hlspformat.py &> logs/s14_to_hlsp.log &
"""

import os, shutil
from glob import glob
import multiprocessing as mp
from os.path import join

from cdips.lcproc import trex_lc_to_mast_lc as tlml
import get_cdips_lc_stats as get_cdips_lc_stats
from how_many_cdips_stars_on_silicon import how_many_cdips_stars_on_silicon

def main():

    ##########################################
    sector = 40
    ## on phtess[N] systems
    #outdir = '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/'
    #symlinkdir = '/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS/'
    #lcbasedir = '/nfs/phtess2/ar0/TESS/FFI/LC/FULL/'
    ## on wh1
    outdir = '/ar1/PROJ/luke/proj/CDIPS_LCS/'
    symlinkdir = '/ar1/PROJ/luke/proj/CDIPS_SYMLINKS/'
    lcbasedir = '/ar1/TESS/FFI/LC/FULL'

    overwrite = 0
    cams = [1,2,3,4]
    ccds = [1,2,3,4]
    OC_MG_CAT_ver = 0.6
    cdipsvnum = 1

    make_symlinks = 1
    ##########################################

    nworkers = mp.cpu_count()

    lcpaths = glob(os.path.join(outdir, 'sector-{}'.format(sector),
                                'cam?_ccd?', 'hlsp*.fits'))

    # turn cdips-pipeline light curves to HLSP light curves
    if len(lcpaths) == 0 or overwrite:
        tlml.trex_lc_to_mast_lc(
            sectors=[sector], cams=cams, ccds=ccds, make_symlinks=make_symlinks,
            reformat_lcs=1, OC_MG_CAT_ver=OC_MG_CAT_ver, cdipsvnum=cdipsvnum,
            outdir=outdir, symlinkdir=symlinkdir, lcbasedir=lcbasedir
        )
    else:
        print('found {} HLSP LCs; wont reformat'.format(len(lcpaths)))

    print('🎉 reformatted!')

    # get stats, make the supp data file, print out the metadata, and move
    # allnan light curves
    from cdips.paths import RESULTSDIR
    if not os.path.exists(RESULTSDIR): os.mkdir(RESULTSDIR)
    lcstatsdir = join(RESULTSDIR, "cdips_lc_stats")
    if not os.path.exists(lcstatsdir): os.mkdir(lcstatsdir)
    lcstatsdir = join(RESULTSDIR, "cdips_lc_stats", f'sector-{sector}')
    if not os.path.exists(lcstatsdir): os.mkdir(lcstatsdir)

    statsfile = join(lcstatsdir, 'cdips_lc_statistics.txt')
    suppstatsfile = statsfile.replace('cdips_lc_statistics',
                                      'supplemented_cdips_lc_statistics')
    if not os.path.exists(suppstatsfile) and not overwrite:
        get_cdips_lc_stats.main(sector, OC_MG_CAT_ver, cdipsvnum, overwrite,
                                filesystem=filesystem)
    else:
        print('found {}'.format(suppstatsfile))

    print('🎉 got stats, moved allnan light curves!')

    # see how many LCs were expected
    run_howmany = 0
    if run_howmany:
        outpath = (
            '/nfs/php1/ar0/TESS/PROJ/lbouma/cdips/results/star_catalog/'+
            'how_many_cdips_stars_on_silicon_sector{}.txt'.
            format(sector)
        )
        if not os.path.exists(outpath) and not overwrite:
            how_many_cdips_stars_on_silicon(sector=sector, ver=OC_MG_CAT_ver)
        else:
            print('found {}'.format(outpath))


if __name__=="__main__":
    main()
