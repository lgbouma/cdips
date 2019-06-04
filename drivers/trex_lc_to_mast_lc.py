"""
pipe-trex outputs fits lightcurves. a few final touches aren't particularly
worth implementing in pipe-trex proper, since they're very specific to the
CDIPS project. they include:

    * symlinking LCs of CDIPS stars into a reasonable directory structure
    (a replacement of the main function of utils/collect_cdips_lightcurves.py)

    * copying from these symlinks, while reformatting the headers to reflect
    the additional knowledge we have about them (i.e. what cluster are they in?
    according to who?). further, during this reformatting: add any niceties
    needed for public use.

python -u trex_lc_to_mast_lc.py &> logs/reformat_trex_to_mast_s6_cam1to4_ccd1to4.log &
"""

import os, shutil
from glob import glob

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.lcproc import reformat_lcs_for_mast as rlm
from cdips.lcproc import mask_orbit_edges as moe

def main():

    trex_lc_to_mast_lc(
        sectors=[6],
        cams=[1,2,3,4],
        ccds=[1,2,3,4],
        make_symlinks=0,
        make_plots=0,
        reformat_lcs=1
    )

def trex_lc_to_mast_lc(
    make_symlinks=1,
    make_plots=0,
    reformat_lcs=1,
    sectors=None,
    cams=None,
    ccds=None,
    symlinkdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS/',
    outdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/',
    OC_MG_CAT_ver=0.3,
    cdipsvnum=1
):

    if make_symlinks:
        ccl.make_local_lc_directories(sectors=sectors, cams=cams, ccds=ccds)
        cdips_sourceids = ccl.get_cdips_sourceids(ver=OC_MG_CAT_ver)
        ccl.symlink_cdips_lcs(cdips_sourceids, sectors=sectors, cams=cams, ccds=ccds)

    if make_plots:
        ccl.plot_cdips_lcs(sectors=sectors, cams=cams)

    if reformat_lcs:

        for sector in sectors:

            sectordir = os.path.join(outdir, 'sector-{}'.format(sector))

            if not os.path.exists(sectordir):
                os.mkdir(sectordir)

            for cam in cams:
                for ccd in ccds:

                    camccddir = os.path.join(sectordir,
                                             'cam{}_ccd{}'.format(cam,ccd))
                    if not os.path.exists(camccddir):
                        os.mkdir(camccddir)

                    lcpaths = glob(os.path.join(symlinkdir,
                                                'sector-{}'.format(sector),
                                                'cam{}_ccd{}'.format(cam,ccd),
                                                '*_llc.fits'
                                               )
                                  )

                    if len(lcpaths) > 0:

                       # NOTE: nt actually faster
                       #rlm.parallel_reformat_headers(lcpaths, camccddir,
                       #                              sector, cdipsvnum,
                       #                              nworkers=1,
                       #                              maxworkertasks=1000)

                       rlm.reformat_headers(lcpaths, camccddir, sector,
                                             cdipsvnum)


if __name__ == "__main__":
    main()
