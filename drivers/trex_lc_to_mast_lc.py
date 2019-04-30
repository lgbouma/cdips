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
"""

import os, shutil
from glob import glob

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.lcproc import reformat_lcs_for_mast as rlm
from cdips.lcproc import mask_orbit_edges as moe

def trex_lc_to_mast_lc(
    make_symlinks=1,
    make_plots=0,
    reformat_lcs=1,
    mask_orbit_start_and_end=1,
    sectors=None,
    cams=None,
    ccds=None,
    symlinkdir='/nfs/phtess1/ar1/TESS/PROJ/lbouma/CDIPS_SYMLINKS/',
    outdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/',
    OC_MG_CAT_ver=0.2
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
                        rlm.reformat_headers(lcpaths, camccddir)

    #TODO : implement. it's worth it. 5% less data. >>5% improvement.
    if mask_orbit_start_and_end:

        for sector in sectors:
            for cam in cams:
                for ccd in ccds:

                    camccddir = os.path.join(sectordir,
                                             'sector-{}'.format(sector),
                                             'cam{}_ccd{}'.format(cam,ccd))
                    lcpaths = glob(os.path.join(camccddir, '*_llc.fits'))

                    if len(lcpaths) > 0:
                        raise NotImplementedError
                        moe.mask_orbit_start_and_end_given_lcpaths(lcpaths)


if __name__ == "__main__":

    trex_lc_to_mast_lc(
        sectors=[6],
        cams=[4],
        ccds=[1,2,3,4],
        make_symlinks=1,
        make_plots=0,
        reformat_lcs=1,
        mask_orbit_start_and_end=0
    )
