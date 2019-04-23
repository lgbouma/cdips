"""
after get_lightcurves_for_cluster_stars, you can run some period-finding
routines
"""
import numpy as np, pandas as pd
from astropy.io import fits
from glob import glob
import os

from astrobase import periodbase, checkplot

def do_period_finding_fitslc(lcpath, ap=2):

    hdulist = fits.open(lcpath)
    hdr, lc = hdulist[0].header, hdulist[1].data

    times, mags, errs = (
        lc['TMID_BJD'],
        lc['TFA{}'.format(ap)],
        lc['IRE{}'.format(ap)]
    )

    #glsp = periodbase.pgen_lsp(times,mags,errs)
    spdm = periodbase.stellingwerf_pdm(times,mags,errs)
    blsp = periodbase.bls_parallel_pfind(times,mags,errs,startp=1.0)

    objectinfo = {}
    keys = ['objectid','ra','decl','pmra','pmdecl','teff','gmag']
    hdrkeys = ['Gaia-ID', 'RA[deg]', 'Dec[deg]', 'PM_RA[mas/yr]',
               'PM_Dec[mas/year]', 'teff_val', 'phot_g_mean_mag']
    for k,hk in zip(keys,hdrkeys):
        if hk in hdr:
            objectinfo[k] = hdr[hk]
        else:
            objectinfo[k] = np.nan

    outdir = os.path.dirname(lcpath)
    outfile = os.path.basename(lcpath).replace(
        '.fits', '_spdm_blsp_checkplot.png'
    )
    outpath = os.path.join(outdir, outfile)

    cp = checkplot.twolsp_checkplot_png(blsp, spdm, times, mags, errs,
                                        objectinfo=objectinfo,
                                        outfile=outpath,
                                        sigclip=30,
                                        plotdpi=100,
                                        phasebin=3e-2,
                                        phasems=6.0,
                                        phasebinms=12.0,
                                        unphasedms=6.0)
    print('did {}'.format(outpath))


def do_period_finding_cluster(clustername, runname, ap=2):
    lcdir = (
        '../data/cluster_data/lightcurves/{}_{}'.format(clustername, runname)
    )
    lcpaths = glob(os.path.join(lcdir,'*_llc.fits'))

    for lcpath in lcpaths:

        do_period_finding_fitslc(lcpath)


if __name__ == "__main__":
    clustername = 'Blanco_1'
    runname = 'ISP_1-2-1186'
    do_period_finding_cluster(clustername, runname)