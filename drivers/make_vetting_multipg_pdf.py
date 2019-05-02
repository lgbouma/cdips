"""
Make multipage PDFs needed to vet CDIPS objects of interest. (TCEs. Whatever).
"""
from glob import glob
import datetime, os, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cdips.plotting import vetting_pdf as vp

from numpy import array as nparr
from astropy.io import fits
from datetime import datetime

def make_vetting_multipg_pdf(tfa_sr_path, lcpath, outpath, mdf, sourceid,
                             mask_orbit_edges=True,
                             nworkers=32):
    """
    tfa_sr_path: path to signal-reconstructed TFA lightcurve.
    lcpath: path to "main" lightcurve.
    outpath: where we're saving the vetting pdf
    mdf: single row dataframe from CDIPS source catalog
    """

    hdul_sr = fits.open(tfa_sr_path)
    hdul = fits.open(lcpath)

    lc_sr = hdul_sr[1].data
    lc, hdr = hdul[1].data, hdul[0].header

    # Create the PdfPages object to which we will save the pages...
    with PdfPages(outpath) as pdf:

        ##########
        # page 1
        ##########
        fig, tlsp, spdm = vp.two_periodogram_checkplot(
            lc_sr, hdr, mask_orbit_edges=mask_orbit_edges, fluxap='TFASR2',
            nworkers=nworkers)
        pdf.savefig(fig)
        plt.close()

        ##########
        # page 2
        ##########
        ap_index=2
        time, rawmag, tfasrmag, bkgdval, tfatime = (
            lc['TMID_BJD'],
            lc['IRM2'],
            lc_sr['TFASR2'],
            lc['BGV'],
            lc_sr['TMID_BJD']
        )
        t0, per = tlsp['tlsresult'].T0, tlsp['tlsresult'].period
        midtimes = np.array([t0 + ix*per for ix in range(-100,100)])
        obsd_midtimes = midtimes[ (midtimes > np.nanmin(time)) &
                                  (midtimes < np.nanmax(time)) ]
        fig = vp.plot_raw_tfa_bkgd(time, rawmag, tfasrmag, bkgdval, ap_index,
                                   obsd_midtimes=obsd_midtimes,
                                   xlabel='BJDTDB', customstr='',
                                   tfatime=tfatime, is_tfasr=True,
                                   figsize=(30,20))
        pdf.savefig(fig)
        plt.close()

        ##########
        # page 3 -- it's a QLP ripoff
        ##########
        fig, infodict = vp.transitcheckdetails(
            tfasrmag, tfatime, tlsp, mdf, hdr, obsd_midtimes=obsd_midtimes,
            figsize=(30,20))
        pdf.savefig(fig)
        plt.close()

        ##########
        # page 4 
        ##########
        fig = vp.scatter_increasing_ap_size(lc_sr, infodict=infodict,
                                            obsd_midtimes=obsd_midtimes,
                                            xlabel='BJDTDB', figsize=(30,20))
        pdf.savefig(fig)
        plt.close()

        ##########
        # set the file's metadata via the PdfPages object:
        ##########
        d = pdf.infodict()
        d['Title'] = 'CDIPS vetting report for GAIADR2-{}'.format(sourceid)
        d['Author'] = 'Luke Bouma'
        d['Keywords'] = 'STARS & PLANETS'
        d['CreationDate'] = datetime.today()
        d['ModDate'] = datetime.today()

        picklepath = outpath.replace('pdfs','pkls').replace('.pdf','.pkl')
        with open(picklepath,'wb') as f:
            pickle.dump(infodict, f)
        print('made {}'.format(picklepath))


def make_all_pdfs(tfa_sr_paths, lcbasedir, resultsdir, cdips_df):

    for tfa_sr_path in tfa_sr_paths:

        sourceid = int(os.path.basename(tfa_sr_path).split('_')[0])
        mdf = cdips_df[cdips_df['source_id']==sourceid]
        if len(mdf) != 1:
            errmsg = 'expected exactly 1 source match in CDIPS cat'
            raise AssertionError(errmsg)

        hdul = fits.open(tfa_sr_path)
        hdr = hdul[0].header
        cam, ccd = hdr['CAMERA'], hdr['CCD']
        hdul.close()

        lcpath = os.path.join(
            lcbasedir,
            'cam{}_ccd{}'.format(cam, ccd),
            os.path.basename(tfa_sr_path)
        )

        outpath = os.path.join(
            resultsdir,'pdfs',
            'vet_'+os.path.basename(tfa_sr_path).replace('.fits','.pdf')
        )

        if not os.path.exists(outpath):
            make_vetting_multipg_pdf(tfa_sr_path, lcpath, outpath, mdf,
                                     sourceid)
        else:
            print('found {}, continue'.format(outpath))


def main(sectornum=6, cdips_cat_vnum=0.2):

    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'vetting/'
        'sector-{}'.format(sectornum)
    )
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)
        os.mkdir(os.path.join(resultsdir,'pdfs'))
        os.mkdir(os.path.join(resultsdir,'pkls'))

    tfasrdir = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                'CDIPS_LCS/sector-{}_TFA_SR'.format(sectornum))
    lcbasedir =  ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                  'CDIPS_LCS/sector-{}/'.format(sectornum))

    cdipscatpath = ('/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
                    'OC_MG_FINAL_GaiaRp_lt_16_v{}.csv'.format(cdips_cat_vnum))
    cdips_df = pd.read_csv(cdipscatpath, sep=';')

    # reconstructive_tfa/RunTFASR.sh applied the SDE cutoff on TFA_SR
    # lightcurves. use whatever is in `tfasrdir` to determine which sources to
    # make pdfs for.
    tfa_sr_paths = glob(os.path.join(tfasrdir, '*_llc.fits'))

    make_all_pdfs(tfa_sr_paths, lcbasedir, resultsdir, cdips_df)

if __name__ == "__main__":
    main(sectornum=6)
