"""
Make multipage PDFs needed to vet CDIPS objects of interest. (TCEs. Whatever).
"""
from glob import glob
import datetime, os, pickle, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cdips.plotting import vetting_pdf as vp

from numpy import array as nparr
from astropy.io import fits
from datetime import datetime

def make_vetting_multipg_pdf(tfa_sr_path, lcpath, outpath, mdf, sourceid,
                             supprow, suppfulldf,
                             mask_orbit_edges=True,
                             nworkers=32):
    """
    args:

        tfa_sr_path: path to signal-reconstructed TFA lightcurve.

        lcpath: path to "main" lightcurve.

        outpath: where we're saving the vetting pdf, by default (if
        isobviouslynottransit is False)

        mdf: single row dataframe from CDIPS source catalog

        supprow: dataframe with LC statistics, Gaia xmatch info, CDIPS xmatch
        info. Cut to match the sourceid of whatever object is getting the PDF
        made.  Columns include:

           'lcobj', 'cat_mag', 'med_rm1', 'mad_rm1', 'mean_rm1', 'stdev_rm1',
           'ndet_rm1', 'med_sc_rm1', 'mad_sc_rm1', 'mean_sc_rm1',
           ...
           '#Gaia-ID[1]', 'RA[deg][2]', 'Dec[deg][3]', 'RAError[mas][4]',
           'DecError[mas][5]', 'Parallax[mas][6]', 'Parallax_error[mas][7]',
           'PM_RA[mas/yr][8]', 'PM_Dec[mas/year][9]', 'PMRA_error[mas/yr][10]',
           'PMDec_error[mas/yr][11]', 'Ref_Epoch[yr][12]', 'phot_g_mean_mag[20]',
           'phot_bp_mean_mag[25]', 'phot_rp_mean_mag[30]', 'radial_velocity[32]',
           'radial_velocity_error[33]', 'teff_val[35]',
           'teff_percentile_lower[36]', 'teff_percentile_upper[37]', 'a_g_val[38]',
           'a_g_percentile_lower[39]', 'a_g_percentile_upper[40]',
           'e_bp_min_rp_val[41]', 'e_bp_min_rp_percentile_lower[42]',
           'e_bp_min_rp_percentile_upper[43]', 'radius_val[44]',
           'radius_percentile_lower[45]', 'radius_percentile_upper[46]',
           'lum_val[47]', 'lum_percentile_lower[48]', 'lum_percentile_upper[49]'
           ...
           'cluster', 'ext_catalog_name', 'reference', 'source_id'

        suppfulldf: as above, but the whole dataframe for all CDIPS sources
        that got lightcurves for this sector. Useful for broader assessment of
        the LC within the sample of cluster lightcurves.
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
            tfasrmag, tfatime, tlsp, mdf, hdr, supprow,
            obsd_midtimes=obsd_midtimes, figsize=(30,20)
        )
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
        # page 5
        ##########
        fig = vp.cluster_membership_check(hdr, supprow, infodict, suppfulldf,
                                          figsize=(30,16))
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

        ##########
        # check if is obviously nottransit. this will be the case if:
        # * ndet_tf2 < 100
        # * depth < 0.85
        # * rp > 6 R_jup = 67.25 R_earth. Based on 2x the limit given at 1Myr
        #   in Burrows+2001, figure 3.
        # * SNR from TLS is < 8 (these will be unbelievable no matter what.
        #   they might exist b/c TFA SR could have lowered overall SDE/SNR)
        # * primary depth is >10%
        # * primary/secondary depth ratios from gaussian fitting in range of
        #   ~1.2-7, accounting for 1-sigma formal uncertainty in measurement
        ##########
        if (
        (float(supprow['ndet_tf2']) < 100) or
        (float(infodict['depth']) < 0.85) or
        (float(infodict['snr']) < 8) or
        (float(infodict['rp']) > 67.25) or
        ( (float(infodict['psdepthratio'] - infodict['psdepthratioerr']) > 1.2)
         &
          (float(infodict['psdepthratio'] + infodict['psdepthratioerr']) < 7.0)
        )
        ):
            isobviouslynottransit = True
        else:
            isobviouslynottransit = False

    if isobviouslynottransit:
        for path in [outpath, picklepath]:
            src = path
            dst = path.replace('pdfs','nottransitpdfs')
            shutil.move(src,dst)
            print('found was nottransit. moved {} -> {}'.format(src,dst))



def _get_supprow(sourceid, supplementstatsdf):

    mdf = supplementstatsdf.loc[supplementstatsdf['lcobj']==sourceid]

    return mdf

def make_all_pdfs(tfa_sr_paths, lcbasedir, resultsdir, cdips_df,
                  supplementstatsdf, sectornum=6,
                  cdipsvnum=1):

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

        lcname = (
            'hlsp_cdips_tess_ffi_'
            'gaiatwo{zsourceid}-{zsector}_'
            'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            zsourceid=str(sourceid).zfill(22),
            zsector=str(sectornum).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )

        lcpath = os.path.join(
            lcbasedir,
            'cam{}_ccd{}'.format(cam, ccd),
            lcname
        )

        # logic: even if you know it's nottransit, it's a TCE. therefore, the pdf will
        # be made. i just dont want to have to look at it. put it in a separate
        # directory.
        outpath = os.path.join(
            resultsdir,'pdfs',
            'vet_'+os.path.basename(tfa_sr_path).replace('.fits','.pdf')
        )
        nottransitpath = os.path.join(
            resultsdir,'nottransitpdfs',
            'vet_'+os.path.basename(tfa_sr_path).replace('.fits','.pdf')
        )

        supprow = _get_supprow(sourceid, supplementstatsdf)
        suppfulldf = supplementstatsdf

        if not os.path.exists(outpath) and not os.path.exists(nottransitpath):
            make_vetting_multipg_pdf(tfa_sr_path, lcpath, outpath, mdf,
                                     sourceid, supprow, suppfulldf)
        else:
            print('found {}, continue'.format(outpath))


def main(sectornum=6, cdips_cat_vnum=0.2):

    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'vetting/'
        'sector-{}'.format(sectornum)
    )
    dirs = [resultsdir, os.path.join(resultsdir,'pdfs'),
            os.path.join(resultsdir,'pkls'),
            os.path.join(resultsdir,'nottransitpdfs'),
            os.path.join(resultsdir,'nottransitpkls')]
    for _d in dirs:
        if not os.path.exists(_d):
            os.mkdir(_d)

    tfasrdir = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                'CDIPS_LCS/sector-{}_TFA_SR'.format(sectornum))
    lcbasedir =  ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                  'CDIPS_LCS/sector-{}/'.format(sectornum))

    cdipscatpath = ('/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
                    'OC_MG_FINAL_GaiaRp_lt_16_v{}.csv'.format(cdips_cat_vnum))
    cdips_df = pd.read_csv(cdipscatpath, sep=';')

    supppath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
                'cdips_lc_stats/sector-{}/'.format(sectornum)+
                'supplemented_cdips_lc_statistics.txt')
    supplementstatsdf = pd.read_csv(supppath, sep=';')

    # reconstructive_tfa/RunTFASR.sh applied the SDE cutoff on TFA_SR
    # lightcurves. use whatever is in `tfasrdir` to determine which sources to
    # make pdfs for.
    tfa_sr_paths = glob(os.path.join(tfasrdir, '*_llc.fits'))

    make_all_pdfs(tfa_sr_paths, lcbasedir, resultsdir, cdips_df,
                  supplementstatsdf, sectornum=sectornum)


if __name__ == "__main__":

    main(sectornum=6)
