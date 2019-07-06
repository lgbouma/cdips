"""
Make multipage PDFs needed to vet CDIPS objects of interest. (TCEs. Whatever).

python -u make_vetting_multipg_pdf.py &> logs/vetting_pdf.log &
"""
from glob import glob
import datetime, os, pickle, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cdips.plotting import vetting_pdf as vp
from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.vetting import centroid_analysis as cdva

from numpy import array as nparr
from astropy.io import fits
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy import units as u

def make_vetting_multipg_pdf(tfa_sr_path, lcpath, outpath, mdf, sourceid,
                             supprow, suppfulldf, pfdf, pfrow, toidf, sector,
                             mask_orbit_edges=True,
                             nworkers=40):
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

        pfdf: dataframe with period finding results for everything from this
        sector. good to check on matching ephemerides.

        toidf: dataframe with alerted TOI results
    """

    hdul_sr = fits.open(tfa_sr_path)
    hdul = fits.open(lcpath)

    lc_sr = hdul_sr[1].data
    lc, hdr = hdul[1].data, hdul[0].header

    # define "detrended mag". by default, this is the TFASR signal.  however,
    # if residual stellar variability was found after TFA detrending, then,
    # this is defined as the TFA LC + penalized spline detrending.

    is_pspline_dtr = bool(pfrow['pspline_detrended'].iloc[0])

    # Create the PdfPages object to which we will save the pages...
    with PdfPages(outpath) as pdf:

        ##########
        # page 1
        ##########
        fluxap = 'TFA2' if is_pspline_dtr else 'TFASR2'
        fig, tlsp, spdm = vp.two_periodogram_checkplot(
            lc_sr, hdr, supprow, pfrow, mask_orbit_edges=mask_orbit_edges,
            fluxap=fluxap, nworkers=nworkers)
        pdf.savefig(fig)
        plt.close()

        ##########
        # page 2
        ##########
        ap_index=2
        time, rawmag, tfasrmag, bkgdval, tfatime = (
            lc['TMID_BJD'],
            lc['IRM2'],
            lc_sr[fluxap],
            lc['BGV'],
            lc_sr['TMID_BJD']
        )

        t0, per = tlsp['tlsresult'].T0, tlsp['tlsresult'].period
        midtimes = np.array([t0 + ix*per for ix in range(-100,100)])
        obsd_midtimes = midtimes[ (midtimes > np.nanmin(time)) &
                                 (midtimes < np.nanmax(time)) ]
        tmag = hdr['TESSMAG']
        customstr = '\nT = {:.1f}'.format(float(tmag))

        fig = vp.plot_raw_tfa_bkgd(time, rawmag, tfasrmag, bkgdval, ap_index,
                                   supprow, pfrow,
                                   obsd_midtimes=obsd_midtimes,
                                   xlabel='BJDTDB', customstr=customstr,
                                   tfatime=tfatime, is_tfasr=True,
                                   figsize=(30,20))
        pdf.savefig(fig)
        plt.close()

        ##########
        # page 3 -- it's a QLP ripoff
        ##########
        fig, infodict = vp.transitcheckdetails(
            tfasrmag, tfatime, tlsp, mdf, hdr, supprow, pfrow,
            obsd_midtimes=obsd_midtimes, figsize=(30,20)
        )
        pdf.savefig(fig)
        plt.close()

        ##########
        # page 4 
        ##########
        fig = vp.scatter_increasing_ap_size(lc_sr, pfrow, infodict=infodict,
                                            obsd_midtimes=obsd_midtimes,
                                            customstr=customstr,
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
        # page 6
        ##########

        ra_obj, dec_obj = hdr['RA_OBJ'], hdr['DEC_OBJ']
        c_obj = SkyCoord(ra_obj, dec_obj, unit=(u.deg), frame='icrs')

        t0,per,dur,sourceid = (float(pfrow['tls_t0']),
                               float(pfrow['tls_period']),
                               float(pfrow['tls_duration']),
                               int(pfrow['source_id'].iloc[0]) )

        outdir = ("/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_cutouts/"
                 "sector-{}_TFA_SR_pkl".format(sector))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        cd = cdva.measure_centroid(t0,per,dur,sector,sourceid,c_obj,outdir)

        #
        # check whether the measured ephemeris matches other TCEs.  cutoffs
        # below came by looking at distribution of errors on the QLP quoted
        # parameters.
        #
        tls_period, tls_t0 = nparr(pfdf['tls_period']), nparr(pfdf['tls_t0'])
        ras, decs = nparr(pfdf['ra_x']), nparr(pfdf['dec_x'])
        coords = SkyCoord(ras, decs, unit=(u.deg), frame='icrs')

        seps_px = c_obj.separation(coords).to(u.arcsec).value/21

        period_cutoff = 2e-3 # about 3 minutes
        t0_cutoff = 5e-3 # 7 minutes

        close_per = np.abs(tls_period - per) < period_cutoff
        close_t0 = np.abs(tls_t0 - t0) < t0_cutoff
        is_close = close_per & close_t0

        if len(seps_px[is_close]) > 1:

            _pfdf = pfdf.loc[is_close]
            _pfdf['seps_px'] = seps_px[is_close]
            _pfdf = _pfdf[_pfdf['source_id'] != sourceid]

        else:
            _pfdf = None

        fig = vp.centroid_plots(c_obj, cd, hdr, _pfdf, toidf, figsize=(30,24))
        if fig is not None:
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
         (float(infodict['psdepthratio'] + infodict['psdepthratioerr']) < 5.0)
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
                  supplementstatsdf, pfdf, toidf, sector=6,
                  cdipsvnum=1):

    for tfa_sr_path in tfa_sr_paths:

        sourceid = int(tfa_sr_path.split('gaiatwo')[1].split('-')[0].lstrip('0'))
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
            zsector=str(sector).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )

        lcpath = os.path.join(
            lcbasedir,
            'cam{}_ccd{}'.format(cam, ccd),
            lcname
        )

        # logic: even if you know it's nottransit, it's a TCE. therefore, the
        # pdf will be made. i just dont want to have to look at it. put it in a
        # separate directory.
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

        pfrow = pfdf.loc[pfdf['source_id']==sourceid]
        if len(pfrow) != 1:
            errmsg = 'expected exactly 1 source match in period find df'
            raise AssertionError(errmsg)

        if not os.path.exists(outpath) and not os.path.exists(nottransitpath):
            make_vetting_multipg_pdf(tfa_sr_path, lcpath, outpath, mdf,
                                     sourceid, supprow, suppfulldf, pfdf,
                                     pfrow,
                                     toidf, sector)
        else:
            print('found {}, continue'.format(outpath))


def main(sector=None, cdips_cat_vnum=None):

    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'vetting/'
        'sector-{}'.format(sector)
    )
    dirs = [resultsdir, os.path.join(resultsdir,'pdfs'),
            os.path.join(resultsdir,'pkls'),
            os.path.join(resultsdir,'nottransitpdfs'),
            os.path.join(resultsdir,'nottransitpkls')]
    for _d in dirs:
        if not os.path.exists(_d):
            os.mkdir(_d)

    tfasrdir = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                'CDIPS_LCS/sector-{}_TFA_SR'.format(sector))
    lcbasedir =  ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
                  'CDIPS_LCS/sector-{}/'.format(sector))

    cdips_df = ccl.get_cdips_catalog(ver=cdips_cat_vnum)
    cddf = ccl.get_cdips_pub_catalog(ver=cdips_cat_vnum)
    import IPython; IPython.embed()
    assert 0
    #FIXME

    supppath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
                'cdips_lc_stats/sector-{}/'.format(sector)+
                'supplemented_cdips_lc_statistics.txt')
    supplementstatsdf = pd.read_csv(supppath, sep=';')

    pfpath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
              'cdips/results/cdips_lc_periodfinding/'
              'sector-{}/'.format(sector)+
              'initial_period_finding_results_supplemented.csv')
    pfdf = pd.read_csv(pfpath)

    toipath = ('/nfs/phtess2/ar0/TESS/PROJ/lbouma/'
              'cdips/data/toi-plus-2019-06-25.csv')
    toidf = pd.read_csv(toipath)

    # reconstructive_tfa/RunTFASR.sh applied the threshold cutoff on TFA_SR
    # lightcurves. use whatever is in `tfasrdir` to determine which sources to
    # make pdfs for.
    tfa_sr_paths = glob(os.path.join(tfasrdir, '*_llc.fits'))

    make_all_pdfs(tfa_sr_paths, lcbasedir, resultsdir, cdips_df,
                  supplementstatsdf, pfdf, toidf, sector=sector)


if __name__ == "__main__":

    main(sector=7, cdips_cat_vnum=0.3)
