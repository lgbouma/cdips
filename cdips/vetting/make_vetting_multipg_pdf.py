#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############
from glob import glob
import datetime, os, pickle, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cdips.plotting import vetting_pdf as vp
from cdips.vetting import (
    centroid_analysis as cdva,
    initialize_neighborhood_information as ini
)

from numpy import array as nparr
from astropy.io import fits
from datetime import datetime

from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

def make_vetting_multipg_pdf(lcpath, outpath, mdf, source_id,
                             supprow, suppfulldf, pfdf, pfrow, toidf, sector,
                             mask_orbit_edges=True, nworkers=40,
                             show_rvs=False):
    """
    args:

        lcpath: path to CDIPS HLSP lightcurve.  (preferred, since PCA1 is the
        default aperture)

        outpath: where we're saving the vetting pdf, by default (if
        isobviouslynottransit is False)

        mdf: single row dataframe from CDIPS source catalog

        supprow: dataframe with LC statistics, Gaia xmatch info, CDIPS xmatch
        info. Cut to match the source_id of whatever object is getting the PDF
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

    hdul = fits.open(lcpath)

    lc, hdr = hdul[1].data, hdul[0].header

    # Create the PdfPages object to which we will save the pages...
    with PdfPages(outpath) as pdf:

        ##########
        # page 1
        ##########
        rawap = 'IRM1'
        pcaap = 'PCA1'
        try:
            fig, tlsp, _, dtrdict = vp.two_periodogram_checkplot(
                lc, hdr, supprow, pfrow, mask_orbit_edges=mask_orbit_edges,
                fluxap=pcaap, nworkers=nworkers)
        except Exception as e:
            # NOTE: if this is raised, probably a detrending singularity issue.
            # perhaps fine-tune the cutoff further.
            raise('two_periodogram_checkplot failed: '+str(e))
            return
        pdf.savefig(fig)
        plt.close()
        if pd.isnull(tlsp):
            return

        ##########
        # page 2
        ##########
        ap_index=1
        time, rawmag, pcamag, bkgdval, pcatime = (
            lc['TMID_BJD'], lc[rawap], lc[pcaap], lc['BGV'], lc['TMID_BJD']
        )

        t0, per = tlsp['tlsresult'].T0, tlsp['tlsresult'].period
        midtimes = np.array([t0 + ix*per for ix in range(-100,100)])
        obsd_midtimes = midtimes[ (midtimes > np.nanmin(time)) &
                                 (midtimes < np.nanmax(time)) ]
        tmag = hdr['TESSMAG']
        customstr = '\nT = {:.1f}'.format(float(tmag))

        #FIXME FIXME TODO: err you might want to show the outlier points on
        #this?
        fig = vp.plot_raw_pca_bkgd(time, rawmag, pcamag, bkgdval, ap_index,
                                   supprow, pfrow,
                                   obsd_midtimes=obsd_midtimes,
                                   xlabel='BJDTDB', customstr=customstr,
                                   dtrdict=dtrdict,
                                   figsize=(30,20))
        pdf.savefig(fig)
        plt.close()

        ##########
        # page 3 -- it's a QLP ripoff
        ##########
        stime, sflux = dtrdict['search_time'], dtrdict['search_flux']
        fig, infodict = vp.transitcheckdetails(
            stime, sflux, tlsp, mdf, hdr, supprow, pfrow, dtrdict,
            obsd_midtimes=obsd_midtimes,
            figsize=(30,20)
        )
        pdf.savefig(fig)
        plt.close()

        ##########
        # page 4 
        ##########
        fig, apdict = vp.scatter_increasing_ap_size(
            lc, pfrow, infodict=infodict, obsd_midtimes=obsd_midtimes,
            customstr=customstr, xlabel='BJDTDB', figsize=(30,20),
            auto_depth_vs_apsize=True
        )
        pdf.savefig(fig)
        plt.close()

        ##########
        # page 5
        ##########

        ra_obj, dec_obj = hdr['RA_OBJ'], hdr['DEC_OBJ']
        c_obj = SkyCoord(ra_obj, dec_obj, unit=(u.deg), frame='icrs')

        t0,per,dur,source_id = (float(pfrow['tls_t0']),
                               float(pfrow['tls_period']),
                               float(pfrow['tls_duration']),
                               int(pfrow['source_id'].iloc[0]) )

        outdir = ("/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_cutouts/"
                 "sector-{}_TFA_SR_pkl".format(sector))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        cd = cdva.measure_centroid(t0,per,dur,sector,source_id,c_obj,outdir)

        catalog_to_gaussian_sep_arcsec = None
        if isinstance(cd, dict):
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
                _pfdf = _pfdf[_pfdf['source_id'] != source_id]

            else:
                _pfdf = None

            fig, catalog_to_gaussian_sep_arcsec = (
                vp.centroid_plots(c_obj, cd, hdr, _pfdf, toidf, figsize=(30,24))
            )

            if fig is not None:
                pdf.savefig(fig)
            plt.close()

        if not isinstance(catalog_to_gaussian_sep_arcsec, float):
            catalog_to_gaussian_sep_arcsec = 0
        infodict['catalog_to_gaussian_sep_arcsec'] = (
            catalog_to_gaussian_sep_arcsec
        )

        ##########
        # page 6
        ##########
        info = (
             ini.get_group_and_neighborhood_information(source_id, overwrite=0)
        )

        if isinstance(info, tuple):

            if DEBUG:
                picklepath = 'nbhd_info_{}.pkl'.format(source_id)
                with open(picklepath , 'wb') as f:
                    pickle.dump(info, f)
                    LOGINFO('made {}'.format(picklepath))

            (targetname, groupname, referencename, group_df_dr2, target_df,
             nbhd_df, pmdec_min, pmdec_max, pmra_min, pmra_max
            ) = info

            fig = vp.plot_group_neighborhood(
                targetname, groupname, referencename, group_df_dr2, target_df,
                nbhd_df, pmdec_min=pmdec_min, pmdec_max=pmdec_max,
                pmra_min=pmra_min, pmra_max=pmra_max, source_id=source_id,
                figsize=(30,20), show_rvs=show_rvs
            )

            pdf.savefig(fig)
            plt.close()

        elif info is None:

            info = ini.get_neighborhood_information(source_id, overwrite=0,
                                                    min_n_nbhrs=1000)

            if info is not None:

                (targetname, groupname, target_df, nbhd_df, pmdec_min,
                 pmdec_max, pmra_min, pmra_max) = info

                fig = vp.plot_neighborhood_only(
                    targetname, groupname, target_df, nbhd_df,
                    pmdec_min=pmdec_min, pmdec_max=pmdec_max,
                    pmra_min=pmra_min, pmra_max=pmra_max,
                    source_id=source_id, figsize=(30,20),
                )

                pdf.savefig(fig)
                plt.close()

        else:

            raise NotImplementedError

        LOGINFO(f'made {outpath}')

        ##########
        # set the file's metadata via the PdfPages object:
        ##########
        d = pdf.infodict()
        d['Title'] = 'CDIPS vetting report for GAIADR2-{}'.format(source_id)
        d['Author'] = 'Luke Bouma'
        d['Keywords'] = 'stars | planets'
        d['CreationDate'] = datetime.today()
        d['ModDate'] = datetime.today()

        picklepath = outpath.replace('pdfs','pkls').replace('.pdf','.pkl')
        with open(picklepath,'wb') as f:
            pickle.dump(infodict, f)
            LOGINFO(f'made {picklepath}')

        ##########
        # check if is obviously nottransit. this will be the case if:
        # * ndet_tf2 < 100
        # * depth < 0.85
        # * rp > 6 R_jup = 67.25 R_earth. Based on 2x the limit given at 1Myr
        #   in Burrows+2001, figure 3.
        # * rp > 3 R_jup and age > 1Gyr
        # * SNR from TLS is < 8 (these will be unbelievable no matter what.
        #   they might exist b/c TFA SR could have lowered overall SDE/SNR)
        # * primary/secondary depth ratios from gaussian fitting in range of
        #   ~1.2-5, accounting for 1-sigma formal uncertainty in measurement
        # * the OOT - intra image gaussian fit centroid is > 2 pixels off the
        #   catalog position.
        # * Kharchenko+2013 gave a cluster parallax, and the GaiaDR2 measured
        #   parallax is >5 sigma away from it. (these are 99% of the time
        #   backgruond stars).
        ##########

        time_key = 'mean_age'
        logt = str(supprow[time_key].iloc[0])

        if not pd.isnull(logt):
            if ',' in str(logt):
                logt_split = list(map(float,logt.split(',')))
                logt = np.mean(logt_split)
            if not pd.isnull(logt):
                logt = float(logt)

        isobviouslynottransit = False
        whynottransit = []
        if float(supprow['ndet_tf2']) < 100:
            isobviouslynottransit = True
            whynottransit.append('N_points < 100')

        if float(infodict['depth']) < 0.85:
            isobviouslynottransit = True
            whynottransit.append('depth < 0.85')

        if float(infodict['snr']) < 8:
            isobviouslynottransit = True
            whynottransit.append('SNR < 8')

        if isinstance(apdict, dict):
            if apdict['depth_vs_apsize_increasing']:
                isobviouslynottransit = True
                msg = (
                    'tcounts {} transits, depths {}, uncs {}'.
                    format(repr(apdict['tcounts']),
                           repr(apdict['depths']),
                           repr(apdict['depth_uncs']))
                )
                whynottransit.append(msg)

        if ((float(infodict['psdepthratio'] - infodict['psdepthratioerr']) > 1.2)
            &
            (float(infodict['psdepthratio'] + infodict['psdepthratioerr']) < 5.0)
        ):
            isobviouslynottransit = True
            whynottransit.append('1.2 < pri/occ < 5.0')

        if float(infodict['catalog_to_gaussian_sep_arcsec']) > 42:
            isobviouslynottransit = True
            whynottransit.append('catalog_to_gaussian_sep > 2px')

        if not pd.isnull(logt):
            if logt > 9 and float(infodict['rp']) > (3*u.Rjup).to(u.Rearth).value:
                isobviouslynottransit = True
                whynottransit.append('age>1Gyr and Rp > 3Rjup')

        if float(infodict['rp']) > (3.5*u.Rjup).to(u.Rearth).value:
            isobviouslynottransit = True
            whynottransit.append('Rp > 3.5Rjup')

        if isinstance(dtrdict, dict):

            # If LS period and TLS period are within 2% of each other, it means
            # that either a) it's an eclipse-dominated light curve (e.g., an
            # eclipsing binary, or a hot Jupiter without any rotation signal),
            # or that b) it's a rotation-dominated light curve that notch
            # triggers on the peaks of.

            ls_period = dtrdict['lsp_dict']['ls_period']
            tls_period = float(pfrow['tls_period'])

            cutoff = 2e-2

            close_tls_ls_period = (
                np.abs(tls_period - ls_period)/ls_period < cutoff
            )

            if close_tls_ls_period:
                isobviouslynottransit = True
                whynottransit.append(
                    f'LS_period {ls_period:.4f} ~= TLS_period {tls_period:.4f}'
                )

    if isobviouslynottransit:

        if len(whynottransit) > 1:
            whynottransit = '|'.join(whynottransit)
        else:
            assert len(whynottransit) == 1

        for path in [outpath, picklepath]:
            src = path
            dst = path.replace('pdfs','nottransitpdfs')
            shutil.move(src,dst)
            LOGINFO('found was nottransit. moved {} -> {}'.format(src,dst))

        # create text file that describes why it's been flagged as "not
        # transit".
        writepath = (
            outpath.replace(
                'pdfs','nottransitpdfs').replace(
                '.pdf','whynottransit.txt')
        )
        with open(writepath, 'w') as f:
            f.writelines(whynottransit)
        LOGINFO('Reason written to {}'.format(writepath))
