"""
Given directories of symlinked PIPE-TREX-output FTIS LCs, make the headers
presentable, and apply minor changes to improve their BLS-searchability.
"""

import pandas as pd, numpy as np
import os
from astropy.io import fits
from datetime import datetime

cdips_cat_file = ('/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
                  'OC_MG_FINAL_GaiaRp_lt_16_v0.2.csv')

def _map_timeseries_key_to_comment(k):
    kcd = {
        "tmid_utc": "Exp mid-time in JD_UTC (from DATE-OBS,DATE-END)",
        "tmid_bjd": "Exp mid-time in BJD_TDB (BJDCORR applied)",
        "bjdcorr": "BJD_TDB = JD_UTC + TDBCOR + BJDCORR",
        "rstfc" : "Unique frame key",
        "starid": "GAIA ID of the object",
        "xcc"   : "Original X coordinate on CCD on photref frame",
        "ycc"   : "Original y coordinate on CCD on photref frame",
        "xic"   : "Shifted X coordinate on CCD on subtracted frame",
        "yic"   : "Shifted Y coordinate on CCD on subtracted frame",
        "fsv"   : "Measured S value (see Pal 2009 eq 31)",
        "fdv"   : "Measured D value (see Pal 2009 eq 31)",
        "fkv"   : "Measured K value (see Pal 2009 eq 31)",
        "bgv"   : "Background value (after bkgd surface subtrxn)",
        "bge"   : "Background measurement error",
        "ifl1"  : "Flux in aperture 1 (ADU)",
        "ife1"  : "Flux error in aperture 1 (ADU)",
        "irm1"  : "Instrumental mag in aperture 1",
        "ire1"  : "Instrumental mag error for aperture 1",
        "irq1"  : "Instrumental quality flag ap 1, 0/G OK, X bad",
        "ifl2"  : "Flux in aperture 2 (ADU)",
        "ife2"  : "Flux error in aperture 2 (ADU)",
        "irm2"  : "Instrumental mag in aperture 2",
        "ire2"  : "Instrumental mag error for aperture 2",
        "irq2"  : "Instrumental quality flag ap 2, 0/G OK, X bad",
        "ifl3"  : "Flux in aperture 3 (ADU)",
        "ife3"  : "Flux error in aperture 3 (ADU)",
        "irm3"  : "Instrumental mag in aperture 3",
        "ire3"  : "Instrumental mag error for aperture 3",
        "irq3"  : "Instrumental quality flag ap 3, 0/G OK, X bad",
        'tfa1'  : "Trend-filtered magnitude in aperture 1",
        'tfa2'  : "Trend-filtered magnitude in aperture 2",
        'tfa3'  : "Trend-filtered magnitude in aperture 3",
        "ccdtemp" : "Mean CCD temperature S_CAM_ALCU_sensor_CCD",
        "ntemps"  : "Number of temperatures avgd to get ccdtemp",
        'dtr_isub': "Img subtraction photometry performed",
        'dtr_epd' : "EPD detrending performed",
        'dtr_tfa' : "TFA detrending performed",
        'projid' :  "PIPE-TREX identifier for software version",
        'btc_ra' : "Right ascen in barycentric time correction",
        'btc_dec' : "Declination in barycentric time correction",
        'rdistpx': "Distance from pixel (1024,1024) [pixels]",
        'thetadeg': "Azimuth angle from pixel center [degrees]"
    }
    return kcd[k]


def _reformat_header(lcpath, cdips_df, outdir):

    hdul = fits.open(lcpath)
    primaryhdr, hdr, data = (
        hdul[0].header, hdul[1].header, hdul[1].data
    )
    hdul.close()

    lcgaiaid = os.path.basename(lcpath).split('_')[0]
    info = cdips_df.loc[cdips_df['source_id'] == np.int64(lcgaiaid)]

    #
    # set CDIPS key/value/comments in primary header.
    #
    primaryhdr.set('CDIPSREF',
                   info['reference'].iloc[0],
                   'Catalog(s) w/ cluster membrshp [,sep]')

    primaryhdr.set('CDCLSTER',
                   info['cluster'].iloc[0],
                   'Name(s) of cluster in CDIPSREF [,sep]')

    primaryhdr.set('CDEXTCAT',
                   info['ext_catalog_name'].iloc[0],
                   'Name(s) of star in CDIPSREF [,sep]')

    primaryhdr.set('CDXMDIST',
                   str(info['dist'].iloc[0]),
                   '[deg] dist btwn CDIPSREF & GAIADR2 locn')

    primaryhdr.set('TIMEUNIT',
                   primaryhdr['TIMEUNIT'],
                   'Time unit for TMID_BJD')

    toremove = ['TSTOP', 'TSTART', 'DATE-END', 'DATE-OBS', 'TELAPSE',
                'LIVETIME']
    for _toremove in toremove:
        primaryhdr.remove(_toremove)

    primaryhdr.set('Gaia-ID',
                   primaryhdr['Gaia-ID'],
                   'GaiaDR2 source_id. ->lum_val from same')

    #
    # set timeseries extension header key comments
    #
    tfields = hdr['TFIELDS'] # number of table fields
    hdrkv = {}
    for ind in range(1,tfields+1):
        key = 'TTYPE{}'.format(ind)
        hdrkv[key] = hdr[key]
    for k,v in hdrkv.items():
        hdr.comments[k] = _map_timeseries_key_to_comment(v.lower())

    #
    # write it (!)
    #
    primary_hdu = fits.PrimaryHDU(header=primaryhdr)
    timeseries_hdu = fits.BinTableHDU(header=hdr, data=data)
    timeseries_hdu.header = hdr
    outhdulist = fits.HDUList([primary_hdu, timeseries_hdu])

    outfile = os.path.join(outdir, os.path.basename(lcpath))

    try:
        outhdulist.writeto(outfile, overwrite=False)
        print('{}: reformatted {}, wrote to {}'.format(
            datetime.utcnow().isoformat(), lcpath, outfile))
    except:
        import IPython; IPython.embed()

    outhdulist.close()


def reformat_headers(lcpaths, outdir):

    cdips_df = pd.read_csv(cdips_cat_file, sep=';')

    for lcpath in lcpaths:
        outpath = os.path.join(outdir, os.path.basename(lcpath))
        if not os.path.exists(outpath):
            _reformat_header(lcpath, cdips_df, outdir)


def mask_orbit_start_and_end(lcpaths):
    pass
