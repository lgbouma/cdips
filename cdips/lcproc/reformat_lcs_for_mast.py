"""
Given directories of symlinked PIPE-TREX-output FTIS LCs, make the headers
presentable, and apply minor changes to improve their BLS-searchability.
"""

import pandas as pd, numpy as np
import os
from astropy.io import fits
from datetime import datetime

cdips_cat_file = ('/nfs/phtess1/ar1/TESS/PROJ/lbouma/'
                  'OC_MG_FINAL_GaiaRp_lt_16_v0.3.csv')

def _map_timeseries_key_to_comment(k):
    kcd = {
        "tmid_utc": "Exp mid-time in JD_UTC (from DATE-OBS,DATE-END)",
        "tmid_bjd": "Exp mid-time in BJD_TDB (BJDCORR applied)",
        "bjdcorr": "BJD_TDB = JD_UTC + TDBCOR + BJDCORR",
        "rstfc" : "Unique frame key",
        "starid": "GAIA ID of the object",
        "xcc"   : "Original X coordinate on CCD on photref frame",
        "ycc"   : "Original Y coordinate on CCD on photref frame",
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


def _map_timeseries_key_to_unit(k):
    kcd = {
        "tmid_utc": "day",
        "tmid_bjd": "day",
        "bjdcorr": "day",
        "rstfc" : "unitless",
        "starid": "unitless",
        "xcc"   : "pixels",
        "ycc"   : "pixels",
        "xic"   : "pixels",
        "yic"   : "pixels",
        "fsv"   : "unitless",
        "fdv"   : "unitless",
        "fkv"   : "unitless",
        "bgv"   : "ADU",
        "bge"   : "ADU",
        "ifl1"  : "ADU",
        "ife1"  : "ADU",
        "irm1"  : "mag",
        "ire1"  : "mag",
        "irq1"  : "unitless",
        "ifl2"  : "ADU",
        "ife2"  : "ADU",
        "irm2"  : "mag",
        "ire2"  : "mag",
        "irq2"  : "unitless",
        "ifl3"  : "ADU",
        "ife3"  : "ADU",
        "irm3"  : "mag",
        "ire3"  : "mag",
        "irq3"  : "unitless",
        'tfa1'  : "mag",
        'tfa2'  : "mag",
        'tfa3'  : "mag",
        "ccdtemp" : "degcelcius",
        "ntemps"  : "unitless",
        'dtr_isub': "bool",
        'dtr_epd' : "bool",
        'dtr_tfa' : "bool",
        'projid' :  "unitless",
        'btc_ra' : "deg",
        'btc_dec' : "deg",
        'rdistpx': "pixels",
        'thetadeg': "degrees"
    }
    return kcd[k]


def _reformat_header(lcpath, cdips_df, outdir, sectornum, cdipsvnum):

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
                   'Cluster name(s) in CDIPSREF [,sep]')

    primaryhdr.set('CDEXTCAT',
                   info['ext_catalog_name'].iloc[0],
                   'Star name(s) in CDIPSREF [,sep]')

    primaryhdr.set('CDXMDIST',
                   str(info['dist'].iloc[0]),
                   '[deg] DIST btwn CDIPSREF & GAIADR2 locn')

    #
    # info for MAST
    #
    toremove = ['TSTOP', 'TSTART', 'DATE-END', 'DATE-OBS', 'TELAPSE',
                'LIVETIME', 'EXPOSURE',
                'DEADC', 'BJDREFF']

    for _toremove in toremove:
        primaryhdr.remove(_toremove)

    primaryhdr.set('TIMEUNIT',
                   primaryhdr['TIMEUNIT'],
                   'Time unit for TMID_BJD')

    primaryhdr.set('BJDREFF',
                   0,
                   'fraction of the day in BTJD reference date')

    primaryhdr.set('BJDREFI',
                   0,
                   'integer part of BTJD reference date')

    primaryhdr.set('FILTER',
                   'TESS',
                   'MAST HLSP required keyword')

    primaryhdr.set('OBJECT',
                   int(lcgaiaid),
                   'Gaia DR2 source_id')

    primaryhdr.set('SECTOR',
                   int(sectornum),
                   'Observing sector')

    primaryhdr.set('RADESYS',
                   'ICRS',
                   'reference frame of celestial coordinates')

    primaryhdr.set('EQUINOX',
                   2015.5,
                   'equinox of celestial coordinate system')

    primaryhdr.set('RA_OBJ',
                   primaryhdr['RA[deg]'],
                   '[deg] right ascension')

    primaryhdr.set('DEC_OBJ',
                   primaryhdr['Dec[deg]'],
                   '[deg] declination')

    toremove = ['RA[deg]', 'Dec[deg]']
    for _toremove in toremove:
        primaryhdr.remove(_toremove)


    primaryhdr.set('Gaia-ID',
                   primaryhdr['Gaia-ID'],
                   'GaiaDR2 source_id. ->lum_val from same')

    #
    # TICv8 xmatch info, once it exists...
    #
    primaryhdr.set('TICVER',
                   8,
                   'TIC version')

    primaryhdr.set('TESSMAG',
                   42, #FIXME
                   '[mag] TESS magnitude')

    primaryhdr.set('TICCONT',
                   42, #FIXME
                   'ContamRatio from TIC')

    #
    # who dun it
    #
    primaryhdr.set('ORIGIN',
                   'Bouma&team|CDIPS|Princeton',
                   'Author|Project|Institution')

    #
    # set timeseries extension header key comments. also set the units.
    #
    tfields = hdr['TFIELDS'] # number of table fields
    hdrkv = {}
    for ind in range(1,tfields+1):
        key = 'TTYPE{}'.format(ind)
        hdrkv[key] = hdr[key]

    for k,v in hdrkv.items():

        hdr.comments[k] = _map_timeseries_key_to_comment(v.lower())

        hdr[k.replace('TTYPE','TUNIT')] = (
            _map_timeseries_key_to_unit( v.lower())
        )

    hdr.set(
        'MJD_BEG',
        np.nanmin(data['TMID_UTC'] - 2400000.5),
        'min(TMID_UTC) - 2400000.5'
    )
    hdr.set(
        'MJD_END',
        np.nanmax(data['TMID_UTC'] - 2400000.5),
        'max(TMID_UTC) - 2400000.5'
    )
    hdr.set(
        'XPOSURE',
        1800,
        '[sec] exposure time per cadence'
    )

    #
    # write it (!)
    #
    primary_hdu = fits.PrimaryHDU(header=primaryhdr)
    timeseries_hdu = fits.BinTableHDU(header=hdr, data=data)
    timeseries_hdu.header = hdr
    outhdulist = fits.HDUList([primary_hdu, timeseries_hdu])

    outname = (
        'hlsp_cdips_tess_ffi_'
        'gaiatwo{zsourceid}-{zsector}_'
        'tess_v{zcdipsvnum}_llc.fits'
    ).format(
        zsourceid=str(lcgaiaid).zfill(22),
        zsector=str(sectornum).zfill(4),
        zcdipsvnum=str(cdipsvnum).zfill(2)
    )

    outfile = os.path.join(outdir, outname)

    outhdulist.writeto(outfile, overwrite=False)
    print('{}: reformatted {}, wrote to {}'.format(
        datetime.utcnow().isoformat(), lcpath, outfile))

    outhdulist.close()


def reformat_headers(lcpaths, outdir, sectornum, cdipsvnum):

    cdips_df = pd.read_csv(cdips_cat_file, sep=';')

    for lcpath in lcpaths:
        outpath = os.path.join(outdir, os.path.basename(lcpath))
        if not os.path.exists(outpath):
            _reformat_header(lcpath, cdips_df, outdir, sectornum, cdipsvnum)


def mask_orbit_start_and_end(lcpaths):
    pass
