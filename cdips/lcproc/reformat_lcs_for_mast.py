"""
Given directories of symlinked PIPE-TREX-output FTIS LCs, make the headers
presentable, and apply minor changes to improve their BLS-searchability.
"""

import pandas as pd, numpy as np
import os, requests, time
from astropy.io import fits
from datetime import datetime
from astroquery.mast import Catalogs
from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord
import multiprocessing as mp

from cdips.utils import collect_cdips_lightcurves as ccl

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
                   str(info['cluster'].iloc[0]),
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
                   str(lcgaiaid),
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
    # TIC xmatch info:
    # for TICv7,
    # require Tmag within 1 mag of Stassun 2019 prediction.
    # then take the closest such star to Gaia position.
    #
    ra, dec = primaryhdr['RA_OBJ'], primaryhdr['DEC_OBJ']
    targetcoord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    radius = 1.0*u.arcminute

    try:
        stars = Catalogs.query_region(
            "{} {}".format(float(targetcoord.ra.value), float(targetcoord.dec.value)),
            catalog="TIC",
            radius=radius
        )
    except requests.exceptions.ConnectionError:
        print('ERR! TIC query failed. trying again...')
        time.sleep(60)
        stars = Catalogs.query_region(
            "{} {}".format(float(targetcoord.ra.value), float(targetcoord.dec.value)),
            catalog="TIC",
            radius=radius
        )

    Tmag_pred = (primaryhdr['phot_g_mean_mag']
                - 0.00522555 * (primaryhdr['phot_bp_mean_mag'] - primaryhdr['phot_rp_mean_mag'])**3
                + 0.0891337 * (primaryhdr['phot_bp_mean_mag'] - primaryhdr['phot_rp_mean_mag'])**2
                - 0.633923 * (primaryhdr['phot_bp_mean_mag'] - primaryhdr['phot_rp_mean_mag'])
                + 0.0324473)
    Tmag_cutoff = 1

    selstars = stars[np.abs(stars['Tmag'] - Tmag_pred)<Tmag_cutoff]

    if len(selstars)>=1:
        mrow = selstars[np.argmin(selstars['dstArcSec'])]

        # TICv8 rebasing on GaiaDR2 allows this
        if not int(mrow['GAIA']) == int(hdr['GAIA-ID']):
            # TODO: should just instead query selstars by Gaia ID u want...
            raise ValueError

        primaryhdr.set('TICVER',
                       mrow['version'],
                       'TIC version')

        primaryhdr.set('TICID',
                       str(mrow['ID']),
                       'TIC identifier of xmatch')

        primaryhdr.set('TESSMAG',
                       mrow['Tmag'],
                       '[mag] TIC catalog magnitude of xmatch')

        primaryhdr.set('TMAGPRED',
                       Tmag_pred,
                       '[mag] predicted Tmag via Stassun+19 Eq1')
        try:
            primaryhdr.set('TICCONT',
                           mrow['contratio'],
                           'TIC contratio of xmatch ')
        except ValueError:
            if isinstance(mrow['contratio'], np.ma.core.MaskedConstant):
                primaryhdr.set('TICCONT',
                               'nan',
                               'TIC contratio of xmatch ')
            else:
                raise ValueError

        primaryhdr.set('TICDIST',
                       mrow['dstArcSec'],
                       '[arcsec] xmatch dist btwn Gaia & TIC')
    else:
        primaryhdr.set('TICVER',
                       'nan',
                       'TIC version')
        primaryhdr.set('TICID',
                       'nan',
                       'TIC identifier of xmatch')
        primaryhdr.set('TESSMAG',
                       'nan',
                       '[mag] TIC catalog magnitude of xmatch')
        primaryhdr.set('TMAGPRED',
                       Tmag_pred,
                       '[mag] predicted Tmag via Stassun+19 Eq1')
        primaryhdr.set('TICCONT',
                       'nan',
                       'TIC contratio of xmatch ')
        primaryhdr.set('TICDIST',
                       'nan',
                       '[arcsec] xmatch dist btwn Gaia & TIC')

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


def reformat_worker(task):

    lcpath, cdips_df, outdir, sectornum, cdipsvnum = task

    lcgaiaid = os.path.basename(lcpath).split('_')[0]

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

    if not os.path.exists(outfile):
        _reformat_header(lcpath, cdips_df, outdir, sectornum, cdipsvnum)
        return 1
    else:
        return 0


def parallel_reformat_headers(lcpaths, outdir, sectornum, cdipsvnum,
                              nworkers=56, maxworkertasks=1000):

    cdips_df = ccl.get_cdips_catalog(ver=0.3)

    tasks = [(x, cdips_df, outdir, sectornum, cdipsvnum) for x in lcpaths[:300]]

    print('%sZ: %s files to reformat' %
          (datetime.utcnow().isoformat(), len(lcpaths)))

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)

    # fire up the pool of workers
    results = pool.map(reformat_worker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    print('%sZ: finished reformatting' (datetime.utcnow().isoformat()))

    return {result for result in results}



def reformat_headers(lcpaths, outdir, sectornum, cdipsvnum):

    cdips_df = ccl.get_cdips_catalog(ver=0.3)

    for lcpath in lcpaths:

        lcgaiaid = os.path.basename(lcpath).split('_')[0]

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

        if not os.path.exists(outfile):
            _reformat_header(lcpath, cdips_df, outdir, sectornum, cdipsvnum)
        else:
            print('found {}'.format(outfile))


def mask_orbit_start_and_end(lcpaths):
    pass
