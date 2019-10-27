"""
Given directories of symlinked PIPE-TREX-output FTIS LCs, make the headers
presentable; apply minor changes to improve their BLS-searchability; compute &
append PCA-detrended light curves.

reformat_headers
reformat_worker
parallel_reformat_headers
"""

import pandas as pd, numpy as np
import os, requests, time
from astropy.io import fits
from datetime import datetime
from astroquery.mast import Catalogs
from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord

import multiprocessing as mp
from copy import deepcopy

from sklearn.linear_model import LinearRegression

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.tests import verify_lightcurves as test

from astrobase.lcmath import find_lc_timegroups

def _get_tic(mrow, key):
    if isinstance(mrow[key], np.ma.core.MaskedConstant):
        return 'nan'
    else:
        return mrow[key]

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
        'tfa1'  : "TFA Trend-filtered magnitude in aperture 1",
        'tfa2'  : "TFA Trend-filtered magnitude in aperture 2",
        'tfa3'  : "TFA Trend-filtered magnitude in aperture 3",
        'pca1'  : "PCA Trend-filtered magnitude in aperture 1",
        'pca2'  : "PCA Trend-filtered magnitude in aperture 2",
        'pca3'  : "PCA Trend-filtered magnitude in aperture 3",
        "ccdtemp" : "Mean CCD temperature S_CAM_ALCU_sensor_CCD",
        "ntemps"  : "Number of temperatures avgd to get ccdtemp",
        'dtr_isub': "Img subtraction photometry performed",
        'dtr_epd' : "EPD detrending performed",
        'dtr_tfa' : "TFA detrending performed",
        'dtr_pca' : "PCA detrending performed",
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
        'pca1'  : "mag",
        'pca2'  : "mag",
        'pca3'  : "mag",
        "ccdtemp" : "degcelcius",
        "ntemps"  : "unitless",
        'dtr_isub': "bool",
        'dtr_epd' : "bool",
        'dtr_tfa' : "bool",
        'dtr_pca' : "bool",
        'projid' :  "unitless",
        'btc_ra' : "deg",
        'btc_dec' : "deg",
        'rdistpx': "pixels",
        'thetadeg': "degrees"
    }
    return kcd[k]


def _reformat_header(lcpath, cdips_df, outdir, sectornum, cdipsvnum,
                     eigveclist=None, n_comp_df=None):
    # eigveclist: length 3, each with a np.ndarray of eigenvectors given by
    # dtr.prepare_pca

    hdul = fits.open(lcpath)
    primaryhdr, hdr, data = (
        hdul[0].header, hdul[1].header, hdul[1].data
    )
    hdul.close()

    ##########################################
    # begin PCA.
    #
    # first, get ensemble pca magnitude vectors. these will be appended.
    # (in an annoying order. so be it).
    #
    # eigenvecs shape: N_templates x N_times
    #
    # model: 
    # y(w, x) = w_0 + w_1 x_1 + ... + w_p x_p.
    #
    # X is matrix of (n_samples, n_features), so each template is a "sample",
    # and each time is a "feature". Analogous to e.g., doing PCA to reconstruct
    # a spectrum, and you want each wavelength bin to be a feature.
    #
    #
    primaryhdr['DTR_PCA'] = False

    pca_mags = {}

    for ix, eigenvecs in enumerate(eigveclist):

        if np.any(pd.isnull(eigenvecs)):
            print('got nans in eigvecs. bad!')
            import IPython; IPython.embed()

        ap = ix+1

        n_components = int(n_comp_df['fa_cv_ap{}'.format(ap)])

        reg = LinearRegression(fit_intercept=True)

        y = data['IRM{}'.format(ap)]

        if np.all(pd.isnull(y)):
            # if all nan, job is easy
            model_mag = y
            pca_mags['PCA{}'.format(ap)] = model_mag
            primaryhdr['PCA{}NCMP'.format(ap)] = (
                'nan',
                'N principal components PCA{}'.format(ap)
            )

        elif np.any(pd.isnull(y)):
            #
            # if some nan in target light curve, tricky. strategy: impose the
            # same nans onto the eigenvectors. then fit without nans. then put
            # nans back in.
            #
            mean_mag = np.nanmean(y)

            _X = eigenvecs[:n_components, :]

            _X = _X[:, ~pd.isnull(y)]

            reg.fit(_X.T, y[~pd.isnull(y)] - mean_mag)

            model_mag = reg.intercept_ + (reg.coef_ @ _X)

            model_mag += mean_mag

            #
            # now insert nans where they were in original target light curve.
            # typically nans occur in groups. insert by groups. this is a
            # tricky procedure, so test after to ensure nan indice in the model
            # and target light curves are the same.
            #
            naninds = np.argwhere(pd.isnull(y)).flatten()

            ngroups, groups = find_lc_timegroups(naninds, mingap=1)

            for group in groups:

                thisgroupnaninds = naninds[group]

                model_mag = np.insert(model_mag,
                                      np.min(thisgroupnaninds),
                                      [np.nan]*len(thisgroupnaninds))

            np.testing.assert_(
                model_mag.shape == y.shape
            )
            np.testing.assert_(
                (len((model_mag-y)[pd.isnull(model_mag-y)])
                 ==
                 len(y[pd.isnull(y)])
                ),
                'got different nan indices in model mag than target mag'
            )

            #
            # save result as IRM mag - PCA model mag + mean IRM mag
            #
            pca_mags['PCA{}'.format(ap)] = y - model_mag + mean_mag

            primaryhdr['PCA{}NCMP'.format(ap)] = (
                n_components,
                'N principal components PCA{}'.format(ap)
            )

        else:
            # otherwise, just directly fit the principal components
            mean_mag = np.nanmean(y)

            _X = eigenvecs[:n_components, :]

            reg.fit(_X.T, y-mean_mag)

            model_mag = reg.intercept_ + (reg.coef_ @ _X)

            model_mag += mean_mag

            #
            # save result as IRM mag - PCA model mag + mean IRM mag
            #
            pca_mags['PCA{}'.format(ap)] = y - model_mag + mean_mag

            primaryhdr['PCA{}NCMP'.format(ap)] = (
                n_components,
                'N principal components PCA{}'.format(ap)
            )

    #
    # now merge the timeseries, as from TFA merge...
    #
    pcanames = ['PCA{}'.format(ap) for ap in range(1,4)]
    pcaformats = ['D'] * len(pcanames)
    pcadatacols = [pca_mags[k] for k in pcanames]

    pcacollist = [fits.Column(name=n, format=f, array=a) for n,f,a in
                  zip(pcanames, pcaformats, pcadatacols)]

    pcahdu = fits.BinTableHDU.from_columns(pcacollist)

    inhdulist = fits.open(lcpath)
    new_columns = inhdulist[1].columns + pcahdu.columns

    #
    # update the flag for whether detrending has been performed
    #
    primaryhdr['DTR_PCA'] = True

    # end PCA.
    ##########################################
    # begin header reformatting

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
    # for TICv8, search within 1 arcminute, then require my Gaia-DR2 ID be
    # equal to the TICv8 gaia ID.
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

    #
    # search for neighbors within 1 arcminute, with finite Gaia-DR2 entry values.
    #
    sel = ~stars['GAIA'].mask
    selstars = stars[sel]

    if len(selstars)>=1:

        #
        # TICv8 rebased on GaiaDR2: enforce that my Gaia-DR2 to TICv8 xmatch is
        # the same as what TICv8 says it should be.
        #
        if np.any(
            np.in1d(np.array(selstars['GAIA']).astype(int),
                    np.array(int(primaryhdr['GAIA-ID'])))
        ):

            ind = (
                int(np.where(np.in1d(np.array(selstars['GAIA']).astype(int),
                                 np.array(int(primaryhdr['GAIA-ID']))))[0])
            )

            mrow = selstars[ind]

        else:

            errmsg = 'FAILED TO GET TIC MATCH. CRITICAL ERROR. PLZ SOLVE.'
            import IPython; IPython.embed()
            raise NotImplementedError(errmsg)

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

        primaryhdr.set('TICCONT',
                       _get_tic(mrow, 'contratio'),
                       'TIC contratio of xmatch')

        primaryhdr.set('TICDIST',
                       mrow['dstArcSec'],
                       '[arcsec] xmatch dist btwn Gaia & TIC')

        primaryhdr.set('TICTEFF',
                       _get_tic(mrow,'Teff'),
                       '[K] TIC effective temperature')

        primaryhdr.set('TICRAD',
                       _get_tic(mrow,'rad'),
                       '[Rsun] TIC stellar radius')

        primaryhdr.set('TICRAD_E',
                       _get_tic(mrow,'e_rad'),
                       '[Rsun] TIC stellar radius uncertainty')

        primaryhdr.set('TICMASS',
                       _get_tic(mrow,'mass'),
                       '[Msun] TIC stellar mass')

        primaryhdr.set('TICLOGG',
                       _get_tic(mrow, 'logg'),
                       '[cgs] TIC log10(surface gravity)')

        primaryhdr.set('TICGDIST',
                       _get_tic(mrow, 'd'),
                       '[pc] TIC geometric distance (Bailer-Jones+2018)')

        primaryhdr.set('TICEBmV',
                       _get_tic(mrow, 'ebv'),
                       '[mag] TIC E(B-V) color excess')



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
        primaryhdr.set('TICTEFF',
                       'nan',
                       '[K] TIC effective temperature')
        primaryhdr.set('TICRAD',
                       'nan',
                       '[Rsun] TIC stellar radius')
        primaryhdr.set('TICRAD_E',
                       'nan',
                       '[Rsun] TIC stellar radius uncertainty')
        primaryhdr.set('TICMASS',
                       'nan',
                       '[Msun] TIC stellar mass')
        primaryhdr.set('TICLOGG',
                       'nan',
                       '[cgs] TIC log10(surface gravity)')
        primaryhdr.set('TICGDIST',
                       'nan',
                       '[pc] TIC geometric distance (Bailer-Jones+2018)')
        primaryhdr.set('TICEBmV',
                       'nan',
                       '[mag] TIC E(B-V) color excess')

    #
    # who dun it
    #
    primaryhdr.set('ORIGIN',
                   'Bouma&team|CDIPS|Princeton',
                   'Author|Project|Institution')

    #
    # set timeseries extension header key comments. also set the units.
    #
    timeseries_hdu = fits.BinTableHDU.from_columns(new_columns)
    hdr = timeseries_hdu.header

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

    test.verify_lightcurve(outhdulist)

    outhdulist.writeto(outfile, overwrite=False)
    print('{}: reformatted {}, wrote to {}'.format(
        datetime.utcnow().isoformat(), lcpath, outfile))

    outhdulist.close()
    inhdulist.close()


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
    # NOTE: no speed increase, b/c it's an i/o limited process.

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



def reformat_headers(lcpaths, outdir, sectornum, cdipsvnum, eigveclist=None,
                     n_comp_df=None):

    cdips_df = ccl.get_cdips_catalog(ver=0.3)

    for lcpath in lcpaths:

        lcgaiaid = os.path.basename(lcpath).split('_')[0]

        cam = os.path.dirname(lcpath).split('/')[-1].split('_')[-1].split('-')[0]
        ccd = os.path.dirname(lcpath).split('/')[-1].split('_')[-1].split('-')[1]
        projid = os.path.dirname(lcpath).split('/')[-1].split('_')[-1].split('-')[2]

        outname = (
            'hlsp_cdips_tess_ffi_'
            'gaiatwo{zsourceid}-{zsector}-cam{cam}-ccd{ccd}_'
            'tess_v{zcdipsvnum}_llc.fits'
        ).format(
            cam=cam,
            ccd=ccd,
            zsourceid=str(lcgaiaid).zfill(22),
            zsector=str(sectornum).zfill(4),
            zcdipsvnum=str(cdipsvnum).zfill(2)
        )

        outfile = os.path.join(outdir, outname)

        if not os.path.exists(outfile):
            _reformat_header(lcpath, cdips_df, outdir, sectornum, cdipsvnum,
                             eigveclist=eigveclist, n_comp_df=n_comp_df)
        else:
            print('found {}'.format(outfile))
