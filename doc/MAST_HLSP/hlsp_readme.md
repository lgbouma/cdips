TODO: FIX ALL "CITE" WITH TRUE CITATIONS

# INTRODUCTION
The TESS mission has been releasing full-frame images recorded at 30 minute
cadence.  Using the TESS images, we have begun a Cluster Difference Imaging
Photometric Survey (CDIPS), in which we are making light curves for stars that
are candidate members of open clusters and moving groups.  We have also
included stars that show photometric indications of youth.  Each light curve
represents between 20 and 25 days of observations of a star brighter than Gaia
Rp magnitude of 16.  The precision of the detrended light curves is generally
in line with theoretical expectations.

The project reference is Bouma et al. (2019) (CITE).
The pipeline is called `cdips-pipeline`; it is
[available for inspection](https://github.com/waqasbhatti/cdips-pipeline) and
citation as an independent software reference (Bhatti et al., 2019).

# DR1 CONTENTS
The first CDIPS data release contains 159,343 light curves of target stars that
fell on silicon during TESS Sectors 6 and 7.  They cover about one sixth of the
galactic plane.  The target stars are described and listed in Bouma et al.,
(2019) (CITE). They are stars for which a mix of Gaia and pre-Gaia kinematic,
astrometric, and photometric information suggest either cluster membership or
youth.

Before using the light curves, we strongly recommend that you become familiar
with the TESS data release notes (CITE), and also consult the TESS Instrument
Handbook (CITE).

# DATA PRODUCTS
The light curves are in a `.fits` format familiar to users of the Kepler, K2,
and TESS-short cadence light curves made by the NASA Ames team.

The primary header contains information about the target star, including the
catalogs that claimed cluster membership or youth (`CDIPSREF`), and a key that
enables back-referencing to those catalogs in order to discover whatever those
investigators said about the object (`CDEXTCAT`).  Membership claims based on
Gaia-DR2 data are typically the highest quality claims. Cross-matches against
TICv8 and Gaia-DR2 are also included.

The sole binary table extension contains the light curves.  Three aperture
sizes are used:
```
APERTURE1: 1 pixel in radius
APERTURE2: 1.5 pixels in radius
APERTURE3: 2.25 pixels in radius
```

Three different types of light curves are available.  The first is the raw
``instrumental'' light curve measured from differenced images.  The second is a
detrended light curve that regresses against the number of principal components
noted in the light curve's header.  The third is a detrended light curve found
by applying TFA with a fixed number of template stars.  The recommended time
stamp is `TMID_BJD`, which is the exposure mid-time at the barycenter of the
solar system (BJD), in the Temps Dynamique Barycentrique standard (TDB).  For
further details, please see Bouma et al., (2019), or send emails to the
authors.

The full set of available time-series vectors is as follows.
```
TTYPE1  = 'BGE     '           / Background measurement error
TTYPE2  = 'BGV     '           / Background value (after bkgd surface subtrxn)
TTYPE3  = 'FDV     '           / Measured D value (see Pal 2009 eq 31)
TTYPE4  = 'FKV     '           / Measured K value (see Pal 2009 eq 31)
TTYPE5  = 'FSV     '           / Measured S value (see Pal 2009 eq 31)
TTYPE6  = 'IFE1    '           / Flux error in aperture 1 (ADU)
TTYPE7  = 'IFE2    '           / Flux error in aperture 2 (ADU)
TTYPE8  = 'IFE3    '           / Flux error in aperture 3 (ADU)
TTYPE9  = 'IFL1    '           / Flux in aperture 1 (ADU)
TTYPE10 = 'IFL2    '           / Flux in aperture 2 (ADU)
TTYPE11 = 'IFL3    '           / Flux in aperture 3 (ADU)
TTYPE12 = 'IRE1    '           / Instrumental mag error for aperture 1
TTYPE13 = 'IRE2    '           / Instrumental mag error for aperture 2
TTYPE14 = 'IRE3    '           / Instrumental mag error for aperture 3
TTYPE15 = 'IRM1    '           / Instrumental mag in aperture 1
TTYPE16 = 'IRM2    '           / Instrumental mag in aperture 2
TTYPE17 = 'IRM3    '           / Instrumental mag in aperture 3
TTYPE18 = 'IRQ1    '           / Instrumental quality flag ap 1, 0/G OK, X bad
TTYPE19 = 'IRQ2    '           / Instrumental quality flag ap 2, 0/G OK, X bad
TTYPE20 = 'IRQ3    '           / Instrumental quality flag ap 3, 0/G OK, X bad
TTYPE21 = 'RSTFC   '           / Unique frame key
TTYPE22 = 'TMID_UTC'           / Exp mid-time in JD_UTC (from DATE-OBS,DATE-END)
TTYPE23 = 'XIC     '           / Shifted X coordinate on CCD on subtracted frame
TTYPE24 = 'YIC     '           / Shifted Y coordinate on CCD on subtracted frame
TTYPE25 = 'CCDTEMP '           / Mean CCD temperature S_CAM_ALCU_sensor_CCD
TTYPE26 = 'NTEMPS  '           / Number of temperatures avgd to get ccdtemp
TTYPE27 = 'TMID_BJD'           / Exp mid-time in BJD_TDB (BJDCORR applied)
TTYPE28 = 'BJDCORR '           / BJD_TDB = JD_UTC + TDBCOR + BJDCORR
TTYPE29 = 'TFA1    '           / TFA Trend-filtered magnitude in aperture 1
TTYPE30 = 'TFA2    '           / TFA Trend-filtered magnitude in aperture 2
TTYPE31 = 'TFA3    '           / TFA Trend-filtered magnitude in aperture 3
TTYPE32 = 'PCA1    '           / PCA Trend-filtered magnitude in aperture 1
TTYPE33 = 'PCA2    '           / PCA Trend-filtered magnitude in aperture 2
TTYPE34 = 'PCA3    '           / PCA Trend-filtered magnitude in aperture 3
```
