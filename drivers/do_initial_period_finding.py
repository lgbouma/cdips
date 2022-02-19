"""
Given >~10k LCs per galactic field, do the period finding (and detrend, since
the LCs should be variable).

Contents:

do_initial_period_finding: runs period-finding, and sets the detection limits
adapted based on the results.

periodfindingworker: given lcpath, run periodograms and detrend

get_tls_sde_versus_period_detection_boundary: given TLS SDE and TLS periods for
"plausible planet detections", define the detection boundary.

select_periodfinding_results_given_searchtype: defines the *stellar and TLS
selection functions* for the CDIPS search that gets uploaded to ExoFOP.

----------
run as (from phtess2 usually):
$ python -u do_initial_period_finding.py &> logs/sector6_initial_period_finding.log &
"""

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

from astropy.timeseries import LombScargle
from transitleastsquares import transitleastsquares

from astrobase import lcmath
from scipy.interpolate import interp1d

import multiprocessing as mp

import numpy as np, pandas as pd
import os, textwrap
from glob import glob
from datetime import datetime
from astropy.io import fits
from copy import deepcopy

from cdips.utils import collect_cdips_lightcurves as ccl
from cdips.utils import lcutils as lcu
from cdips.lcproc import mask_orbit_edges as moe
from cdips.lcproc import detrend as dtr
from cdips.lcproc.find_planets import run_periodograms_and_detrend
from cdips.testing import check_dependencies
from skim_cream import plot_initial_period_finding_results

from astrobase.lcmath import sigclip_magseries

from wotan import slide_clip

nworkers = mp.cpu_count()

def main():

    check_dependencies()

    for s in range(26, 27):
        do_initial_period_finding(
            sectornum=s, nworkers=nworkers, maxworkertasks=1000,
            outdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding',
            OC_MG_CAT_ver=0.6
        )


def periodfindingworker(task):

    lcpath, outcsvpath = task

    if os.path.exists(outcsvpath):
        LOGINFO(f'Found {outcsvpath}. Skipping.')
        return

    # NOTE: default search aperture: PCA1
    APNAME = 'PCA1'
    source_id, time, mag, xcc, ycc, ra, dec, _, _ = (
        lcu.get_lc_data(lcpath, mag_aperture=APNAME, tfa_aperture='TFA1')
    )

    dtr_method, break_tolerance, window_length = 'best', 0.5, 0.5
    dtr_dict = {'method':dtr_method,
                'break_tolerance':break_tolerance,
                'window_length':window_length}

    DEFAULTDICT = {
        'source_id': source_id, 'lcpath': lcpath, 'tls_status': 'FAILED',
        'ls_period': np.nan, 'ls_fap': np.nan, 'ls_amplitude': np.nan,
        'tls_period': np.nan, 'tls_sde': np.nan, 'tls_snr': np.nan,
        'tls_t0': np.nan, 'tls_depth': np.nan, 'tls_duration': np.nan,
        'tls_distinct_transit_count': np.nan, 'tls_odd_even': np.nan,
        'dtr_method': dtr_method, 'xcc': xcc, 'ycc': ycc, 'ra': ra,
        'dec': dec
    }

    if np.all(pd.isnull(mag)):
        r = DEFAULTDICT
        r['tls_status'] = 'FAILED_ALLNANLC'

    else:
        # Detrend and run the LS & TLS periodograms.
        try:
            r = run_periodograms_and_detrend(
                source_id, time, mag, dtr_dict
            )
            r['xcc'] = xcc
            r['ycc'] = ycc
            r['ra'] = ra
            r['dec'] = dec
            r['tls_status'] = 'PASSED'
            r['lcpath'] = lcpath

        except Exception as e:
            msg = (
                f'run_periodograms_and_detrend failed for GAIA DR2 '
                f'{source_id}.\n Error was "{e}"'
            )
            LOGERROR(msg)
            r = DEFAULTDICT
            r['tls_status'] = 'FAILED_SEARCHERROR'

    # Write the results to CSV.
    outd = r
    outdf = pd.DataFrame(outd, index=[0])

    outdf.to_csv(outcsvpath, index=False)
    LOGINFO(f'Wrote {outcsvpath}.')

    return


def do_initial_period_finding(
    sectornum=None,
    nworkers=None,
    maxworkertasks=1000,
    outdir='/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding',
    OC_MG_CAT_ver=None
    ):

    check_dependencies()

    lcdirectory = (
        f'/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{sectornum}/'
    )
    lcglob = 'cam?_ccd?/*_llc.fits'

    np.random.seed(42)
    lcpaths = glob(os.path.join(lcdirectory, lcglob))
    np.random.shuffle(lcpaths)

    if DEBUG:
        # when debugging, period find on fewer LCs
        lcpaths = np.random.choice(lcpaths,size=100)

    outdir = os.path.join(outdir, f'sector-{sectornum}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outpath = os.path.join(outdir, 'initial_period_finding_results.csv')

    if not os.path.exists(outpath):

        workerdir = os.path.join(outdir, 'worker_output')
        if not os.path.exists(workerdir):
            os.mkdir(workerdir)

        tasks = [
            (x,
             os.path.join(workerdir, os.path.basename(x).
                          replace('.fits','_periodfindingresults.csv'))
            )
            for x in lcpaths
        ]

        N_lcs = len(lcpaths)

        LOGINFO(f'{len(lcpaths)} files to run initial periodograms on')

        pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
        _ = pool.map(periodfindingworker, tasks)
        pool.close()
        pool.join()

        # Merge results from worker CSVs.
        csvpaths = glob(os.path.join(workerdir, '*_periodfindingresults.csv'))
        df = pd.concat((pd.read_csv(f) for f in csvpaths))
        df = df.sort_values(by='source_id')
        df.to_csv(outpath, index=False)
        LOGINFO(f'made {outpath}')

    else:
        LOGINFO('found periodfinding results, loading them')
        df = pd.read_csv(outpath)

    # you want some idea of what references, and what clusters are most
    # important.
    outpath = os.path.join(
        outdir, 'initial_period_finding_results_supplemented.csv'
    )
    if not os.path.exists(outpath):
        cd = ccl.get_cdips_catalog(ver=OC_MG_CAT_ver)

        #
        # add Gaia plx, ra, dec, pmra, pmdec, G, BP, RP, cluster, age,
        # mean_age, reference_id, and bibcode.
        #
        df['source_id'] = df['source_id'].astype(np.int64)
        mdf = df.merge(cd, how='left', on='source_id')

        #
        # Supplement with initial Prot/color age classification.
        # sub-NGC6811 (2; <~1 Gyr), sub-Praesepe (1; <~700 Myr); and
        # sub-Pleiades (0; <~120 Myr).  Anything else is given a 3.
        # Note that J.Curtis+2019's NGC6811 stalling paper only went down to
        # ~M0V at 1 Gyr; the "sub-Gyr" requirement for redder stars therefore
        # matches "sub-Praesepe".
        # Note also that this initial classification ignores reddening.  So,
        # stars at larger distances (e.g., 500-1000pc) will appear redder, and
        # may shift a little at the class boundaries.  The latter two flaws
        # with this approach seem small enough relative to the benefits it
        # provides (a quick-look age check) that we keep it.  It also prevents
        # introducing wrong extinction corrections.  (Though those could be
        # implemented using say the all-sky Lallament+ STILISM maps).
        #

        bpmrp = mdf.phot_bp_mean_mag - mdf.phot_rp_mean_mag

        # default is 3
        prot_color_class = 3*np.ones(len(mdf))

        # those that meet the <1 Gyr requirement are "class 2"
        from cdips.gyroage import NGC6811InterpModel
        sel = (mdf.ls_period < NGC6811InterpModel(bpmrp, bounds_error=False))
        prot_color_class[sel] = 2

        # those that meet the <700 Myr requirement are "class 1"
        from cdips.gyroage import PraesepeInterpModel
        sel = (mdf.ls_period < PraesepeInterpModel(bpmrp, bounds_error=False))
        prot_color_class[sel] = 1

        # those that meet the <120 Myr requirement are "class 0"
        from cdips.gyroage import PleiadesInterpModel
        sel = (mdf.ls_period < PleiadesInterpModel(bpmrp, bounds_error=False))
        prot_color_class[sel] = 0

        mdf['prot_color_class'] = prot_color_class

        # ";" sep needed b/c reference is ","-separated
        mdf.to_csv(outpath, index=False, sep=';')
        LOGINFO(f'made {outpath}')

    else:

        LOGINFO('found supplemented periodfinding results, loading them')
        mdf = pd.read_csv(outpath, sep=';')

    if 'reference' in mdf:
        u_ref, u_ref_count = np.unique(mdf['reference'], return_counts=True)
    elif 'reference_id' in mdf:
        u_ref, u_ref_count = np.unique(mdf['reference_id'], return_counts=True)
    else:
        u_ref, u_ref_count = ('N/A', 0)
    u_cluster, u_cluster_count = np.unique(
        np.array(mdf['cluster']).astype(str), return_counts=True)

    outpath = os.path.join(outdir, 'which_references_and_clusters_matter.txt')

    if not os.path.exists(outpath):
        with open(outpath, mode='w') as f:
            txt = (
            """
            ==========================================
            of {ntotal} total CDIPS lightcurves made (including nans)

            top 20 referred sources:
            {top20ref}

            top 20 referred sources counts:
            {top20refcounts}

            top 20 referred sources counts fraction:
            {top20refcountsfrac}

            ==========

            top 20 clusters quoted:
            {top20cluster}

            top 20 clusters counts:
            {top20clustercounts}

            top 20 clusters counts fraction:
            {top20clustercountsfrac}
            """
            ).format(
                ntotal=len(mdf),
                top20ref=repr(
                    u_ref[np.argsort(u_ref_count)[::-1]][:20]
                ),
                top20refcounts=repr(
                    u_ref_count[np.argsort(u_ref_count)[::-1]][:20]
                ),
                top20refcountsfrac=repr(
                    u_ref_count[np.argsort(u_ref_count)[::-1]][:20]/len(mdf)
                ),
                top20cluster=repr(
                    u_cluster[np.argsort(u_cluster_count)[::-1]][:20]
                ),
                top20clustercounts=repr(
                    u_cluster_count[np.argsort(u_cluster_count)[::-1]][:20]
                ),
                top20clustercountsfrac=repr(
                    u_cluster_count[np.argsort(u_cluster_count)[::-1]][:20]/len(mdf)
                )
            )
            f.write(textwrap.dedent(txt))
        LOGINFO(f'made {outpath}')
    else:
        pass

    # plot results distribution
    resultsdir = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/'
        'cdips_lc_periodfinding/'
        f'sector-{sectornum}'
    )
    initpfresultspath = (
        os.path.join(resultsdir, 'initial_period_finding_results.csv')
    )
    df = pd.read_csv(initpfresultspath)

    msg = (
        f'Finished do_initial_periodfinding for sector {sectornum}.'
        f'\nYou might wish to verify that the detection boundary looks OK.'
    )
    LOGINFO(msg)

    SEARCHTYPE = 'SINGLESECTOR_TRANSITING_PLANETS_AROUND_SUBGYR_STARS'
    LOGINFO(42*'-')
    LOGINFO(f'Applying star+planet+age selection function for {SEARCHTYPE}')
    LOGINFO(f'For this sector/cam/ccd, started with {len(mdf)} LCs.')
    smdf = select_periodfinding_results_given_searchtype(
        SEARCHTYPE, mdf
    )
    LOGINFO(f'After selection, {len(smdf)} LCs remain '
            f'({100*len(smdf)/len(mdf):.2f}%).')
    LOGINFO(f'Proceeding to generate SDE limit.')

    limit, abovelimit = get_tls_sde_versus_period_detection_boundary(
        smdf.tls_sde, smdf.tls_period
    )
    smdf['limit'] = limit
    smdf['abovelimit'] = abovelimit

    outpath = os.path.join(
        resultsdir, 'initial_period_finding_results_with_limit.csv'
    )
    smdf.to_csv(outpath, index=False)
    LOGINFO(f'made {outpath}')

    plot_initial_period_finding_results(smdf, resultsdir)

    return


def select_periodfinding_results_given_searchtype(SEARCHTYPE, df):
    """
    This function applies cuts on the raw period-finding results to define the
    "base sample" of signals.  Note this doesn't involve a cut on SDE, but it
    does involve cuts on some other TLS parameters (e.g., transit count, and
    odd-even). It can also involve cuts on stellar parameters.

    The SEARCHTYPE argument currently only has one implemented type of search:
        SINGLESECTOR_TRANSITING_PLANETS_AROUND_SUBGYR_STARS.

    In this case, four cuts are applied:

        apply_singlesector_cuts:
            * TLS period < 21 days.

        apply_planet_cuts
            * TLS gave finite ephemeris, SDE, duration, and transit count.
            * >=3 transits
            * TLS transit depth < 20%

        apply_star_cuts:
            * parallax / parallax_error > 3
            * parallax > 1 mas (i.e., distance < 1 kpc)

            (note: these are just to make stars actually possible for
            follow-up. color requirements are needed for age cuts.)

        apply_subgyr_cuts:

            * LS gave finite period and amplitude.

            * Stellar color (Bp-Rp) implies Teff <~ 6600 K. Bp-Rp > 0.5
            (>F2-F3V), according to Mamajek's 2021.03.02.  Note this is *observed*
            (i.e., without reddening correction). So, some hotter stars will
            make it in near the boundary.

            * LS period and color imply < 1 Gyr, assuming that LS period is the
            rotation period.  (See description of prot_color_class in comments
            above).

            * No cut on amplitude is applied due to the low-amplitude tail for
            the hot stars (e.g., Rebull+2020, Figure 10).
    """

    # NOTE: you may wish to change these, depending on your search
    # parameters.
    apply_planet_cuts = 1
    apply_star_cuts = 1
    apply_subgyr_cuts = 1
    apply_singlesector_cuts = 1 if 'SINGLESECTOR' in SEARCHTYPE else 0

    if SEARCHTYPE not in ["SINGLESECTOR_TRANSITING_PLANETS_AROUND_SUBGYR_STARS"]:
        raise NotImplementedError

    sel = ~pd.isnull(df.source_id)

    if apply_singlesector_cuts:
        #
        # Require planet orbital period < 21 days for single sector. (The max
        # period in this TLS search is 27 days, but there's a big systematic
        # pileup).
        #
        sel &= (df.tls_period < 21)

    if apply_planet_cuts:

        #
        # TLS gave finite ephemeris, SDE, duration, and transit count.
        #
        sel &= (~pd.isnull(df.tls_t0))
        sel &= (~pd.isnull(df.tls_period))
        sel &= (~pd.isnull(df.tls_sde))
        sel &= (~pd.isnull(df.tls_duration))
        sel &= (~pd.isnull(df.tls_distinct_transit_count))
        sel &= (np.isfinite(df.tls_odd_even))

        #
        # TLS requirements.
        # 1. At least three transits.
        # 2. Depth < 20%, i.e., excluding mega-obvious EBs.  Odd-even
        # requirements are done after vetting reports (since three-transit
        # cases can fail to converge).
        #

        sel &= (df.tls_distinct_transit_count >= 3)
        sel &= (df.tls_depth >= 0.8)

    if apply_star_cuts:

        #
        # Parallax cuts: to be able to follow it up, the star pretty much
        # always has to be within ~1 kpc (i.e., parallax < 1 mas).  Also, if
        # the Gaia parallax solution has S/N < 3, something is probably wrong.  
        #
        sel &= (df.parallax/df.parallax_error > 3)
        sel &= (df.parallax > 1)


    if apply_subgyr_cuts:

        #
        # LS gave finite period, and amplitude.
        #
        sel &= (~pd.isnull(df.ls_period))
        sel &= (~pd.isnull(df.ls_amplitude))

        if apply_singlesector_cuts:

            # NB. some M dwarfs at >=Praesepe age do have rotation periods
            # longer than 15 days.  However we won't reliably be able to
            # measure their rotation periods.

            sel &= df.ls_period < 15

        #
        # Bp-Rp > 0.5. Rotation periods aren't measurable at hotter
        # temperatures.
        #
        bpmrp = df.phot_bp_mean_mag - df.phot_rp_mean_mag
        sel &= (bpmrp > 0.5)

        #
        # Rotation and color imply age below 1 Gyr.
        #

        sel &= (df.prot_color_class <= 2)

    return df[sel]


def get_tls_sde_versus_period_detection_boundary(tls_sde, tls_period,
                                                 make_plots=False):
    """
    Given np.ndarrays of tls_sde and tls_period, return an array "limit", which
    represents the boundary in SDE versus Period space between "above
    threshold" and "below threshold" planets.  Also, return the array
    "abovelimit", which is just the boolean array `tls_sde > limit`.

    The construction of the boundary is performed by sorting on tls_period
    """

    assert np.all(np.isfinite(tls_period))
    assert np.all(np.isfinite(tls_sde))
    assert len(tls_sde) == len(tls_period)

    N_lcs = len(tls_period)

    WRN_THRESHOLD = 5e3

    if N_lcs < WRN_THRESHOLD:
        msg = (
            f'Only got {N_lcs} light curves for definition of TLS versus '
            'SDE boundary. Will likely often default to baseline window.'
        )
        LOGWARNING(msg)

    # the cleanest window for single-sector TESS data is typically
    # from 2 to 10 days. use that to set the "bare minimum" threshold, as the
    # 95th percentile of the TLS SDE..
    base_window = (tls_period > 2) & (tls_period < 10)
    sde_95 = np.percentile(tls_sde[base_window], 95)

    # ~1e4 LCs total typical per sector (after applying viable star cuts).
    # --> ~200 stars per bin.
    if N_lcs < 2e4:
        denom = 50
    else:
        denom = 100
    N_bins = int(N_lcs / denom)

    hist, bin_edges = np.histogram(np.log10(tls_period), N_bins)

    sde_boundary = []
    midpoints = []
    N_cutoff = 50

    hist_50 = np.percentile(hist, 50)
    hist_75 = np.percentile(hist, 75)
    hist_90 = np.percentile(hist, 90)
    hist_95 = np.percentile(hist, 95)
    hist_98 = np.percentile(hist, 98)

    for lo,hi,count in zip(bin_edges[0:-1], bin_edges[1:], hist):

        midpoint = (lo+hi)/2

        _sel = (
            (np.log10(tls_period) >= lo)
            &
            (np.log10(tls_period) < hi)
        )

        these_sdes = tls_sde[_sel]

        # you want a higher cutoff in regions with more points.
        if count > hist_98:
            this_sde = np.percentile(these_sdes, 99.8)
        elif count > hist_95:
            this_sde = np.percentile(these_sdes, 99.5)
        elif count > hist_90:
            this_sde = np.percentile(these_sdes, 98)
        elif count > hist_75 and N_lcs > int(2e4):
            this_sde = np.percentile(these_sdes, 97)
        elif count > hist_50 and N_lcs > int(2e4):
            this_sde = np.percentile(these_sdes, 96)
        else:
            this_sde = sde_95

        midpoints.append(midpoint)
        if len(these_sdes) > N_cutoff:
            sde_boundary.append(
                np.max([
                    sde_95, this_sde
                ])
            )
        else:
            sde_boundary.append(
                sde_95
            )
    sde_boundary = np.array(sde_boundary)
    period_bin_edges = 10**bin_edges

    fn_log10period_to_limit = interp1d(
        midpoints, sde_boundary, kind='linear', bounds_error=False,
        fill_value='extrapolate'
    )

    interp_log10period = np.linspace(
        min(np.log10(tls_period)), max(np.log10(tls_period)), 1000
    )
    interp_limit = fn_log10period_to_limit(interp_log10period)

    limit = fn_log10period_to_limit(np.log10(tls_period))
    abovelimit = np.array(tls_sde > limit).astype(bool)

    if make_plots:
        # assessment plots

        # 1d period histogram
        plt.hist(np.log10(tls_period), bins=N_bins)
        plt.xlabel('log10 tls period [d]')
        plt.ylabel('count')
        plt.savefig('temp_tlsperiod_log10hist.png', dpi=400)
        plt.close('all')

        # 2d period vs SDE histogram
        from fast_histogram import histogram1d, histogram2d
        from astropy.visualization.mpl_normalize import ImageNormalize
        from astropy.visualization import LogStretch
        norm = ImageNormalize(vmin=0.1, vmax=50, stretch=LogStretch())
        fig = plt.figure()
        fig.set_dpi(600)
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        ax.scatter_density(np.log10(tls_period), tls_sde, cmap='Greys', norm=norm)
        plt.xlabel('log10 tls period [d]')
        plt.ylabel('SDE')
        fig.savefig('temp_log10period_log10sde_scatterdensity.png', dpi=600)
        plt.close('all')

        # scatter period vs SDE with boundary
        #plt.plot(bin_edges, sde_boundary, c='orange', lw=0.5, zorder=3)
        title = f'N={len(tls_sde)}. N_above={len(tls_sde[abovelimit])}'
        plt.plot(interp_log10period, interp_limit, c='orange', lw=0.5, zorder=3)
        plt.scatter(np.log10(tls_period), tls_sde,
                    s=0.5, c='k', zorder=2, marker='.', linewidths=0)
        plt.scatter(np.log10(tls_period[abovelimit]), tls_sde[abovelimit],
                    s=4, c='C0', zorder=3, marker='.', linewidths=0)
        plt.xlabel('log10 tls period [d]')
        plt.ylabel('SDE')
        plt.title(title)
        plt.savefig('temp_period_sde_scatter_boundary.png', dpi=400)
        plt.close('all')

        # hist of above limit
        plt.hist(np.log10(tls_period[abovelimit]), bins=50)
        plt.xlabel('log10 tls period [d]')
        plt.ylabel('count')
        plt.savefig('temp_tlsperiod_log10hist_abovelimit.png', dpi=400)
        plt.close('all')

    return limit, abovelimit.astype(int)


if __name__ == "__main__":
    main()
