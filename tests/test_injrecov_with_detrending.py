"""
[environment: phtess2]
[date: June 10, 2019; refactored Oct 19 2021]
[author: Luke Bouma bouma.luke@gmail.com]

usage: $ python -u test_injrecov_with_detrending.py &> dtr.log &
       $ tail -f dtr.log | grep 'inj_'

Contents:
main
    get_inj_recov_results
        inj_recov_worker
    get_random_lc_paths
    get_cluster_lc_paths
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, textwrap, re
from glob import glob
from datetime import datetime
from astropy.io import fits
from numpy import array as nparr
from collections import Counter

import batman
import astropy.units as u, astropy.constants as c
from wotan import flatten

from astrobase.periodbase.htls import tls_parallel_pfind
from astrobase.lcmath import sigclip_magseries
from astrobase import imageutils as iu
from wotan import slide_clip

from cdips.lcproc import mask_orbit_edges as moe, detrend as dtr
from cdips.utils import lcutils as lcu
from cdips.testing import check_dependencies

import multiprocessing as mp

resultsdirectory = (
    '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/test_injrecov_with_detrending'
)
if not os.path.exists(resultsdirectory):
    os.mkdir(resultsdirectory)

def get_inj_recov_results(
    dtr_dict,
    overwrite=0, # whether to overwrite ALL previous saved csvs
    N_periods_per_lc=10, #10
    N_lcs=100, # number to draw
    P_lower=1, # lower bound of injected periods
    P_upper=10, # 10 days
    inj_depth=0.005, # 5 mmag, 0.5% central transit
    use_random_lcs=0,  # whether to draw random LCs
    use_cluster_lcs=1 # whether to use LCs from selected nearby clusters
    ):
    """
    Given a detrending method and its tuning parameters, run injection recovery
    of a b=0 planet with periods in the range [P_lower, P_upper].

    Do this on a set of CDIPS light curves.  Originally, in 2019, this was 100
    random TFA light curves.  With two years of added experience, we instead
    will choose light curves of stars in known nearby clusters.

    Args:

        dtr_dict (dict): e.g., Must have at least key "method" (could also have
        "break_tolerance", and "window_length").  `Method` can be ["pspline",
        "none", "biweight", "median", "notch", "locor", "best"].

    Returns:

        Dataframe with concatenated summary statistics from an
        injection-recovery run.
    """

    np.random.seed(42) # for random choice of LCs

    assert np.all(
        np.in1d(
            list(dtr_dict.keys()),
            ['method', 'window_length', 'break_tolerance']
        )
    )

    outpath = os.path.join(
        resultsdirectory,
        'detrend_check_method-{}_window_length-{}.csv'.
        format(dtr_dict['method'], dtr_dict['window_length'])
    )
    if os.path.exists(outpath) and not overwrite:
        print('found {} and no overwrite'.format(outpath))
        return pd.read_csv(outpath)

    if overwrite:
        csvpaths = glob(os.path.join(
            resultsdirectory, 'worker_output', dtr_dict['method'], '*.csv')
        )
        for c in csvpaths:
            os.remove(c)

    assert sum([use_random_lcs, use_cluster_lcs]) == 1
    if use_random_lcs:
        _lcpaths = get_random_lc_paths(N_lcs=N_lcs)
    elif use_cluster_lcs:
        _lcpaths = get_cluster_lc_paths(N_lcs=N_lcs)
    else:
        raise NotImplementedError(
            'use_random_lcs or use_cluster_lcs must be true'
        )

    lcpaths = np.tile(_lcpaths, N_periods_per_lc)

    inj_periods = np.random.uniform(P_lower, P_upper, N_lcs*N_periods_per_lc)
    inj_epochs = inj_periods * np.random.uniform(0, 1, N_lcs*N_periods_per_lc)
    inj_depths = np.ones_like(inj_periods)*inj_depth

    # each job is one injection / recovery experiment, and associated result.
    tasks = []
    for lcpath, inj_period, inj_epoch, inj_depth in zip(
        lcpaths, inj_periods, inj_epochs, inj_depths
    ):
        inj_dict = {
            'period':inj_period,
            'epoch':inj_epoch,
            'depth':inj_depth
        }

        tasks.append(
            (lcpath, inj_dict, dtr_dict, True)
        )

    nworkers = mp.cpu_count()
    maxworkertasks = 1000

    print(42*'-')
    print(f"{datetime.now().isoformat()}: beginning {dtr_dict['method']} ")
    print(42*'-')

    DEBUG = 0
    if DEBUG:
        inj_recov_worker(tasks[0])
        import IPython; IPython.embed()
        assert 0

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
    results = pool.map(inj_recov_worker, tasks)
    pool.close()
    pool.join()

    print(42*'-')
    print(f"{datetime.now().isoformat()}: finished {dtr_dict['method']} ")
    print(42*'-')

    # merge and save results; e.g.,
    # 3334295094569278976_period-2.58577_method-pspline_windowlength-0.3.csv
    csvpaths = glob(os.path.join(
        resultsdirectory, 'worker_output', dtr_dict['method'],
        '*_method-{}_windowlength-{}.csv'.
        format(dtr_dict['method'],dtr_dict['window_length']))
    )
    df = pd.concat((pd.read_csv(f) for f in csvpaths))
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))

    return df


def inj_recov_worker(task):

    lcpath, inj_dict, dtr_dict, plot_detrend_check = task

    # output is a csv file with (a) injected params, (b) whether injected
    # signals was recovered in first peak, (c) whether it was in first three
    # peaks.

    APNAME = 'PCA1'
    source_id, time, ap_mag, xcc, ycc, ra, dec, tmag, tfa_mag = (
        lcu.get_lc_data(lcpath, mag_aperture=APNAME, tfa_aperture='TFA1')
    )

    outdir = os.path.join(
        resultsdirectory, 'worker_output', dtr_dict['method']
    )
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outpath = os.path.join(
        outdir,
        f'tmag{tmag:.2f}_'+
        str(source_id)+
        f"_period-{inj_dict['period']:.5f}"+
        f"_method-{dtr_dict['method']}"+
        f"_windowlength-{dtr_dict['window_length']}"+
        ".csv"
    )
    if os.path.exists(outpath):
        print('found {}'.format(outpath))
        return

    if np.all(pd.isnull(ap_mag)):

        t = {
            'source_id':source_id,
            'tmag':tmag,
            'allnan':True,
            'recovered_as_best_peak':False,
            'recovered_in_topthree_peaks':False
        }
        d = {
            'bestperiod_1':np.nan,
            't0_1':np.nan,
            'duration_1':np.nan,
            'depth_1':np.nan,
            'rp_rs_1':np.nan,
            'snr_1':np.nan,
            'odd_even_mismatch_1':np.nan,
            'sde_1':np.nan,
            'bestperiod_2':np.nan,
            'bestperiod_3':np.nan,
            't0_2':np.nan,
            't0_3':np.nan
        }
        outd = {**t, **inj_dict, **dtr_dict, **d}
        outdf = pd.DataFrame(outd, index=[0])

        outdf.to_csv(outpath, index=False)
        print('made {}'.format(outpath))

        return

    #
    # otherwise, begin the injection recovery + detrending experiment.
    #
    try:

        # mags to flux
        f_x0 = 1e4
        m_x0 = 10
        ap_flux = f_x0 * 10**( -0.4 * (ap_mag - m_x0) )
        ap_flux /= np.nanmedian(ap_flux)

        # ignore the times near the edges of orbits for TLS.
        time, flux = moe.mask_orbit_start_and_end(
            time, ap_flux, raise_expectation_error=False
        )

        # inject signal.
        inj_time, inj_flux, t0 = lcu.inject_transit_signal(time, flux, inj_dict)
        inj_dict['epoch'] = t0

        search_time, search_flux, dtr_stages_dict = (
            dtr.clean_rotationsignal_tess_singlesector_light_curve(
            inj_time, inj_flux, magisflux=True, dtr_dict=dtr_dict,
            maskorbitedge=False)
        )

        if 'lsp_dict' in dtr_stages_dict:
            # copy periodogram information to saved output file
            dtr_dict['ls_period'] = dtr_stages_dict['lsp_dict']['ls_period']
            dtr_dict['ls_amplitude'] = dtr_stages_dict['lsp_dict']['ls_amplitude']
            dtr_dict['ls_fap'] = dtr_stages_dict['lsp_dict']['ls_fap']

        sel0 = dtr_stages_dict['sel0']
        clipped_inj_flux = dtr_stages_dict['clipped_flux']
        clipped_flat_flux = dtr_stages_dict['clipped_flat_flux']
        flat_flux = dtr_stages_dict['flat_flux']
        trend_flux = dtr_stages_dict['trend_flux']

        if plot_detrend_check:
            hdrlist =  (
                'CDCLSTER,CDIPSAGE,TESSMAG,phot_bp_mean_mag,phot_rp_mean_mag'.
                split(',')
            )
            infodict = iu.get_header_keyword_list(lcpath, hdrlist)
            cluster = infodict['CDCLSTER']
            age = float(infodict['CDIPSAGE'])
            tmag = float(infodict['TESSMAG'])
            bpmrp = (
                float(infodict['phot_bp_mean_mag']) -
                float(infodict['phot_rp_mean_mag'])
            )
            titlestr = (
                f"{source_id}: {cluster[:16]}, logt={age:.2f}, T={tmag:.2f}, Bp-Rp={bpmrp:.2f}"
            )
            outpng = os.path.join(
                resultsdirectory, 'worker_output', dtr_dict['method'],
                f'tmag{tmag:.2f}_'+
                str(source_id)+
                "_period-{:.5f}".format(inj_dict['period'])+
                "_method-{}".format(dtr_dict['method'])+
                '_windowlength-{}'.format(dtr_dict['window_length'])+
                ".png"
            )
            if not os.path.exists(outpng):
                f,axs = plt.subplots(nrows=2, sharex=True)
                # lower: "raw" data; upper: sigma-clipped
                axs[0].scatter(
                    inj_time, clipped_inj_flux, c='black', s=1, zorder=2,
                    rasterized=True
                )
                axs[0].scatter(
                    inj_time, inj_flux, c='red', s=1, zorder=1, rasterized=True
                )
                axs[0].plot(inj_time[sel0], trend_flux, c='C0')
                axs[1].scatter(
                    inj_time[sel0], clipped_flat_flux, c='black', s=1,
                    zorder=2, rasterized=True, label='searched flux'
                )
                axs[1].scatter(
                    inj_time[sel0], flat_flux, c='red', s=0.5, zorder=1,
                    rasterized=True
                )
                axs[1].legend(loc='best',fontsize='xx-small')
                axs[0].set_ylabel(f'flux {APNAME}')
                axs[0].set_title(titlestr, fontsize='x-small')
                axs[1].set_ylabel('flattened')
                axs[1].set_xlabel('time [bjd]')
                f.savefig(outpng, dpi=300)
                print(f"{datetime.now().isoformat()}: Made {outpng}")

        # period find
        err = np.ones_like(search_flux)*1e-4
        tlsp = tls_parallel_pfind(
            search_time, search_flux, err, magsarefluxes=True, tls_rstar_min=0.1,
            tls_rstar_max=10, tls_mstar_min=0.1, tls_mstar_max=5.0,
            tls_oversample=8, tls_mintransits=1,
            tls_transit_template='default', nbestpeaks=5, sigclip=None,
            nworkers=1
        )

        nbestperiods = tlsp['nbestperiods']
        lspbestperiods = nbestperiods[::]

        d = {
            # best power recovery specific stats
            'bestperiod_1':tlsp['tlsresult']['period'],
            't0_1':tlsp['tlsresult']['T0'],
            'duration_1':tlsp['tlsresult']['duration'],
            'depth_1':tlsp['tlsresult']['depth'],
            'rp_rs_1':tlsp['tlsresult']['rp_rs'], #Rp/Rstar, different bc LD
            'snr_1':tlsp['tlsresult']['snr'],
            'odd_even_mismatch_1':tlsp['tlsresult']['odd_even_mismatch'],
            'sde_1':tlsp['tlsresult']['SDE'],
            # next two best peaks...
            'bestperiod_2':lspbestperiods[1],
            'bestperiod_3':lspbestperiods[2],
            # the epochs are not actually found. NOTE to solve this would need to
            # implement something like transitleastsquares.stats.final_T0_fit for
            # each peak.
            't0_2':tlsp['tlsresult']['T0'],
            't0_3':tlsp['tlsresult']['T0']
        }

        # If BLS recovers the injected period within +/- 0.1 days of the injected
        # period, and recovers t0 (mod P) within +/- 5% of the injected values, the
        # injected signal is "recovered". Otherwise, it is not.

        atol = 0.1 # days
        rtol = 0.05

        reldiff_1 = np.abs(d['t0_1'] - inj_dict['epoch']) % inj_dict['period']
        reldiff_2 = np.abs(d['t0_2'] - inj_dict['epoch']) % inj_dict['period']
        reldiff_3 = np.abs(d['t0_3'] - inj_dict['epoch']) % inj_dict['period']

        recovered_as_best_peak = False
        recovered_in_topthree_peaks = False

        if (
            np.abs(d['bestperiod_1'] - inj_dict['period']) < atol and
            reldiff_1 < rtol
        ):
            recovered_as_best_peak = True
            recovered_in_topthree_peaks = True

        elif (
            ( np.abs(d['bestperiod_2'] - inj_dict['period']) < atol and
            reldiff_2 < rtol )
            or
            ( np.abs(d['bestperiod_3'] - inj_dict['period']) < atol and
            reldiff_3 < rtol )
        ):
            recovered_in_topthree_peaks = True

        print(
        "inj_t0: {:.3f}, inj_period: {:.5f}, got_t0: {:.3f}, got_period: {:.3f}, reldiff_1: {:.3f}".
        format(inj_dict['epoch'], inj_dict['period'], d['t0_1'], d['bestperiod_1'], reldiff_1)
        )

        t = {
            'source_id':source_id,
            'tmag':tmag,
            'allnan':False,
            'recovered_as_best_peak':recovered_as_best_peak,
            'recovered_in_topthree_peaks':recovered_in_topthree_peaks
        }
        # t: star info
        # inj_dict: injection-recovery information
        # dtr_dict: detrending info, including Prot
        # d: TLS results
        outd = {**t, **infodict, **inj_dict, **dtr_dict, **d}
        outdf = pd.DataFrame(outd, index=[0])

        outdf.to_csv(outpath, index=False)
        print('made {}'.format(outpath))

        return

    except Exception as e:
        print('ERR!: {}'.format(repr(e)))
        t = {
            'source_id':source_id,
            'tmag':tmag,
            'allnan':True,
            'recovered_as_best_peak':False,
            'recovered_in_topthree_peaks':False
        }
        d = {
            'bestperiod_1':np.nan,
            't0_1':np.nan,
            'duration_1':np.nan,
            'depth_1':np.nan,
            'rp_rs_1':np.nan,
            'snr_1':np.nan,
            'odd_even_mismatch_1':np.nan,
            'sde_1':np.nan,
            'bestperiod_2':np.nan,
            'bestperiod_3':np.nan,
            't0_2':np.nan,
            't0_3':np.nan
        }
        outd = {**t, **inj_dict, **dtr_dict, **d}
        outdf = pd.DataFrame(outd, index=[0])

        outdf.to_csv(outpath, index=False)
        print('made {}'.format(outpath))

        return


def get_random_lc_paths(N_lcs=100):

    lcglob = 'cam?_ccd?/*_llc.fits'

    SECTORNUM = 6

    lcdirectory = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/'.
        format(SECTORNUM)
    )

    lcpaths = glob(os.path.join(lcdirectory, lcglob))
    np.random.shuffle(lcpaths)

    lcpaths = np.random.choice(lcpaths,size=N_lcs)

    return lcpaths


def get_cluster_lc_paths(N_lcs=100):
    """
    Draw light curves from a set of clusters that I think might be good in
    S14-S19 to test planet injection/recovery.

    These are in a list of 8 clusters, and have RP<14.
    """
    from cdips.utils.catalogs import get_cdips_catalog

    df = get_cdips_catalog(ver=0.6)
    df = df[~pd.isnull(df.cluster)]

    cluster_names = [
        "Stephenson_1", # logt 7.5, 100-500pc, Theia 73, we are familiar
        "Alpha_per",# logt 7.8, 100-200pc, Theia 133
        "RSG_5", # logt 7.5, 300-400pc.
        "kc19group_506", # 350 Myr, 100pc (!!!).
        "kc19group_507", # 350 Myr, 100-200pc.
        "AB_Dor", # ~120 Myr, 100-300pc from KC19.
        "Pleiades", # 120 Myr, 120 pc.
    ]
    special_names = [
        "UBC_1", # 350 Myr, 300-400pc, Theia 520
    ]

    sel = np.zeros(len(df)).astype(bool)
    for cluster_name in cluster_names:
        sel |= df.cluster.str.contains(cluster_name)

    # avoid e.g., "UBC_186" matching for "UBC_1".
    for cluster_name in special_names:
        sel |= (df.cluster == cluster_name)
        sel |= df.cluster.str.contains(cluster_name+',')

    sdf = df[sel]

    # NOTE: a further cut, to deal with less crappy data
    sel = sdf.phot_rp_mean_mag < 14
    sdf = sdf[sel]

    outdir = resultsdirectory
    camdpath = os.path.join(outdir, f'camd_{"_".join(cluster_names)}.png')
    if not os.path.exists(camdpath):
        plt.close('all')
        get_yval = (
            lambda _df: np.array(
                _df['phot_g_mean_mag'] + 5*np.log10(_df['parallax']/1e3) + 5
            )
        )
        get_xval = (
            lambda _df: np.array(
                _df['phot_bp_mean_mag'] - _df['phot_rp_mean_mag']
            )
        )
        fig, axs = plt.subplots(ncols=2)

        s=0.5
        axs[0].scatter(
            get_xval(sdf), get_yval(sdf), c='k', alpha=1, zorder=3,
            s=s, rasterized=False, linewidths=0.1, marker='o',
            edgecolors='k'
        )
        axs[0].set_xlabel("BP-RP [mag]")
        axs[0].set_ylabel("M_G [mag]")

        axs[1].scatter(
            get_xval(sdf), sdf.phot_g_mean_mag, c='k', alpha=1, zorder=3,
            s=s, rasterized=False, linewidths=0.1, marker='o',
            edgecolors='k'
        )
        axs[1].set_xlabel("BP-RP [mag]")
        axs[1].set_ylabel("G [mag]")

        fig.savefig(camdpath, bbox_inches='tight', dpi=400)

    # NOTE: OK!  Now, find the light curves, from S14-S19, corresponding to
    # these clusters.
    from cdips.utils.lcutils import make_lc_list
    lclistpath = (
        "/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/S14_S19_lc_list_20211019.txt"
    )

    if not os.path.exists(lclistpath):
        make_lc_list(
            lclistpath, sector_interval=[14,19]
        )

    lcdf = pd.read_csv(lclistpath, names=["lcpath"])
    lcdf['source_id'] = lcdf.lcpath.apply(
        lambda x:
        x.split('/')[-1].split('gaiatwo')[1].split('-')[0].lstrip('0')
    )

    sdf['source_id'] = sdf.source_id.astype(str)
    mdf = sdf.merge(lcdf, on="source_id", how="left")

    smdf = mdf[~pd.isnull(mdf.lcpath)]

    res = Counter(smdf.cluster)
    print(42*'-')
    print(
        f'Got {len(smdf)} T<14 light curves in {cluster_names} & {special_names}...'
    )
    print(
        f'Most common 40:\n{res.most_common(n=40)}'
    )

    outpath = f'S14_S19_20211020_lcpaths_{"_".join(cluster_names)}.csv'
    smdf.to_csv(outpath, index=False)
    print(f'Saved {outpath}')

    lcpaths = nparr(smdf.lcpath)
    np.random.shuffle(lcpaths)

    lcpaths = np.random.choice(lcpaths,size=N_lcs)

    return lcpaths


def main():

    check_dependencies()

    df_locor = get_inj_recov_results(
        {'method':'locor', 'break_tolerance':None, 'window_length':"prot"}
    )
    df_notch = get_inj_recov_results(
        {'method':'notch', 'break_tolerance':None, 'window_length':0.5}
    )
    df_pspline = get_inj_recov_results(
        {'method':'pspline', 'break_tolerance':0.5, 'window_length':0.5}
    )
    df_best = get_inj_recov_results(
        {'method':'best', 'break_tolerance':0.5, 'window_length':0.5}
    )

    descrps = ['notch', 'locor', 'pspline', 'best']
    dfs = [df_notch, df_locor, df_pspline, df_best]

    for df, d in zip(dfs, descrps):
        outstr = (
        """
        type: {}
        {} inj-recov experiments
        {} unique sourceids
        {} failed b/c allnan or other
        {} recovered as best peak
        {} recovered in top three peaks
        """
        ).format(
            d, len(df), len(np.unique(df['source_id'])),
            len(df[df['allnan']]),
            len(df[df['recovered_as_best_peak']]),
            len(df[df['recovered_in_topthree_peaks']])
        )
        print(textwrap.dedent(outstr))

    #
    # print some of the notch/best overlap
    #
    mdf = df_notch.merge(df_best, how='left', on=['source_id', 'period'],
                           suffixes=['_notch','_best'])

    pok_nbad = mdf[mdf['recovered_in_topthree_peaks_notch'] &
                   ~mdf['recovered_in_topthree_peaks_best']].sort_values(
                       by=['source_id','period'])
    nok_pbad = mdf[mdf['recovered_in_topthree_peaks_best'] &
                   ~mdf['recovered_in_topthree_peaks_notch']].sort_values(
                       by=['source_id','period'])

    print('notch found, but best didnt find for:\n{}'.format(
        pok_nbad[['source_id','period','sde_1_best','sde_1_notch']]
    ))

    print('best found, but notch didnt find for:\n{}'.format(
        nok_pbad[['source_id','period','sde_1_best','sde_1_notch']]
    ))

    print('\ndone\n')


if __name__=="__main__":
    main()
