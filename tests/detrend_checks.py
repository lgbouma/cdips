"""
[environment: phtess2]

Randomly select N=100 (or 1000) TFA CDIPS light curves.
(Optionally select them to be only the LCs with strong LS FAPs already?)

Then inject a 0.25% = 2.5 mmag central-transit planet with periods in the range
1–12 d.  Try recovering via TLS (a) without detrending, (b) with detrending.
(For each light curve, do say 10 experiments).

For case (b), first-pass try using the robust Huber spline (with w = 0.3 d) and
also the sliding biweight (w = 0.25 d). For a sliding biweight, if w/T14 > 2.2,
most (> 98%) of the flux integral is preserved. So anything between w=0.25 days
to w=0.5 days should be good...

usage: $ python -u detrend_checks.py &> dtr.log &
       $ tail -f dtr.log | grep 'inj_'
"""

import numpy as np, pandas as pd
import os, textwrap
from glob import glob
from datetime import datetime
from astropy.io import fits

import batman
import astropy.units as u, astropy.constants as c
from wotan import flatten

from astrobase import periodbase

from cdips.lcproc import mask_orbit_edges as moe
from cdips.utils import lcutils as lcu

import multiprocessing as mp

SECTORNUM = 6

lcdirectory = (
    '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/'.
    format(SECTORNUM)
)
resultsdirectory = (
    '/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/detrend_checks'
)

def main():

    ##########################################
    np.random.seed(42)

    overwrite = 0 # whether to overwrite ALL previous saved csvs

    N_periods_per_lc = 10 #10
    N_lcs = 1000 #100

    P_lower = 1        # 1 day
    P_upper = 10       # 10 days
    inj_depth = 0.005 # 5 mmag, 0.5% central transit

    # For case (b), first-pass try using the robust Huber spline (with w = 0.3 d) and
    # also the sliding biweight (w = 0.25 d). For a sliding biweight, if w/T14 > 2.2,
    # most (> 98%) of the flux integral is preserved. So anything between w=0.25 days
    # to w=0.5 days should be good...
    dtr_dict = {
        'method':'hspline',    # none, hspline, or biweight
        'window_length':0.3,   # for the detrending method
        'break_tolerance':0.3  # make same as window_length
    }

    nworkers = 52
    maxworkertasks = 1000
    ##########################################

    if overwrite:
        csvpaths = glob(os.path.join(resultsdirectory,'worker_output','*.csv'))
        for c in csvpaths:
            os.remove(c)

    _lcpaths = get_lc_paths(N_lcs=N_lcs)
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
            (lcpath, inj_dict, dtr_dict)
        )

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)
    results = pool.map(inj_recov_worker, tasks)
    pool.close()
    pool.join()

    # merge and save results
    csvpaths = glob(os.path.join(resultsdirectory,'worker_output','*.csv'))
    df = pd.concat((pd.read_csv(f) for f in csvpaths))
    outpath = os.path.join(resultsdirectory,
                           'detrend_check_method-{}_window_length-{}.csv'.
                           format(dtr_dict['method'],dtr_dict['window_length']))
    df.to_csv(outpath, index=False)
    print('made {}'.format(outpath))



def inj_recov_worker(task):

    lcpath, inj_dict, dtr_dict = task

    # output is a csv file with (a) injected params, (b) whether injected
    # signals was recovered in first peak, (c) whether it was in first three
    # peaks.

    source_id, tfa_time, tfa_mag, xcc, ycc, ra, dec, tmag = (
        lcu.get_lc_data(lcpath, tfa_aperture='TFA2')
    )

    outpath = os.path.join(
        resultsdirectory,'worker_output',
        str(source_id)+
        "_period-{:.5f}".format(inj_dict['period'])+
        "_method-{}".format(dtr_dict['method'])+
        '_windowlength-{}'.format(dtr_dict['window_length'])+
        ".csv"
    )
    if os.path.exists(outpath):
        print('found {}'.format(outpath))
        return

    if np.all(pd.isnull(tfa_mag)):

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

    # inject
    inj_time, inj_flux, t0 = inject_signal(tfa_time, tfa_mag, inj_dict)
    inj_dict['epoch'] = t0

    # detrend
    flat_flux = detrend_lightcurve(inj_time, inj_flux, dtr_dict)

    # period find
    err = np.ones_like(flat_flux)*1e-4
    tlsp = periodbase.tls_parallel_pfind(inj_time, flat_flux, err,
                                         magsarefluxes=True,
                                         tls_rstar_min=0.1, tls_rstar_max=10,
                                         tls_mstar_min=0.1, tls_mstar_max=5.0,
                                         tls_oversample=8, tls_mintransits=1,
                                         tls_transit_template='default',
                                         nbestpeaks=5, sigclip=[50.,5.],
                                         nworkers=1)

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
    outd = {**t, **inj_dict, **dtr_dict, **d}
    outdf = pd.DataFrame(outd, index=[0])

    outdf.to_csv(outpath, index=False)
    print('made {}'.format(outpath))

    return


def detrend_lightcurve(time, flux, dtr_dict):

    if dtr_dict['method'] == 'none':
        flatten_lc = flux

    else:
        flatten_lc, _ = flatten(time, flux,
                                window_length=dtr_dict['window_length'],
                                method=dtr_dict['method'],
                                return_trend=True,
                                edge_cutoff=False,
                                break_tolerance=dtr_dict['window_length']
                               )

    return flatten_lc


def get_lc_paths(N_lcs=100):

    lcglob = 'cam?_ccd?/*_llc.fits'

    lcpaths = glob(os.path.join(lcdirectory, lcglob))
    np.random.shuffle(lcpaths)

    lcpaths = np.random.choice(lcpaths,size=N_lcs)

    return lcpaths


def inject_signal(tfa_time, tfa_mag, inj_dict):

    # mags to flux
    f_x0 = 1e4
    m_x0 = 10
    tfa_flux = f_x0 * 10**( -0.4 * (tfa_mag - m_x0) )
    tfa_flux /= np.nanmedian(tfa_flux)

    # ignore the times near the edges of orbits for TLS.
    time, flux = moe.mask_orbit_start_and_end(tfa_time, tfa_flux)
    del tfa_time, tfa_flux

    # initialize model to inject: 90 degrees, LD coeffs for 5000 K dwarf star
    # in TESS band (Claret 2018) no eccentricity, random phase, b=0, stellar
    # density set to 1.5x solar.  Eq (30) Winn 2010 to get a/Rstar.
    params = batman.TransitParams()

    density_sun = 3*u.Msun / (4*np.pi*u.Rsun**3)
    density_star = 1.5*density_sun
    a_by_rstar = ( ( c.G * (inj_dict['period']*u.day)**2 /
                     (3*np.pi) * density_star )**(1/3) ).cgs.value

    params.inc = 90.
    q1, q2 = 0.4, 0.2
    params.ecc = 0
    params.limb_dark = "quadratic"
    params.u = [q1,q2]
    w = np.random.uniform(low=np.rad2deg(-np.pi), high=np.rad2deg(np.pi))
    params.w = w
    params.per = inj_dict['period']
    t0 = np.nanmin(time) + inj_dict['epoch']
    params.t0 = t0
    params.rp = np.sqrt(inj_dict['depth'])
    params.a = a_by_rstar

    exp_time_minutes = 30.
    exp_time_days = exp_time_minutes / (24.*60)
    ss_factor = 10

    m_toinj = batman.TransitModel(params, time,
                                  supersample_factor=ss_factor,
                                  exp_time=exp_time_days)

    # calculate light curve and inject
    flux_toinj = m_toinj.light_curve(params)
    inj_flux = flux + (flux_toinj-1.)*np.nanmedian(flux)

    return time, inj_flux, t0


if __name__=="__main__":
    main()
