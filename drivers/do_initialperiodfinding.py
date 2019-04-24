"""
given >~10k LCs per galactic field, we want to skim the cream off the top.
"""
from astropy.stats import LombScargle
from transitleastsquares import transitleastsquares

import os
from glob import glob
from datetime import datetime
from astropy.io import fits

def run_periodograms(source_id, tfa_time, tfa_mag):

    #
    # Lomb scargle w/ uniformly weighted points.
    #
    ls = LombScargle(tfa_time, tfa_mag, tfa_mag*1e-3)
    freq, power = ls.autopower()
    ls_fap = ls.false_alarm_probability(power.max())
    ls_period = 1/freq[np.argmax(power)]

    #
    # Transit least squares, ditto.
    #
    f_x0 = 1e4
    m_x0 = 10
    tfa_flux = f_x0 * 10**( -0.4 * (tfa_mag - m_x0) )
    model = transitleastsquares(tfa_time, tfa_flux)
    results = model.power(use_threads=1)

    tls_sde = results.SDE
    tls_period = results.period

    r = [source_id, ls_period, ls_fap, tls_period, tls_sde]

    return r


def get_lc_data(lcpath):

    hdul = fits.open(lcpath)

    tfa_time = hdul[1].data['TMID_BJD']
    tfa_mag = hdul[1].data['TFA2']

    hdul.close()

    source_id = os.path.basename(lcpath).split('_')[0]

    return source_id, tfa_time, tfa_mag


def periodfindingworker(lcpath):

    source_id, tfa_time, tfa_mag = get_lc_data(lcpath)
    r = run_periodograms(source_id, tfa_time, tfa_mag)


def do_initial_period_finding(sectornum=6, nworkers=52, maxworkertasks=1000):

    lcdirectory = (
        '/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-{}/'.
        format(sectornum)
    )
    lcglob = 'cam?_ccd?/*_llc.fits'

    lcpaths = glob(os.path.join(lcdirectory, lcglob))

    print('%sZ: %s files to run initial periodograms on' %
          (datetime.utcnow().isoformat(), len(lcpaths)))

    pool = mp.Pool(nworkers,maxtasksperchild=maxworkertasks)

    tasks = [(x) for x in lcpaths]

    # fire up the pool of workers
    results = pool.map(periodfindingworker, tasks)

    # wait for the processes to complete work
    pool.close()
    pool.join()

    import IPython; IPython.embed()

    return {result for result in results}

