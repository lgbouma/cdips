"""
testing.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Assertion statements to require that features extraction from light curves
equal particular values. These are useful for instance when testing different
detrending methods, and during injection recovery analyses.

Contents:
    assert_lsperiod_is_approx
    assert_spdmperiod_is_approx
    assert_tlsperiodepoch_is_approx
"""

from numpy.testing import assert_approx_equal

def assert_lsperiod_is_approx(time, flux, err, target_period, significant=4,
                              verbose=True):
    """
    Given a light curve, require the Lomb Scargle period to be near a target
    value.

    args:
        time, flux, err: np.ndarrays
        target_period: float
        significant: int, number of significant digits used in assertion
        statement.
    """
    from astropy.stats import LombScargle

    period_min = target_period/10
    period_max = target_period*10

    ls = LombScargle(time, flux, err)
    freq, power = ls.autopower(
        minimum_frequency=1/period_max, maximum_frequency=1/period_min,
        samples_per_peak=20
    )

    ls_fap = ls.false_alarm_probability(power.max())

    ls_period = 1/freq[np.argmax(power)]

    if verbose:
        msg = (
            f'LS Period: got {ls_period:.4f} d, target {target_period:.4f} d'
        )
        print(msg)

    assert_approx_equal(ls_period, target_period, significant=significant)


def assert_spdmperiod_is_approx(time, flux, err, target_period, significant=4,
                                verbose=True):
    """
    Given a light curve, require the Stellingwerf Phase Dispersion Minimization
    period to be near a target value.
    """
    raise NotImplementedError


def assert_tlsperiodepoch_is_approx(
    time, flux, err, target_period, target_epoch, significant=4
):
    from transitleastsquares import transitleastsquares
    raise NotImplementedError
