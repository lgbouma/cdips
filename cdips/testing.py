"""
testing.py - Luke Bouma (bouma.luke@gmail) - Nov 2020

Assertion statements to require that features extraction from light curves
equal particular values. These are useful for instance when testing different
detrending methods, and during injection recovery analyses.

Contents:

    check_dependencies

    assert_lsperiod_is_approx
    assert_spdmperiod_is_approx
    assert_tlsperiodepoch_is_approx
"""

from numpy.testing import assert_approx_equal

def check_dependencies():
    """
    some dependencies can be hidden; this makes them explicit, especially if
    you are worried about versions for some packages.
    """
    #
    # wotan install https://wotan.readthedocs.io/en/latest/Installation.html
    #
    import pygam
    import wotan
    from wotan import version
    wotanversion = version.WOTAN_VERSIONING
    wotanversiontuple = tuple(wotanversion.split('.'))
    assert int(wotanversiontuple[0]) >= 1
    assert int(wotanversiontuple[1]) >= 9

    #
    # notch and locor:
    # clone and setup.py install https://github.com/lgbouma/Notch_and_LOCoR,
    # which was forked from Aaron Rizzuto's implentation.
    #
    from notch_and_locor.core import sliding_window
    from notch_and_locor.core import rcomb

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
