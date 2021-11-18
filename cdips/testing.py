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

import numpy as np
from numpy.testing import assert_approx_equal

def check_dependencies():
    """
    This is a general "dependency-catcher" function, which checks whether
    a few packages important for can be imported.

    Versions for some packages
    some dependencies can be hidden; this makes them explicit, especially if
    you are worried about .
    """

    import cdips as cd

    # require astroquery >= 0.4.0
    # see: https://astroquery.readthedocs.io/en/latest/
    # `pip install --pre astroquery`
    import astroquery as aq
    updatemsg = (
        'Need to update astroquery. Preferred method '
        '`pip install --pre astroquery`'
    )
    assert int(aq.__version__.split('.')[1]) >= 4, updatemsg

    # require astropy >= 4.0
    # generally `conda update astropy` should be fine.
    import astropy as ap
    updatemsg = (
        'Need to update astropy. Preferred method '
        '`conda update astropy`'
    )
    assert int(ap.__version__.split('.')[0]) >= 4, updatemsg

    #
    # wotan bleeding edge install 
    # or better (bleeding edge): clone and setup.py install
    #
    import pygam
    import wotan
    from wotan import version
    wotanversion = version.WOTAN_VERSIONING
    wotanversiontuple = tuple(wotanversion.split('.'))
    updatemsg = (
        'Need to update wotan. Please clone & setup.py '
        'install https://github.com/hippke/wotan'
    )
    assert int(wotanversiontuple[0]) >= 1
    assert int(wotanversiontuple[1]) >= 10, updatemsg

    #
    # TLS bleeding edge install for verbose kwarg, among others
    #
    from transitleastsquares.tls_constants import TLS_VERSION
    tlsversion = TLS_VERSION.split(' ')[4].split('.')
    updatemsg = (
        'Need to update transitleastsquares. Please clone & setup.py '
        'install https://github.com/hippke/tls'
    )
    assert int(tlsversion[0]) >= 1
    assert int(tlsversion[1]) >= 0, updatemsg
    assert int(tlsversion[2]) >= 28, updatemsg

    # check if pspline works with the expected number of args as of wotan v1.10
    # nb. also requires the bugfix with the stdev cut.
    rng = np.random.default_rng(42)
    time = np.linspace(0,10,100)
    flux = (
        np.ones(len(time)) + 1e-2*np.linspace(-1,1,len(time)) +
        rng.random(len(time))*1e-3
    )
    edge_cutoff = 0
    max_splines = 4
    stdev_cut = 1.5
    return_nsplines = False
    verbose = False
    from wotan.pspline import pspline
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    trend_flux, n_splines = pspline(
        time, flux, edge_cutoff, max_splines, stdev_cut, return_nsplines,
        verbose
    )

    #
    # notch and locor:
    # clone and setup.py install https://github.com/lgbouma/Notch_and_LOCoR,
    # which was forked from Aaron Rizzuto's implentation.
    # also requires
    # clone and setup.py install https://github.com/evertrol/mpyfit
    #
    from notch_and_locor.core import sliding_window
    from notch_and_locor.core import rcomb

    print('testing.check_dependencies passed!')

    #
    # photutils: used in vetting report creation
    # $ conda install -c conda-forge photutils
    #
    import photutils



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
