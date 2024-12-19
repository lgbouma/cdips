"""
Main wrapper function:
    run_nuance
"""
#############
## LOGGING ##
#############
import logging
from cdips import log_sub, log_fmt, log_date_fmt

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
import os, pickle
from os.path import join
from tqdm import tqdm
import numpy as np, matplotlib.pyplot as plt

from tinygp import kernels, GaussianProcess

from cdips.utils.periodogramutils import find_good_peaks, flag_harmonics
from cdips.lcproc.lccleaning import basic_cleaning, iterativegp_cleaning

def _build_gp(params, time):
    """single simple harmonic oscillator GP kernel; usually you'll want
    `nuance.kernels.rotation` instead.
    """

    kernel = kernels.quasisep.SHO(
        sigma = jnp.exp(params["log_sigma"]),
        omega = 2. * jnp.pi / jnp.exp(params["log_period"]),
        quality = jnp.exp(params["log_Q"]),
    )

    return GaussianProcess(kernel, time, diag=params["error"]**2, mean=1.)


def run_nuance(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    star_id: str,
    cachedir: str,
    n_cpus: int = 10,
    lsp_dict: dict = None,
    make_optimized_gp_plot: bool = True,
    make_snr_periodogram_plot: bool = True,
    cleaning_type: str = 'iterativegp',
    search_params: dict = {
        'period_min': 1,
        'period_max': 30,
        'oversample': 5
    },
    verbose: bool = True,
    ) -> dict:
    """
    Run the Nuance algorithm for transit searches.

    Produces vetting plots and search results in `cachedir`.

    Args:
        time (np.ndarray): Array of time values for the light curve.
        flux (np.ndarray): Array of flux values for the light curve.
        flux_err (np.ndarray): Array of flux error values for the light curve.
        star_id (str): Identifier for the star being analyzed.
        cachedir (str): Directory for storing cached results.
        cleaning_type (str): "iterativegp", "basic", or "none"
        lsp_dict (dict, optional): Dictionary containing the Lomb-Scargle
            periodogram parameters and results.  If not passed, will be
            computed.
        search_params (dict, optional): Parameters to be used for the nuance
            period search.
        n_cpus (int, optional): Number of CPUs to use for parallel computation.
            Defaults to 10.

    Returns:
        outdict (dict): Dictionary containing search results.
    """

    # set number of cpus to use
    cpu_count = os.cpu_count()
    if n_cpus > cpu_count / 2 :
        LOGWARNING(
            f'Found n_cpus > cpu_count / 2 '
            f'(n_cpus = {n_cpus}, cpu_count={cpu_count}...'
        )
        LOGWARNING(
            'Throttling n_cpus to avoid overflow...'
        )
        n_cpus = cpu_count / 2

    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_cpus}"

    # import jax after the XLA_FLAGS environment variable is set
    import jax
    import jax.numpy as jnp

    from nuance import core, Star
    from nuance.core import gp_model
    from nuance.utils import minimize
    from nuance import utils
    from nuance.kernels import rotation

    from nuance.linear_search import linear_search
    from nuance.periodic_search import periodic_search

    jax.config.update("jax_enable_x64", True)

    # if this has been done before pull from the cache
    dtrcachepath = join(cachedir, f'{star_id}_nuance_result.pkl')
    if os.path.exists(dtrcachepath):
        LOGINFO(f'Found {dtrcachepath}, loading & returning.')
        with open(dtrcachepath, 'rb') as f:
            d = pickle.load(f)
        return d

    assert isinstance(time, np.ndarray)
    assert isinstance(flux, np.ndarray)
    assert isinstance(flux_err, np.ndarray)

    # Drop NaNs & median normalize.
    mask = np.isnan(flux) | np.isnan(time)
    time = time[~mask].astype(float)
    flux = flux[~mask].astype(float)

    flux_median = np.median(flux)
    flux /= flux_median
    flux_err /= flux_median

    if verbose:
        N = len(time)
        LOGINFO(f'{star_id}: Got N={N} points.')

    assert cleaning_type in ['basic', 'iterativegp', 'none']

    if cleaning_type == 'none':
        pass

    elif cleaning_type == 'basic':
        # default: slide_clip_lo=20, slide_clip_hi=3, clip_window=3
        time, flux, dtr_stages_dict = basic_cleaning(time, flux)

    elif cleaning_type == 'iterativegp':
        # NOTE: this 'cleaning' is further below
        pass

    if lsp_dict is None:
        from cdips.lcproc.lccleaning import _rotation_period
        lsp_options = {'period_min':0.1, 'period_max':20}
        lsp_dict = _rotation_period(time, flux, lsp_options=lsp_options)
        prot = lsp_dict['ls_period']

    if verbose:
        LOGINFO(f'{star_id}: Got Prot={lsp_dict["ls_period"]} days.')

    # define period grid.  recall regular grids in period rather than freq
    # would oversample at large periods and undersample at small periods.
    period_min, period_max, oversample = (
        search_params['period_min'],
        search_params['period_max'],
        search_params['oversample']
    )
    t_baseline = np.nanmax(time) - np.nanmin(time)
    f_min = 1 / period_max
    f_max = 1 / period_min
    N_freq = int( oversample * t_baseline * f_max )
    freq_grid = np.linspace(f_min, f_max, N_freq)
    period_grid = 1 / freq_grid

    if verbose:
        LOGINFO(f'{star_id}: Will search over N_freq={N_freq} frequencies.')

    # same duration grid as notch
    duration_grid = np.array([0.75, 1., 2., 4.]) / 24

    # epoch grid.  over 2 minute cadence this is probably overkill.
    epoch_grid = time.copy()

    ###########################################
    # define a GP using an appropriate kernel #
    ###########################################
    # build GP using the "rotation" kernel, which is a re-implementation of
    # https://celerite2.readthedocs.io/en/latest/api/python/#celerite2.terms.RotationTerm
    # but with two exponential dampings, over "short_scale" and "long_scale",
    # in addition to the quality factors which are supposed to help with this
    # anyway...
    # This is discussed in Section 4.2.2 of the nuance paper.
    build_gp, init_params = rotation(prot, flux_err.mean(), long_scale=0.5)
    mu, nll = gp_model(time, flux, build_gp)

    gp_params = minimize(
        nll, init_params, ["log_sigma", "log_short_scale", "log_short_sigma", "log_long_sigma"]
    )
    gpfitted_time, gpfitted_flux = time*1., flux*1.

    # NOTE: yo dog.  i herd you don't like overfitting.  so i tried fitting
    # your overfitting to prevent the overfitting.  and i got some more
    # overfitting.
    RUN_OPTIMIZATION = 0
    if not RUN_OPTIMIZATION:
        LOGWARNING('skipping iterativegp cleaning because there is no point.')
    if RUN_OPTIMIZATION:
        init_gp_params = minimize(nll, gp_params)

        if cleaning_type == 'iterativegp':
            gpfitted_time, gpfitted_flux, gp_params, dtr_stages_dict = (
                iterativegp_cleaning(
                    time, flux, nll, init_gp_params, mu,
                    N_iter=3, sigma_clip=3, clipwindow=5, verbose=True
                )
            )

    # viz 1: plot optimized gp model...
    if make_optimized_gp_plot:
        plt.close("all")
        title = f"LS P={prot:.1f}d, opt gp param = {gp_params}"
        LOGINFO(title)
        fig, axs = plt.subplots(nrows=2, figsize=(8, 6))
        axs[0].plot(time, flux, ".", c="lightgray", ms=3, label="all")
        axs[0].plot(gpfitted_time, gpfitted_flux, ".", c="k", ms=2, label="gpfitted")
        axs[1].plot(time, flux, ".", c="lightgray", ms=3, label="all")
        axs[1].plot(gpfitted_time, gpfitted_flux, ".", c="k", ms=2, label="gpfitted")

        # plot two GPs: one built on the masked time/flux, and the other built
        # on the full time-series.  ideally, the latter shouldn't be
        # overfitting...
        mu, _ = gp_model(gpfitted_time, gpfitted_flux, build_gp)
        gp_mean = mu(gp_params)
        split_idxs = [
            0,
            *np.flatnonzero(np.diff(gpfitted_time) > 5*30 / 60 / 24),
            len(time),
        ]
        _ = True
        for i in range(len(split_idxs) - 1):
            x = gpfitted_time[split_idxs[i] + 1 : split_idxs[i + 1]]
            y = gp_mean[split_idxs[i] + 1 : split_idxs[i + 1]]
            axs[1].plot(x, y, "k", label="GP mean" if _ else None, lw=1)
            _ = False

        _mu, _ = gp_model(time, flux, build_gp)
        label = f"GP model (all time)"
        axs[1].plot(time, _mu(gp_params), c="darkgray", label=label, lw=0.3)

        axs[1].set_xlabel("time")
        axs[0].set_ylabel("flux")
        axs[1].set_ylabel("flux")
        axs[1].legend()
        fig.tight_layout()
        outpath = join(cachedir, f'{star_id}_gpopt_initial_gp_model.png')
        LOGINFO(f'wrote {outpath}')
        fig.savefig(outpath, dpi=300)

    # compute gp used for search
    gp = build_gp(gp_params, time)

    # Run linear search
    ls = linear_search(
        time, flux, gp=gp, backend='cpu', batch_size=n_cpus
    )(epoch_grid, duration_grid)

    # Run periodic search:
    # Pick the best period that is not a harmonic of the rotation period.
    snr_function = jax.jit(core.snr(time, flux, gp=gp))
    ps_function = periodic_search(epoch_grid, duration_grid, ls, snr_function)
    snr, params = ps_function(period_grid)

    param_dict = {'t0': params[:,0], 'tdur': params[:,1], 'P': params[:,2]}

    pgdict = find_good_peaks(period_grid, snr, param_dict=param_dict)
    pgdict['periodmask'] = (
        flag_harmonics(pgdict['nbestperiods'], prot)
        |
        flag_harmonics(pgdict['nbestperiods'], 0.5*prot)
    )
    min_index_not_true = (
        next(i for i, val in enumerate(pgdict['periodmask']) if not val)
    )
    period_ind = int(
        np.argwhere(
            np.float64(pgdict['nbestparams']['P'][min_index_not_true]
        ) == period_grid
    ))

    t0, D, P = params[period_ind]
    if verbose:
        LOGINFO(f'{star_id}: getting P={P}')

    linear, found, noise = core.separate_models(time, flux, gp=gp)(t0, D, P)
    detrended = flux - noise - linear
    phi = utils.phase(time, t0, P)

    if make_snr_periodogram_plot:
        # Plot SNR periodogram
        plt.close("all")
        fig = plt.figure(figsize=(8.5, 4))

        ax = plt.subplot(121, xlabel="period", ylabel="SNR")
        ax.axvline(P, c="0.8", ls="-", label="top")
        ax.plot(period_grid, snr)
        ax.legend()

        ax = plt.subplot(222, xlabel="time", ylabel="flux")
        ax.plot(time, flux, ".", c="0.8")
        ax.plot(time, found + 1, c="k", label="found")
        ax.plot(time, detrended + 1, ".", c="C0", label="dtr")
        ax.legend()

        ax = plt.subplot(224, xlabel="time", ylabel="flux", xlim=(-0.2, 0.2))
        plt.plot(phi, detrended, ".", c=".8")
        bx, by, be = utils.binn_time(phi, detrended, bins=7 / 60 / 24)
        plt.errorbar(bx, by, yerr=be, fmt=".", c="k")

        plt.tight_layout()

        outpath = join(
            cachedir, f'{star_id}_gpopt_periodic_transit_search_result.png'
        )
        LOGINFO(f'wrote {outpath}')
        fig.savefig(outpath, dpi=300)


    outdict = {
        # GP optimization
        'gp_prot': lsp_dict['ls_period'],
        #'initial_params': initial_params,
        'gp_params': gp_params,
        # search result
        't0': t0,
        'D': D,
        'P': P,
        # periodogram
        'period_grid': period_grid,
        'snr': snr,
        # data & processing stages
        'time': time,
        'phi': phi,
        'flux': flux,
        'linear': linear,
        'found': found,
        'noise': noise,
        'detrended': detrended,
    }
    if not os.path.exists(dtrcachepath):
        with open(dtrcachepath, 'wb') as f:
            pickle.dump(outdict, f)
            print(f"Wrote {dtrcachepath}")

    return outdict
