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
import jax
import numpy as np, matplotlib.pyplot as plt
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
from nuance import core, Star
from nuance.core import gp_model
from nuance.utils import minimize, sigma_clip_mask
from nuance import utils

from nuance.linear_search import linear_search
from nuance.periodic_search import periodic_search

from cdips.utils.periodogramutils import find_good_peaks, flag_harmonics
from cdips.lcproc.detrend import basic_cleaning

jax.config.update("jax_enable_x64", True)

def _build_gp(params, time):

    kernel = kernels.quasisep.SHO(
        sigma = jnp.exp(params["log_sigma"]),
        omega = 2. * jnp.pi / jnp.exp(params["log_period"]),
        quality = jnp.exp(params["log_Q"]),
    )
    #kernel = kernels.quasisep.SHO(10.0, 10.0, 0.002)

    #FIXME TODO LEFT OFF HERE...
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

    dtrcachepath = join(cachedir, f'{star_id}_nuance_result.pkl')
    if os.path.exists(dtrcachepath):
        LOGINFO(f'Found {dtrcachepath}, loading & returning.')
        with open(dtrcachepath, 'rb') as f:
            d = pickle.load(f)
        return d

    assert isinstance(time, np.ndarray)
    assert isinstance(flux, np.ndarray)
    assert isinstance(flux_err, np.ndarray)

    assert cleaning_type in ['basic', 'iterativegp', 'none']

    if cleaning_type == 'none':
        pass

    elif cleaning_type == 'basic':
        # default: slide_clip_lo=20, slide_clip_hi=3, clip_window=3
        time, flux, _ = basic_cleaning(time, flux)

    elif cleaning_type == 'iterativegp':

    # basic light curve cleaning: mask NaNs and normalize

    mask = np.isnan(time) | np.isnan(flux)
    time = time[~mask].astype(float)
    flux = flux[~mask].astype(float)

    flux_median = np.median(flux)
    flux /= flux_median
    flux_err /= flux_median

    if lsp_dict is None:
        from cdips.lcproc.lccleaning import _rotation_period
        lsp_options = {'period_min':0.1, 'period_max':20}
        lsp_dict = _rotation_period(time, flux, lsp_options=lsp_options)

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

    # define a GP using an appropriate kernel
    initial_params = {
        "log_period": jnp.log(lsp_dict['ls_period']),
        "log_Q": jnp.log(100),  # 10: overfitting more in coherence time, 1000: less
        "log_sigma": jnp.log(1e-1),
        "error": np.mean(flux_err),
    }

    mu, nll = gp_model(time, flux, _build_gp)

    gp_params = minimize(nll, initial_params)

    ##FIXME FIXME FIXME
    #crap_params = minimize(nll, initial_params)

    #from copy import deepcopy
    #gp_params = deepcopy(initial_params)

    #crap_params2 = minimize(nll, initial_params, param_names=['log_Q'])
    #import IPython; IPython.embed()
    #FIXME

    # viz 1: plot optimized gp model...
    prot = lsp_dict['ls_period']
    if make_optimized_gp_plot:
        plt.close("all")
        title = f"LS P={prot:.1f}d, opt gp param = {gp_params}"
        fig, axs = plt.subplots(nrows=2, figsize=(8, 6))
        axs[0].plot(time, flux, ".", c="0.7", ms=2, label="flux")
        axs[1].plot(time, flux, ".", c="0.7", ms=2, label="flux")
        label = f"GP model"
        axs[1].plot(time, mu(gp_params), c="k", label=label, lw=0.5)
        axs[1].plot(time, mu(initial_params), c="r", label=label, lw=0.5)
        axs[1].set_xlabel("time")
        axs[0].set_ylabel("flux")
        axs[1].set_ylabel("flux")
        axs[1].legend()
        fig.tight_layout()
        outpath = join(cachedir, f'{star_id}_gpopt_initial_gp_model.png')
        LOGINFO(f'wrote {outpath}')
        fig.savefig(outpath)

    # compute gp used for search
    gp = _build_gp(gp_params, time)

    # Run linear search
    ls = linear_search(time, flux, gp=gp)(epoch_grid, duration_grid)

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
    min_index_not_true = next(i for i, val in enumerate(pgdict['periodmask']) if not val)
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
        fig.savefig(outpath)


    outdict = {
        # GP optimization
        'gp_prot': lsp_dict['ls_period'],
        'initial_params': initial_params,
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
