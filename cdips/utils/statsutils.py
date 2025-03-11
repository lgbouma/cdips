"""
Contents:
    compute_median_and_uncertainty_given_param
    compute_median_and_uncertainty_given_chisq
"""
import numpy as np
from typing import Tuple, Sequence

def compute_median_and_uncertainty_given_param(
    param: Sequence[float]
) -> Tuple[float, float, float]:
    """Compute the median and ±1σ uncertainties of given parameter values.

    The function calculates the median value and the uncertainties defined as
    the differences between the median and the 16th and 84th percentiles,
    which approximate the -1σ and +1σ uncertainties, respectively.

    Args:
        param: A sequence of parameter values (e.g., list or numpy array).

    Returns:
        A tuple containing:
            - median (float): The median of the input values.
            - sigma_minus (float): The difference between the median and
              the 16th percentile (i.e., median - 16th percentile).
            - sigma_plus (float): The difference between the 84th percentile and
              the median (i.e., 84th percentile - median).
    """
    median_val = np.median(param)
    perc16 = np.percentile(param, 16)
    perc84 = np.percentile(param, 84)
    sigma_minus = median_val - perc16
    sigma_plus = perc84 - median_val
    return median_val, sigma_minus, sigma_plus


def compute_median_and_uncertainty_given_chisq(
    param: np.ndarray, chisq: np.ndarray
) -> Tuple[float, float, float]:
    """Compute the median and uncertainties from chi-squared values.

    Given arrays of parameter values and corresponding chi^2 values,
    this function computes the likelihood and the normalized probability
    distribution. The median is defined as the 50th percentile, and the
    lower and upper uncertainties are computed from the 16th and 84th
    percentiles, respectively.

    Args:
        param (np.ndarray): One-dimensional array of parameter values.
        chisq (np.ndarray): One-dimensional array of chi-squared values
            corresponding to the parameter values.

    Returns:
        Tuple[float, float, float]:
            A tuple containing the median, the lower uncertainty (median minus
            the 16th percentile), and the upper uncertainty (84th percentile
            minus the median).
    """
    # Compute likelihood from chi-squared
    likelihood = np.exp(-0.5 * chisq)

    # Normalize likelihood to obtain probability density
    norm = np.trapz(likelihood, param)
    prob = likelihood / norm

    # Compute differential step size
    dx = np.gradient(param)

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(prob * dx)

    # Interpolate to find the 16th, 50th, and 84th percentiles
    lower = np.interp(0.16, cdf, param)
    median = np.interp(0.5, cdf, param)
    upper = np.interp(0.84, cdf, param)

    err_low = median - lower
    err_high = upper - median

    return median, err_low, err_high

