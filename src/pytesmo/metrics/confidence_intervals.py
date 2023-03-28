import numpy as np
from scipy import stats

from pytesmo.metrics import pairwise
from pytesmo.metrics.tcol import tcol_metrics


def has_analytical_ci(metric_func):
    """
    Checks whether an analytical CI implementation is available.

    Parameters
    ----------
    metric_func : callable
        Function that calculates metric value. Must be from
        :py:mod:`pytesmo.metrics`.

    Returns
    -------
    has_ci : bool
        ``True`` if there is a function with name ``metric_func.__name__ +
        "_ci"`` in :py:mod:`pytesmo.metric_cis`.
    """
    return hasattr(pairwise, metric_func.__name__ + "_ci")


def with_analytical_ci(metric_func, x, y, alpha=0.05):
    """
    Evaluates metric and analytical confidence interval.

    This calculates the metric value and an analytical confidence interval. For
    this to work, the metric function must be from
    :py:mod:`pytesmo.pairwise_metric_cis` Use
    :py:func:`pytesmo.metrics.has_analytical_ci` to check whether it's
    available.

    Parameters
    ----------
    metric_func : callable or str
        Function that calculates metric value. Must be from
        :py:mod:`pytesmo.metrics`, and have an analytical CI function
        implemented in :py:mod:`pytesmo.metric_cis`.
        The metric function must have the following signature:
        ``(x : np.ndarray, y : np.ndarray) -> float``
        Alternatively can be the name of the function in `pytesmo.metrics`.
    x, y : np.ndarray
        Data to be compared.
    alpha : float, optional
        Confidence level, default is 0.05.

    Returns
    -------
    m : float
        Metric value
    lower : float
        Lower bound of confidence interval
    upper : float
        Upper bound of confidence interval

    Raises
    ------
    ValueError :
        If no analytical CI function is available."""
    if isinstance(metric_func, str):
        metric_func = getattr(pairwise, metric_func)
    m = metric_func(x, y)
    name = metric_func.__name__
    if has_analytical_ci(metric_func):
        ci = getattr(pairwise, name + "_ci")(x, y, m, alpha)
    else:
        raise ValueError(f"No analytical CI implemented for {name}.")
    return m, ci[0], ci[1]


def with_bootstrapped_ci(
    metric_func,
    x,
    y,
    alpha=0.05,
    method="percentile",
    nsamples=1000,
    minimum_data_length=100,
):
    """
    Evaluates metric and bootstraps confidence interval.

    This calculates the metric value and uses bootstrapping to find a
    confidence interval for the metric.
    This works only for pairwise metrics, use
    :py:func:`pytesmo.metrics.tcol_metrics_with_bootstrap_ci` for TCA metrics.

    Parameters
    ----------
    metric_func : callable
        Function that calculates metric value.  The metric function must have
        the following signature:
        ``(x : np.ndarray, y : np.ndarray) -> float``
    x, y : np.ndarray
        Data to be compared.
    alpha : float, optional
        Confidence level, default is 0.05.
    method : str, optional
        The method to use to calculate confidence intervals. Available methods
        are:

        - "percentile" (default): Uses the percentiles of the bootstrapped
          metric distribution.
        - "basic": Uses the percentiles of the differences to the original
          metric value.
        - "BCa": Bias-corrected and accelerated bootstrap.

        For more info and a comparison of the methods, see [Gilleland2010]_,
        especially Table 6 on page 38.
    nsamples : int, optional
        Number of bootstrap samples, default is 1000.
    minimum_data_length : int, optional
        Minimum amount of data required to do bootstrapping. Default is 100.

    Returns
    -------
    m : float
        Metric value for pairwise metrics
    lower : float or array of floats
        Lower bound of confidence interval
    upper : float or array of floats
        Upper bound of confidence interval

    References
    ----------
    .. [Gilleland2010] Gilleland, E. (2010). *Confidence Intervals for Forecast\
    Verification* (No. NCAR/TN-479+STR). University Corporation for Atmospheric\
    Research. doi:10.5065/D6WD3XJM
    """
    # Prototype, might be better to implement this in Cython if it's too slow.
    # Then it would probably be best to make this function only a lookup table
    # which calls a cpdef'd method, which itself calls the cdef'd metric
    # function with a cdef'd bootstrap implementation.
    n = len(x)
    if n < minimum_data_length:
        raise ValueError(
            "Not enough data for bootstrapping. Your data should contain at"
            f" least {minimum_data_length} samples.\n"
            f"You can pass 'minimum_data_length={n}' if you want to do"
            "bootstrapping nevertheless."
        )
    orig_metric = metric_func(x, y)
    bootstrapped_metric = np.empty(nsamples, dtype=float)
    if method == "BCa":
        orig_jk = _jackknife(metric_func, x, y)
    for i in range(nsamples):
        idx = np.random.choice(n, size=n)
        _x, _y = x[idx], y[idx]
        bootstrapped_metric[i] = metric_func(_x, _y)
    if method == "percentile":
        lower, upper = _percentile_bs_ci(
            bootstrapped_metric, orig_metric, alpha
        )
    elif method == "basic":
        lower, upper = _basic_bs_ci(bootstrapped_metric, orig_metric, alpha)
    elif method == "BCa":
        lower, upper = _BCa_bs_ci(
            bootstrapped_metric, orig_metric, alpha, orig_jk
        )
    return orig_metric, lower, upper


def tcol_metrics_with_bootstrapped_ci(
    x,
    y,
    z,
    ref_ind=0,
    alpha=0.05,
    method="percentile",
    nsamples=1000,
    minimum_data_length=100,
):
    """
    Evaluates triple collocation metrics and bootstraps confidence interval.

    This calculates the SNR, error standard deviation, and scaling parameter
    value using Triple Collocation Analysis and uses bootstrapping to find
    confidence intervals.

    Parameters
    ----------
    x, y, z : np.ndarray
        Data to be compared.
    ref_ind: int
        Index of reference data set for estimating scaling
        coefficients. Default: 0 (x)
    alpha : float, optional
        Confidence level, default is 0.05.
    method : str, optional
        The method to use to calculate confidence intervals. Available methods
        are:

        - "percentile" (default): Uses the percentiles of the bootstrapped
          metric distribution.
        - "basic": Uses the percentiles of the differences to the original
          metric value.
        - "BCa": Bias-corrected and accelerated bootstrap.

        For more info and a comparison of the methods, see [Gilleland2010]_,
        especially Table 6 on page 38.
    nsamples : int, optional
        Number of bootstrap samples, default is 1000.
    minimum_data_length : int, optional
        Minimum amount of data required to do bootstrapping. Default is 100.

    Returns
    -------
    snr_result : tuple
        ``(snr, lower, upper)``, where each entry is an array of length 3.
    err_std_result : tuple
        ``(err_std, lower, upper)``, where each entry is an array of length 3.
    beta_result : tuple
        ``(beta, lower, upper)``, where each entry is an array of length
        3. Note that beta is always 1 for the reference dataset (see
        `ref_ind`), therefore the lower and upper values are set to 1 too.

    References
    ----------
    .. [Gilleland2010] Gilleland, E. (2010). *Confidence Intervals for Forecast\
    Verification* (No. NCAR/TN-479+STR). University Corporation for Atmospheric\
    Research. doi:10.5065/D6WD3XJM
    """
    # Prototype, might be better to implement this in Cython if it's too slow.
    # Then it would probably be best to make this function only a lookup table
    # which calls a cpdef'd method, which itself calls the cdef'd metric
    # function with a cdef'd bootstrap implementation.
    n = len(x)
    if n < minimum_data_length:
        raise ValueError(
            "Not enough data for bootstrapping. Your data should contain at"
            f" least {minimum_data_length} samples.\n"
            f"You can pass 'minimum_data_length={n}' if you want to do"
            "bootstrapping nevertheless."
        )
    orig_snr, orig_err_std, orig_beta = tcol_metrics(x, y, z, ref_ind=ref_ind)
    bootstrapped_snr = np.empty((nsamples, 3), dtype=float)
    bootstrapped_err_std = np.empty((nsamples, 3), dtype=float)
    bootstrapped_beta = np.empty((nsamples, 3), dtype=float)
    if method == "BCa":
        snr_jk = np.empty((n, 3))
        err_std_jk = np.empty((n, 3))
        beta_jk = np.empty((n, 3))
        mask = np.ones(len(x), dtype=bool)
        for i in range(len(x)):
            mask[i] = False
            snr_jk[i, :], err_std_jk[i, :], beta_jk[i, :] = tcol_metrics(
                x[mask], y[mask], z[mask], ref_ind=ref_ind
            )
            mask[i] = True
    for i in range(nsamples):
        idx = np.random.choice(n, size=n)
        _x, _y, _z = x[idx], y[idx], z[idx]
        (
            bootstrapped_snr[i, :],
            bootstrapped_err_std[i, :],
            bootstrapped_beta[i, :],
        ) = tcol_metrics(_x, _y, _z, ref_ind=ref_ind)

    lower_snr = np.empty(3)
    upper_snr = np.empty(3)
    lower_err_std = np.empty(3)
    upper_err_std = np.empty(3)
    lower_beta = np.empty(3)
    upper_beta = np.empty(3)
    for i in range(3):
        if method == "percentile":
            lower_snr[i], upper_snr[i] = _percentile_bs_ci(
                bootstrapped_snr[:, i], orig_snr[i], alpha
            )
            lower_err_std[i], upper_err_std[i] = _percentile_bs_ci(
                bootstrapped_err_std[:, i], orig_err_std[i], alpha
            )
            if i != ref_ind:
                lower_beta[i], upper_beta[i] = _percentile_bs_ci(
                    bootstrapped_beta[:, i], orig_beta[i], alpha
                )
            else:
                lower_beta[i], upper_beta[i] = 1, 1
        elif method == "basic":
            lower_snr[i], upper_snr[i] = _basic_bs_ci(
                bootstrapped_snr[:, i], orig_snr[i], alpha
            )
            lower_err_std[i], upper_err_std[i] = _basic_bs_ci(
                bootstrapped_err_std[:, i], orig_err_std[i], alpha
            )
            if i != ref_ind:
                lower_beta[i], upper_beta[i] = _basic_bs_ci(
                    bootstrapped_beta[:, i], orig_beta[i], alpha
                )
            else:
                lower_beta[i], upper_beta[i] = 1, 1
        elif method == "BCa":
            lower_snr[i], upper_snr[i] = _BCa_bs_ci(
                bootstrapped_snr[:, i],
                orig_snr[i],
                alpha,
                snr_jk,
            )
            lower_err_std[i], upper_err_std[i] = _BCa_bs_ci(
                bootstrapped_err_std[:, i], orig_err_std[i], alpha, err_std_jk
            )
            if i != ref_ind:
                lower_beta[i], upper_beta[i] = _BCa_bs_ci(
                    bootstrapped_beta[:, i], orig_beta[i], alpha, beta_jk
                )
            else:
                lower_beta[i], upper_beta[i] = 1, 1
    return (
        (orig_snr, lower_snr, upper_snr),
        (orig_err_std, lower_err_std, upper_err_std),
        (orig_beta, lower_beta, upper_beta),
    )


def _jackknife(metric_func, x, y):
    jk = np.empty_like(x)
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        mask[i] = False
        jk[i] = metric_func(x[mask], y[mask])
        mask[i] = True
    return jk


def _percentile_bs_ci(bs_m, m, alpha):
    """Calculates the CI using percentiles"""
    lower = np.nanquantile(bs_m, alpha / 2)
    upper = np.nanquantile(bs_m, 1 - alpha / 2)
    return lower, upper


def _basic_bs_ci(bs_m, m, alpha):
    """Basic bootstrap"""
    lower = 2 * m - np.nanquantile(bs_m, 1 - alpha / 2)
    upper = 2 * m - np.nanquantile(bs_m, alpha / 2)
    return lower, upper


def _BCa_bs_ci(bs_m, m, alpha, jk):
    """BCa bootstrap"""
    # see also here regarding implementation:
    # https://www.erikdrysdale.com/bca_python/
    z_alpha = stats.norm.ppf(alpha)
    z_1_alpha = stats.norm.ppf(1 - alpha)

    # bias correction
    z0 = stats.norm.ppf(np.mean(bs_m <= m))

    # acceleration
    jk_mean = np.mean(jk)
    a = np.sum((jk_mean - jk) ** 3) / (
        6 * (np.sum((jk_mean - jk) ** 2)) ** (1.5)
    )

    # calculate adjusted percentiles
    alpha_lower = stats.norm.cdf(
        z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    )
    alpha_upper = stats.norm.cdf(
        z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
    )
    lower = np.nanquantile(bs_m, alpha_lower)
    upper = np.nanquantile(bs_m, alpha_upper)
    return lower, upper
