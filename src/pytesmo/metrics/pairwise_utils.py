import numpy as np
from scipy import stats

from pytesmo.metrics import pairwise


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
        If no analytical CI function is available.     """
    if isinstance(metric_func, str):
        metric_func = getattr(pairwise, metric_func)
    m = metric_func(x, y)
    name = metric_func.__name__
    if has_analytical_ci(metric_func):
        ci = getattr(pairwise, name + "_ci")(x, y, m, alpha)
    else:
        raise ValueError(f"No analytical CI implemented for {name}.")
    return m, ci[0], ci[1]


def with_bootstrapped_ci(metric_func, x, y, alpha=0.05, method="BCa",
                         nsamples=1000, minimum_data_length=100):
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
    z : np.ndarray, optional
        Third dataset for triplet metric. Default is ``None`` (pairwise
        metric).
    alpha : float, optional
        Confidence level, default is 0.05.
    method : str, optional
        The method to use to calculate confidence intervals. Available methods
        are:

        - "BCa" (default): Bias-corrected and accelerated bootstrap.
        - "percentile": Uses the percentiles of the bootstrapped metric
          distribution.
        - "basic": Uses the percentiles of the differences to the original
          metric value.

        For more info and a comparison of the methods, see [1]_, especially
        Table 6 on page 38.
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
    .. [1] Gilleland, E. (2010). *Confidence Intervals for Forecast
    Verification* (No. NCAR/TN-479+STR). University Corporation for Atmospheric
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
            bootstrapped_metric,
            orig_metric,
            alpha,
            orig_jk
        )
    return orig_metric, lower, upper


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
    lower = np.quantile(bs_m, alpha / 2)
    upper = np.quantile(bs_m, 1 - alpha / 2)
    return lower, upper


def _basic_bs_ci(bs_m, m, alpha):
    """Basic bootstrap"""
    lower = 2 * m - np.quantile(bs_m, 1 - alpha / 2)
    upper = 2 * m - np.quantile(bs_m, alpha / 2)
    return lower, upper


def _BCa_bs_ci(bs_m, m, alpha, jk):
    """BCa bootstrap"""
    # see also here regarding implementation:
    # https://www.erikdrysdale.com/bca_python/
    z_alpha = stats.norm.ppf(alpha)
    z_1_alpha = stats.norm.ppf(1-alpha)

    # bias correction
    z0 = stats.norm.ppf(
        np.mean(bs_m <= m)
    )

    # acceleration
    jk_mean = np.mean(jk)
    a = (
        np.sum((jk_mean - jk)**3)
        / (6 * (np.sum((jk_mean - jk)**2))**(1.5))
    )

    # calculate adjusted percentiles
    alpha_lower = stats.norm.cdf(
        z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    )
    alpha_upper = stats.norm.cdf(
        z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha))
    )
    lower = np.quantile(bs_m, alpha_lower)
    upper = np.quantile(bs_m, alpha_upper)
    return lower, upper
