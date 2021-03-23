import numpy as np

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


def with_bootstrapped_ci(metric_func, x, y, alpha=0.05,
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
    """
    # prototype, probably inefficient
    n = len(x)
    if n < minimum_data_length:
        raise ValueError(
            "Not enough data for bootstrapping. Your data should contain at"
            f" least {minimum_data_length} samples.\n"
            f"You can pass 'minimum_data_length={n}' if you want to do"
            "bootstrapping nevertheless."
        )
    mvals = np.empty(n, dtype=float)
    for i in range(nsamples):
        idx = np.random.choice(n, size=n)
        _x, _y = x[idx], y[idx]
        mvals[i] = metric_func(_x, _y)
    lower = np.quantile(mvals, alpha / 2)
    upper = np.quantile(mvals, 1 - alpha / 2)
    m = metric_func(x, y)
    return m, lower, upper
