"""
Metrics for pairs and triplets
==============================

This module provides functions to calculate metrics and their associated
confidence intervals.

Pairwise Metrics
----------------

For pairwise metrics, confidence intervals can be calculated analytically (if
available) or via boostrapping. Example with bias::

    from pytesmo.metrics import bias, with_analytical_ci

    x, y = np.random.randn(100), np.rand.randn(100)
    # only metric calculation
    b = bias(x, y)

    # with analytical CI
    b, lower, upper = with_analytical_ci(bias, x, y, alpha=0.05)

    # using bootstrapping instead (uses nsamples=1000 by default)
    b, lower, upper = with_bootstrapped_ci(bias, x, y, alpha=0.05)

Analytical CIs are available for

* bias
* rmsd
* nrmsd
* ubrmsd
* mse
* pearson_r
* spearman_r
* kendall_tau

You can use :py:func:`pytesmo.metrics.has_analytical_ci` to check
programmatically whether analytical CIs are available.

Triple Collocation Metrics
--------------------------

Triple collocation metric can be calculated with the function
:py:func:`pytesmo.metrics.tcol_metrics`. For bootstrapping CIs, use
:py:func:`pytesmo.metrics.tcol_metrics_with_bootstrapped_ci`::

    from pytesmo.metrics import tcol_metrics, tcol_metric

    x, y, z = ...

    # only metrics
    snr, err_std, beta = tcol_metrics(x, y, z)

    # with cis, using nsamples=1000 and alpha=0.05 (default)
    snr_and_ci, err_std_and_ci, beta_and_ci = (
        tcol_metrics_with_bootstrapped_ci(x, y, z)
    )
    snr, lower_snr, upper_snr = snr_and_ci



Developer notes
---------------
Pairwise metrics are implemented in :py:mod:`pytesmo.metrics.pairwise`, triple
collocation metrics in :py:mod:`pytesmo.metrics.tcol`.
"""


from pytesmo.metrics.pairwise import (
    aad,
    bias,
    mad,
    msd,
    rmsd,
    nrmsd,
    ubrmsd,
    mse_bias,
    mse_var,
    mse_corr,
    mse_decomposition,
    RSS,
    pearson_r,
    spearman_r,
    kendall_tau,
    nash_sutcliffe,
    index_of_agreement,
    rolling_pr_rmsd,
)
from pytesmo.metrics.confidence_intervals import (
    has_analytical_ci,
    with_analytical_ci,
    with_bootstrapped_ci,
    tcol_metrics_with_bootstrapped_ci,
)

from pytesmo.metrics.tcol import (
    tcol_metrics,
    ecol
)

from pytesmo.metrics.deprecated import (
    mse,
    tcol_error,
    tcol_snr,
    pearsonr,
    spearmanr,
    kendalltau,
    pearson_conf,
    pearsonr_recursive,
)

__all__ = [
    "bias",
    "mse_bias",
    "mse_var",
    "mse_corr",
    "mse_decomposition",
    "RSS",
    "rolling_pr_rmsd",
    "aad",
    "mad",
    "msd",
    "rmsd",
    "nrmsd",
    "ubrmsd",
    "pearson_r",
    "spearman_r",
    "kendall_tau",
    "index_of_agreement",
    "nash_sutcliffe",
    "has_analytical_ci",
    "with_analytical_ci",
    "with_bootstrapped_ci",
    "tcol_metrics_with_bootstrapped_ci",
    "tcol_metrics",
    "ecol",
]
