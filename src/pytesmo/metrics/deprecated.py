"""
Old metrics implementations that have been deprecated.
"""

import numpy as np
from scipy import stats
from scipy.special import betainc

from pytesmo.utils import deprecated
from pytesmo.metrics.pairwise import bias


__all__ = []


@deprecated()
def tcol_error(x, y, z):
    """
    DEPRECATED: Use ``pytesmo.metrics.tcol_metrics`` instead.

    Triple collocation error estimate of three calibrated/scaled
    datasets.

    Parameters
    ----------
    x : numpy.ndarray
        1D numpy array to calculate the errors
    y : numpy.ndarray
        1D numpy array to calculate the errors
    z : numpy.ndarray
        1D numpy array to calculate the errors

    Returns
    -------
    e_x : float
        Triple collocation error for x.
    e_y : float
        Triple collocation error for y.
    e_z : float
        Triple collocation error for z.

    Notes
    -----
    This function estimates the triple collocation error based
    on already scaled/calibrated input data. It follows formula 4
    given in [Scipal2008]_.

    .. math:: \\sigma_{\\varepsilon_x}^2 = \\langle (x-y)(x-z) \\rangle

    .. math:: \\sigma_{\\varepsilon_y}^2 = \\langle (y-x)(y-z) \\rangle

    .. math:: \\sigma_{\\varepsilon_z}^2 = \\langle (z-x)(z-y) \\rangle

    where the :math:`\\langle\\rangle` brackets mean the temporal mean.

    References
    ----------
    .. [Scipal2008] Scipal, K., Holmes, T., De Jeu, R., Naeimi, V., Wagner,
       W. (2008). A possible solution for the problem of estimating the error
       structure of global soil moisture data sets. Geophysical Research
       Letters, 35(24), .
    """
    e_x = np.sqrt(np.abs(np.mean((x - y) * (x - z))))
    e_y = np.sqrt(np.abs(np.mean((y - x) * (y - z))))
    e_z = np.sqrt(np.abs(np.mean((z - x) * (z - y))))

    return e_x, e_y, e_z


@np.errstate(invalid="ignore")
@deprecated()
def tcol_snr(x, y, z, ref_ind=0):
    """
    DEPRECATED: Use the function `tcol_metrics` instead.

    Triple collocation based estimation of signal-to-noise ratio, absolute
    errors, and rescaling coefficients

    Parameters
    ----------
    x: 1D numpy.ndarray
        first input dataset
    y: 1D numpy.ndarray
        second input dataset
    z: 1D numpy.ndarray
        third input dataset
    ref_ind: int
        Index of reference data set for estimating scaling
        coefficients. Default: 0 (x)

    Returns
    -------
    snr: numpy.ndarray
        signal-to-noise (variance) ratio [dB]
    err_std: numpy.ndarray
        **SCALED** error standard deviation
    beta: numpy.ndarray
         scaling coefficients (i_scaled = i * beta_i)

    Notes
    -----

    This function estimates the triple collocation errors, the scaling
    parameter :math:`\\beta` and the signal to noise ratio directly from the
    covariances of the dataset. For a general overview and how this function
    and :py:func:`pytesmo.metrics.tcol_error` are related please see
    [Gruber2015]_.

    Estimation of the error variances from the covariances of the datasets
    (e.g. :math:`\\sigma_{XY}` for the covariance between :math:`x` and
    :math:`y`) is done using the following formula:

    .. math::

       \\sigma_{\\varepsilon_x}^2 =
           \\sigma_{X}^2 - \\frac{\\sigma_{XY}\\sigma_{XZ}}{\\sigma_{YZ}}
    .. math::

       \\sigma_{\\varepsilon_y}^2 =
           \\sigma_{Y}^2 - \\frac{\\sigma_{YX}\\sigma_{YZ}}{\\sigma_{XZ}}
    .. math::

       \\sigma_{\\varepsilon_z}^2 =
           \\sigma_{Z}^2 - \\frac{\\sigma_{ZY}\\sigma_{ZX}}{\\sigma_{YX}}

    :math:`\\beta` can also be estimated from the covariances:

    .. math:: \\beta_x = 1
    .. math:: \\beta_y = \\frac{\\sigma_{XZ}}{\\sigma_{YZ}}
    .. math:: \\beta_z=\\frac{\\sigma_{XY}}{\\sigma_{ZY}}

    The signal to noise ratio (SNR) is also calculated from the variances
    and covariances:

    .. math::

       \\text{SNR}_X[dB] = -10\\log\\left(\\frac{\\sigma_{X}^2\\sigma_{YZ}}
                                         {\\sigma_{XY}\\sigma_{XZ}}-1\\right)
    .. math::

       \\text{SNR}_Y[dB] = -10\\log\\left(\\frac{\\sigma_{Y}^2\\sigma_{XZ}}
                                         {\\sigma_{YX}\\sigma_{YZ}}-1\\right)
    .. math::

       \\text{SNR}_Z[dB] = -10\\log\\left(\\frac{\\sigma_{Z}^2\\sigma_{XY}}
                                         {\\sigma_{ZX}\\sigma_{ZY}}-1\\right)

    It is given in dB to make it symmetric around zero. If the value is zero
    it means that the signal variance and the noise variance are equal. +3dB
    means that the signal variance is twice as high as the noise variance.

    References
    ----------
    .. [Gruber2015] Gruber, A., Su, C., Zwieback, S., Crow, W., Dorigo, W.,
       Wagner, W.  (2015). Recent advances in (soil moisture) triple
       collocation analysis.  International Journal of Applied Earth
       Observation and Geoinformation, in review
    """

    cov = np.cov(np.vstack((x, y, z)))

    ind = (0, 1, 2, 0, 1, 2)
    no_ref_ind = np.where(np.arange(3) != ref_ind)[0]

    snr = 10 * np.log10(
        [
            (
                (cov[i, i] * cov[ind[i + 1], ind[i + 2]])
                / (cov[i, ind[i + 1]] * cov[i, ind[i + 2]])
                - 1
            )
            ** (-1)
            for i in np.arange(3)
        ]
    )
    err_var = np.array(
        [
            cov[i, i]
            - (cov[i, ind[i + 1]] * cov[i, ind[i + 2]])
            / cov[ind[i + 1], ind[i + 2]]
            for i in np.arange(3)
        ]
    )

    beta = np.array(
        [
            cov[ref_ind, no_ref_ind[no_ref_ind != i][0]]
            / cov[i, no_ref_ind[no_ref_ind != i][0]]
            if i != ref_ind
            else 1
            for i in np.arange(3)
        ]
    )

    return snr, np.sqrt(err_var) * beta, beta


@deprecated()
def mse(x, y, ddof=0):
    """
    DEPRECATED: use `msd`, `msd_corr`, `msd_bias`, or `msd_var` for the
    individual components or `pytesmo.metrics.msd_decomposition` for the full
    decomposition instead.

    Mean square error (MSE) as a decomposition of the RMSD into individual
    error components. The MSE is the second moment (about the origin) of the
    error, and thus incorporates both the variance of the estimator and
    its bias. For an unbiased estimator, the MSE is the variance of the
    estimator. Like the variance, MSE has the same units of measurement as
    the square of the quantity being estimated.
    The delta degree of freedom keyword (ddof) can be used to correct for
    the case the true variance is unknown and estimated from the population.
    Concretely, the naive sample variance estimator sums the squared deviations
    and divides by n, which is biased. Dividing instead by n - 1 yields an
    unbiased estimator.
    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.
    Returns
    -------
    mse : float
        Mean square error (MSE).
    mse_corr : float
        Correlation component of MSE.
    mse_bias : float
        Bias component of the MSE.
    mse_var : float
        Variance component of the MSE.
    """
    mse_corr = (
        2
        * np.std(x, ddof=ddof)
        * np.std(y, ddof=ddof)
        * (1 - pearsonr(x, y)[0])
    )
    mse_bias = bias(x, y) ** 2
    mse_var = (np.std(x, ddof=ddof) - np.std(y, ddof=ddof)) ** 2
    mse = mse_corr + mse_bias + mse_var
    return mse, mse_corr, mse_bias, mse_var


@deprecated()
def pearsonr(x, y):
    """
    DEPRECATED: use :func:`pytesmo.metrics.pearson_r` instead if you want
    only the value or the confidence interval, and
    :func:`scipy.stats.pearsonr` if you want the p-value.

    Wrapper for scipy.stats.pearsonr. Calculates a Pearson correlation
    coefficient and the p-value for testing non-correlation.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.
    Returns
    -------
    r : float
        Pearson's correlation coefficent.
    p-value : float
        2 tailed p-value.
    See Also
    --------
    scipy.stats.pearsonr
    """
    return stats.pearsonr(x, y)


@np.errstate(invalid="ignore")
@deprecated()
def pearsonr_recursive(
    x, y, n_old=0, sum_xi_yi=0, sum_xi=0, sum_yi=0, sum_x2=0, sum_y2=0
):
    """
    DEPRECATED: use :func:`pytesmo.metrics.pearson_r` instead if you want
    only the value or the confidence interval, and
    :func:`scipy.stats.pearsonr` if you want the p-value.

    Calculate pearson correlation in a recursive manner based on
    .. math::

       r_{xy} = \\frac{n\\sum x_iy_i-\\sum x_i\\sum y_i}
                      {\\sqrt{n\\sum x_i^2-(\\sum x_i)^2}~
                       \\sqrt{n\\sum y_i^2-(\\sum y_i)^2}}
    Parameters
    ----------
    x: numpy.ndarray
        New values for x
    y: numpy.ndarray
        New values for y
    n_old: float, optional
        number of observations from previous pass
    sum_xi_yi: float, optional
        .. math:: \\sum x_iy_i
        from previous pass
    sum_xi: float, optional
        .. math:: \\sum x_i
        from previous pass
    sum_yi: float, optional
        .. math:: \\sum y_i
        from previous pass
    sum_x2: float, optional
        .. math:: \\sum x_i^2
        from previous pass
    sum_y2: float, optional
        .. math:: \\sum y_i^2
        from previous pass
    Returns
    -------
    r: float
        Pearson correlation coefficient
    params: tuple
       tuple of (n_new, sum_xi_yi, sum_xi, sum_yi, sum_x2, sum_y2) .
       Can be used when calling the next iteration as ``*params``.
    """
    n_new = n_old + len(x)
    sum_xi_yi = sum_xi_yi + np.sum(np.multiply(x, y))
    sum_xi = sum_xi + np.sum(x)
    sum_yi = sum_yi + np.sum(y)
    sum_x2 = sum_x2 + np.sum(x ** 2)
    sum_y2 = sum_y2 + np.sum(y ** 2)

    r = (n_new * sum_xi_yi - sum_xi * sum_yi) / (
        np.sqrt(n_new * sum_x2 - sum_xi ** 2)
        * np.sqrt(n_new * sum_y2 - sum_yi ** 2)
    )

    return r, (n_new, sum_xi_yi, sum_xi, sum_yi, sum_x2, sum_y2)


@deprecated()
def pearson_conf(r, n, c=95):
    """
    DEPRECATED: use :func:`pytesmo.metrics.pearson_r` instead.

    Calcalates the confidence interval of a given pearson
    correlation coefficient using a fisher z-transform,
    only valid for correlation coefficients calculated from
    a bivariate normal distribution

    Parameters
    ----------
    r : float or numpy.ndarray
        Correlation coefficient
    n : int or numpy.ndarray
        Number of observations used in determining the correlation coefficient
    c : float
        Level of confidence in percent, from 0-100.
    Returns
    -------
    r_lower : float or numpy.ndarray
        Lower confidence boundary.
    r_upper : float or numpy.ndarray
        Upper confidence boundary.
    """
    # fisher z transform using the arctanh
    z = np.arctanh(r)
    # calculate the standard error
    std_err = 1 / np.sqrt(n - 3)
    # calculate the quantile of the normal distribution
    # for the given confidence level
    n_quant = 1 - (1 - c / 100.0) / 2.0
    norm_z_value = stats.norm.ppf(n_quant)
    # calculate upper and lower limit for normally distributed z
    z_upper = z + std_err * norm_z_value
    z_lower = z - std_err * norm_z_value
    # inverse fisher transform using the tanh
    return np.tanh(z_lower), np.tanh(z_upper)


@deprecated()
def spearmanr(x, y):
    """
    DEPRECATED: use :func:`pytesmo.metrics.spearman_r` instead if you
    want only the value or the confidence interval, and
    :func:`scipy.stats.spearmanr` if you want the p-value.

    Wrapper for scipy.stats.spearmanr. Calculates a Spearman
    rank-order correlation coefficient and the p-value to
    test for non-correlation.

    Parameters
    ----------
    x : numpy.array
        First input vector.
    y : numpy.array
        Second input vector.
    Returns
    -------
    rho : float
        Spearman correlation coefficient
    p-value : float
        The two-sided p-value for a hypothesis test whose null hypothesis
        is that two sets of data are uncorrelated
    See Also
    --------
    scipy.stats.spearmenr
    """
    return stats.spearmanr(x, y)


@deprecated()
def kendalltau(x, y):
    """
    DEPRECATED: use :func:`pytesmo.metrics.kendall_tau` instead if you want
    only the value or the confidence interval, and
    :func:`scipy.stats.kendalltau` if you want the p-value.

    Wrapper for scipy.stats.kendalltau

    Parameters
    ----------
    x : numpy.array
        First input vector.
    y : numpy.array
        Second input vector.
    Returns
    -------
    Kendall's tau : float
        The tau statistic
    p-value : float
        The two-sided p-palue for a hypothesis test whose null hypothesis
        is an absence of association, tau = 0.
    See Also
    --------
    scipy.stats.kendalltau
    """
    return stats.kendalltau(x, y)


def rolling_pr_rmsd(timestamps, data, window_size, center, min_periods):
    """
    DEPRECATED: use the faster version in metrics._fast instead! Only here for
    testing.

    Computation of rolling Pearson R.

    Parameters
    ----------
    timestamps : float64
        Time stamps as julian dates.
    data : numpy.ndarray
        Time series data in 2d array.
    window_size : float
        Window size in fraction of days.
    center : bool
        Set window at the center.
    min_periods : int
        Minimum number of observations in window required for computation.

    Results
    -------
    pr_arr : numpy.array
        Pearson R and p-value.
    """
    pr_arr = np.empty((timestamps.size, 2), dtype=np.float32)
    rmsd_arr = np.empty(timestamps.size, dtype=np.float32)
    ddof = 0

    for i in range(timestamps.size):
        time_diff = timestamps - timestamps[i]

        if center:
            inside_window = np.abs(time_diff) <= window_size
        else:
            inside_window = (time_diff <= 0) & (time_diff > -window_size)

        idx = np.nonzero(inside_window)[0]
        n_obs = inside_window.sum()

        if n_obs == 0 or n_obs < min_periods:
            pr_arr[i, :] = np.nan
        else:
            sub1 = data[idx[0]: idx[-1] + 1, 0]
            sub2 = data[idx[0]: idx[-1] + 1, 1]

            # pearson r
            pr_arr[i, 0] = np.corrcoef(sub1, sub2)[0, 1]

            # p-value
            if np.abs(pr_arr[i, 0]) == 1.0:
                pr_arr[i, 1] = 0.0
            else:
                df = n_obs - 2.0
                t_squared = (
                    pr_arr[i, 0]
                    * pr_arr[i, 0]
                    * (df / ((1.0 - pr_arr[i, 0]) * (1.0 + pr_arr[i, 0])))
                )
                x = df / (df + t_squared)
                x = np.ma.where(x < 1.0, x, 1.0)
                pr_arr[i, 1] = betainc(0.5 * df, 0.5, x)

            # rmsd
            rmsd_arr[i] = np.sqrt(
                np.sum((sub1 - sub2) ** 2) / (sub1.size - ddof)
            )

    return pr_arr, rmsd_arr
