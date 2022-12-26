# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as cnp
from numpy.math cimport NAN
cimport cython
from cython cimport floating
from libc.math cimport sqrt, fabs
from scipy.special.cython_special import betainc


cpdef _moments_welford(floating [:] x, floating [:] y):
    """
    Calculates means, variances, and covariance of the given input array using
    Welford's algorithm.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    mx, my, varx, vary, cov : floating
    """
    cdef int i, n
    cdef floating mx, my, mxold, myold, M2x, M2y, C
    cdef floating nobs
    n = len(x)

    mx = my = M2x = M2y = C = 0
    nobs = 0
    for i in range(n):
        nobs += 1
        mxold = mx
        myold = my
        mx += (x[i] - mx) / nobs
        my += (y[i] - my) / nobs
        M2x += (x[i] - mx) * (x[i] - mxold)
        M2y += (y[i] - my) * (y[i] - myold)
        C += (x[i] - mx) * (y[i] - myold)
    return mx, my, M2x/n, M2y/n, C/n




cpdef bias(floating [:] x, floating [:] y):
    """
    Difference of the mean values.

    Sign of output depends on argument order. We calculate mean(x) - mean(y).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    bias : float
        Bias between x and y.
    """
    cdef floating b = 0
    cdef int i, n = len(x)
    for i in range(n):
        b += x[i] - y[i]
    return b / n


cpdef RSS(floating [:] x, floating [:] y):
    """
    Residual sum of squares.

    Parameters
    ----------
    x : numpy.ndarray
        Observations.
    y : numpy.ndarray
        Predictions.

    Returns
    -------
    res : float
        Residual sum of squares.
    """
    cdef floating sum = 0
    cdef int i
    cdef int n = len(x)
    for i in range(n):
        sum += (x[i] - y[i])**2
    return sum


cpdef mse_corr(floating [:] x, floating [:] y):
    r"""
    Correlation component of MSE.

    MSE can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::

        MSE &= MSE_{corr} + MSE_{var} + MSE_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the term :math:`MSE_{corr} =
    2\sigma_x\sigma_y(1-r)`.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    mse_corr : float
        Correlation component of MSE.
    """
    cdef floating mx, my, varx, vary, cov
    mx, my, varx, vary, cov = _moments_welford(x, y)
    return _mse_corr_from_moments(mx, my, varx, vary, cov)


cpdef _mse_corr_from_moments(
    floating mx, floating my, floating vx, floating vy, floating cov
):
    return max(2 * sqrt(vx) * sqrt(vy) - 2 * cov, 0)



cpdef mse_var(floating [:] x, floating [:] y):
    r"""
    Variance component of MSE.

    MSE can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::

        MSE &= MSE_{corr} + MSE_{var} + MSE_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the term :math:`MSE_{var} = (\sigma_x -
    \sigma_y)^2`.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    mse_var : float
        Variance component of MSE.
    """
    cdef floating mx, my, varx, vary, cov
    mx, my, varx, vary, cov = _moments_welford(x, y)
    return _mse_var_from_moments(mx, my, varx, vary, cov)


cpdef _mse_var_from_moments(
    floating mx, floating my, floating vx, floating vy, floating cov
):
    return (sqrt(vx) - sqrt(vy)) ** 2


cpdef mse_bias(floating [:] x, floating [:] y):
    r"""
    Bias component of MSE.

    MSE can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::

        MSE &= MSE_{corr} + MSE_{var} + MSE_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the term :math:`MSE_{bias} = (\mu_x - \mu_y)^2`.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    mse_bias : float
        Bias component of MSE.
    """
    return bias(x, y) ** 2


cpdef _mse_bias_from_moments(
    floating mx, floating my, floating vx, floating vy, floating cov
):
    return (mx - my) ** 2


# Faster implementation of the old `mse`
# mse:
# 1.52 ms ± 138 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# mse_decomposition:
# 48.3 µs ± 8.69 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
# -> 32 times faster!
cpdef mse_decomposition(floating [:] x, floating [:] y):
    r"""
    Mean square deviation/mean square error.

    For validation, MSE (same as MSE) is defined as

    ..math::

        MSE = \frac{1}{n}\sum\limits_{i=1}^n (x_i - y_i)^2

    MSE can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::

        MSE &= MSE_{corr} + MSE_{var} + MSE_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the all components as well as the sum.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector
    y : numpy.ndarray
        Second input vector

    Returns
    -------
    mse : float
        Mean square deviation
    mse_corr : float
        Correlation component of MSE.
    mse_bias : float
        Bias component of the MSE.
    mse_var : float
        Variance component of the MSE.
    """
    cdef floating mx, my, varx, vary, cov
    mx, my, varx, vary, cov = _moments_welford(x, y)

    # decompositions
    mse_corr =  _mse_corr_from_moments(mx, my, varx, vary, cov)
    mse_var =  _mse_var_from_moments(mx, my, varx, vary, cov)
    mse_bias =  _mse_bias_from_moments(mx, my, varx, vary, cov)
    mse = mse_corr + mse_var + mse_bias
    return mse, mse_corr, mse_bias, mse_var


cpdef _ubrmsd(floating [:] x, floating [:] y):
    r"""
    Unbiased root-mean-square deviation (uRMSD).

    This corresponds to RMSD with mean biases removed beforehand, that is

    ..math::

        ubRMSD = \sqrt{\frac{1}{n}\sum\limits_{i=1}^n
                           \left((x - \bar{x}) - (y - \bar{y}))^2}

    NOTE: If you are scaling the data beforehand to have zero mean bias, this
    is exactly the same as RMSD.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    ubrmsd : float
        Unbiased root-mean-square deviation (uRMSD).
    """
    cdef floating mx = 0, my = 0
    cdef floating sum = 0
    cdef int i, n = len(x)

    if n == 0:
        return NAN

    # calculate means
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n

    # get unbiased values
    for i in range(n):
        sum += ((x[i] - mx) - (y[i] - my))**2
    return sqrt(sum / n)


cpdef _pearsonr_from_moments(floating varx, floating vary, floating cov, int n):
    cdef floating R, p_R, t_squared, df, z

    # not defined in this case
    if varx == 0 or vary == 0:
        return np.nan, np.nan

    # due to rounding errors, values slightly above 1 are possible
    R = max(-1, min(cov / sqrt(varx * vary), 1))

    # p-value for R
    if fabs(R) == 1.0:
        p_R = 0.0
    else:
        df = n - 2
        t_squared = R * R * (df / ((1.0 - R) * (1.0 + R)))
        z = min(float(df) / (df + t_squared), 1.0)
        p_R = betainc(0.5*df, 0.5, z)
    return R, p_R


# This implementation is much faster than the old version with numba:
# old: 76.7 ms ± 1.62 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# new: 117 µs ± 836 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
cpdef rolling_pr_rmsd(double [:] timestamps,
                      floating [:] x,
                      floating [:] y,
                      double window_size,
                      int center,
                      int min_periods):
    """
    Computation of rolling Pearson R and RMSD.

    Parameters
    ----------
    timestamps : float64
        Time stamps as julian dates.
    data : numpy.ndarray
        Time series data in 2d array.
    window_size : float64
        Window size in fraction of days.
    center : bool
        Set window at the center and include window_size in both directions.
    min_periods : int
        Minimum number of observations in window required for computation.

    Results
    -------
    pr_arr : numpy.array
        Rolling Pearson R and p-value.
    rmsd_arr : numpy.array
        Rolling RMSD
    """
    # This uses an adapted Welford's algorithm to calculate the rolling mean,
    # variance, covariance, and mean squared difference.
    cdef int i, j, n_ts, num_obs, lower, upper, lold, uold
    cdef floating mx, my, mxold, myold, M2x, M2y, C, msd,
    cdef floating rolling_nobs  # a float because we divide by it
    cdef cnp.ndarray[floating, ndim=2] pr_arr
    cdef cnp.ndarray[floating, ndim=1] rmsd_arr
    cdef floating [:,:] pr_view
    cdef floating [:] rmsd_view

    n_ts = len(timestamps)
    # allocate numpy arrays of the correct dtype for returning
    if floating is float:
        pr_arr = np.empty((n_ts, 2), dtype=np.float32)
        rmsd_arr = np.empty(n_ts, dtype=np.float32)
    elif floating is double:
        pr_arr = np.empty((n_ts, 2), dtype=np.float64)
        rmsd_arr = np.empty(n_ts, dtype=np.float64)
    else:
        raise ValueError("Unkown floating type")
    # work on memoryviews instead of on arrays directly
    pr_view = pr_arr
    rmsd_view = rmsd_arr

    mx = my = msd = M2x = M2y = C = 0
    lower = 0
    upper = -1
    rolling_nobs = 0
    for i in range(n_ts):
        lold = lower
        uold = upper


        # find interval
        if center:
            # find new start
            for j in range(lower, n_ts):
                lower = j
                if timestamps[j] >= timestamps[i] - window_size:
                    break

            # find new end:
            # we have to check separately whether the last entry is outside
            # the window, otherwise we will always get n_ts - 2 as result

            # if we're not at the end yet
            if timestamps[n_ts - 1] > timestamps[i] + window_size:
                if i == 0:
                    # in the first iteration upper was -1 in order to set uold
                    # correctly, but in the following upper must be at least 1,
                    # so we don't access an invalid index
                    upper = 1
                for j in range(upper, n_ts):
                    # j - 1 because we want the index that is still inside the
                    # window
                    upper = j - 1
                    if timestamps[j] > timestamps[i] + window_size:
                        break
            else:
                upper = n_ts - 1

        else:
            for j in range(lower, n_ts):
                lower = j
                if timestamps[j] > timestamps[i] - window_size:
                    break
            upper = i



        # Regardless of the number of observations we calculate the moments
        # here, because we need to update them at one point anyway

        # first, add the new terms with Welford's algorithm
        for j in range(uold+1, upper+1):
            mxold = mx
            myold = my
            rolling_nobs += 1
            mx += (x[j] - mx) / rolling_nobs
            my += (y[j] - my) / rolling_nobs
            msd += ((x[j] - y[j])**2 - msd) / rolling_nobs
            M2x += (x[j] - mx) * (x[j] - mxold)
            M2y += (y[j] - my) * (y[j] - myold)
            C += (x[j] - mx) * (y[j] - myold)

        # now subtract the ones that fell out the window
        # the old values here correspond to the m_n values in the formula,
        # that's why the order is different here
        for j in range(lold, lower):
            mxold = mx
            myold = my
            rolling_nobs -= 1
            mx -= (x[j] - mx) / rolling_nobs
            my -= (y[j] - my) / rolling_nobs
            msd -= ((x[j] - y[j])**2 - msd) / rolling_nobs
            M2x -= (x[j] - mxold) * (x[j] - mx)
            M2y -= (y[j] - myold) * (y[j] - my)
            C -= (x[j] - mxold) * (y[j] - my)

        # check if we have enough observations
        num_obs = upper - lower + 1
        if num_obs == 0 or num_obs < min_periods:
            pr_view[i, 0] = NAN
            pr_view[i, 1] = NAN
            rmsd_view[i] = NAN
        else:
            # to get var and cov we would need to divide by n, but since
            # we're only interested in the ratio that's not necessary
            pr_view[i, 0], pr_view[i, 1] = _pearsonr_from_moments(
                M2x, M2y, C, num_obs
            )
            rmsd_view[i] = sqrt(msd)

    return pr_arr, rmsd_arr
