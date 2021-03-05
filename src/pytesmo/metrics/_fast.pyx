# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt, fabs

cdef extern from "incbeta.c":
    cdef double incbeta(double, double, double)

ctypedef fused numeric:
    cnp.float32_t
    cnp.float64_t


cpdef bias(numeric [:] x, numeric [:] y):
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
    cdef numeric b = 0
    cdef int i, n = len(x)
    for i in range(n):
        b += x[i] - y[i]
    return b / n


cpdef RSS(numeric [:] x, numeric [:] y):
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
    cdef numeric sum = 0
    cdef int i
    cdef int n = len(x)
    for i in range(n):
        sum += (x[i] - y[i])**2
    return sum


cpdef mse_corr(numeric [:] x, numeric [:] y):
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
    cdef numeric mx = 0, my = 0
    cdef numeric varx = 0, vary = 0, cov = 0
    cdef int i, n = len(x)
    
    # calculate means
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    
    # calculate variances and covariance
    for i in range(n):
        varx += (x[i] - mx)**2
        vary += (y[i] - my)**2
        cov += (x[i] - mx) * (y[i] - my)
    varx /= n
    vary /= n
    cov /= n
    return 2 * sqrt(varx) * sqrt(vary) - 2 * cov


cpdef mse_var(numeric [:] x, numeric [:] y):
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
    cdef numeric mx = 0, my = 0
    cdef numeric varx = 0, vary = 0
    cdef int i, n = len(x)
    
    # calculate means
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    
    # calculate variance
    for i in range(n):
        varx += (x[i] - mx)**2
        vary += (y[i] - my)**2
    varx /= n
    vary /= n
    return (sqrt(varx) - sqrt(vary)) ** 2


cpdef mse_bias(numeric [:] x, numeric [:] y):
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


# Faster implementation of the old `mse`
# mse:
# 1.52 ms ± 138 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# mse_decomposition:
# 48.3 µs ± 8.69 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
# -> 32 times faster!
cpdef mse_decomposition(numeric [:] x, numeric [:] y):
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
    cdef numeric mx = 0, my = 0
    cdef numeric varx = 0, vary = 0, cov = 0
    cdef numeric mse, mse_corr, mse_var, mse_bias
    cdef int i, n = len(x)
    
    # calculate means
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    
    # calculate variances and covariance
    for i in range(n):
        varx += (x[i] - mx)**2
        vary += (y[i] - my)**2
        cov += (x[i] - mx) * (y[i] - my)
    varx /= n
    vary /= n
    cov /= n

    # decompositions
    mse_corr =  2 * sqrt(varx) * sqrt(vary) - 2 * cov
    mse_var = (sqrt(varx) - sqrt(vary)) ** 2
    mse_bias = (mx - my) ** 2
    mse = mse_corr + mse_var + mse_bias
    return mse, mse_corr, mse_bias, mse_var


cpdef _ubrmsd(numeric [:] x, numeric [:] y):
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
    cdef numeric mx = 0, my = 0
    cdef numeric sum = 0
    cdef int i, n = len(x)

    if n == 0:
        return float("nan")
    
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


# This implementation is much faster than the old version with numba:
# old: 76.7 ms ± 1.62 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# new: 117 µs ± 836 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
cpdef rolling_pr_rmsd(numeric [:] timestamps,
                      numeric [:] x,
                      numeric [:] y,
                      numeric window_size,
                      int center,
                      int min_periods):
    """
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
    cdef int i, j, n_ts, n_obs, df, start_idx, end_idx
    cdef cnp.float32_t mx, my, vx, vy, cov, r, t_squared, z, msd
    cdef cnp.float32_t nan = float("nan")

    n_ts = len(timestamps)
    cdef cnp.float32_t [:,:] pr_arr = np.empty((n_ts, 2), dtype=np.float32)
    cdef cnp.float32_t [:] rmsd_arr = np.empty(n_ts, dtype=np.float32)

    start_idx = 0
    end_idx = 0
    for i in range(n_ts):
        
        # find interval
        if center:
            # find new start
            for j in range(start_idx, n_ts):
                start_idx = j
                if timestamps[j] >= timestamps[i] - window_size:
                    break
            # find new end
            if timestamps[n_ts - 1] > timestamps[i] + window_size:
                # we have to check separately whether the last entry is outside the window,
                # otherwise we will always get n_ts - 2 as result
                for j in range(end_idx, n_ts):
                    end_idx = j - 1
                    if timestamps[j] > timestamps[i] + window_size:
                        break
            else:
                end_idx = n_ts - 1
                
        else:
            for j in range(start_idx, n_ts):
                start_idx = j
                if timestamps[j] > timestamps[i] - window_size:
                    break
            end_idx = i
        
        # check if we have enough observations
        n_obs = end_idx - start_idx + 1
        if n_obs == 0 or n_obs < min_periods:
            pr_arr[i, 0] = nan
            pr_arr[i, 1] = nan
        else:
            
            # calculate means
            mx = 0
            my = 0
            for j in range(start_idx, end_idx+1):
                mx += x[j]
                my += y[j]
            mx /= n_obs
            my /= n_obs
            
            # calculate variances and covariance
            vx = 0
            vy = 0
            cov = 0
            for j in range(start_idx, end_idx+1):
                vx += (x[j] - mx)**2
                vy += (y[j] - my)**2
                cov += (x[j] - mx) * (y[j] - my)
            vx /= n_obs
            vy /= n_obs
            cov /= n_obs
    
            r = cov / (sqrt(vx*vy))
            pr_arr[i, 0] = r

            # p-value
            if fabs(r) == 1.0:
                pr_arr[i, 1] = 0.0
            else:
                df = n_obs - 2
                t_squared = r * r * (df / ((1.0 - r) * (1.0 + r)))
                z = min(float(df) / (df + t_squared), 1.0)
                pr_arr[i, 1] = incbeta(0.5*df, 0.5, z)

            # rmsd
            msd = 0
            for j in range(start_idx, end_idx+1):
                msd += (x[j] - y[j])**2
            msd /= n_obs
            rmsd_arr[i] = sqrt(msd)

    return pr_arr, rmsd_arr