# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as cnp
from numpy.math cimport NAN
cimport cython
from cython cimport floating
from libc.math cimport sqrt, fabs
from scipy.special.cython_special import betainc


cpdef _moments_welford_nd(floating [:,:] X):
    """
    Calculates means, variances, and covariance of the given input array using
    Welford's algorithm.

    Parameters
    ----------
    X : numpy.ndarray
        num_samples x num_datasets data array

    Returns
    -------
    mean : numpy.ndarray
        Array of means (length num_datasets)
    cov : numpy.ndarray
        Covariance matrix (num_datasets x num_datasets)
    """
    cdef int i, j, k, n, d
    cdef floating nobs
    cdef floating [:] mean
    cdef floating [:] oldmean
    cdef floating [:,:] C
    n = X.shape[0]
    d = X.shape[1]

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    mean = np.zeros(d, dtype=dtype)
    oldmean = np.zeros(d, dtype=dtype)
    C = np.zeros((d, d), dtype=dtype)

    nobs = 0
    for i in range(n):
        nobs += 1
        for j in range(d):
            oldmean[j] = mean[j]
            mean[j] += (X[i, j] - mean[j]) / nobs
            for k in range(j, d):
                C[j, k] += (X[i, j] - mean[j]) * (X[i, k] - mean[k])

    for j in range(d):
        for k in range(j, d):
            # divide C by nobs to get covariance
            C[j, k] /= nobs
            # populate lower half of C
            if k != j:
                C[k, j] = C[j, k]

    return mean, C
