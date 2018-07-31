# Copyright (c) 2015, Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology,
#     Department of Geodesy and Geoinformation nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import division

import numpy as np
import scipy.stats as sc_stats
from itertools import permutations,combinations

def bias(x, y):
    """
    Difference of the mean values.
    Sign of output depends on argument order.
    We calculate mean(x) - mean(y).

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
    return np.mean(x) - np.mean(y)


def aad(x, y):
    """
    Average (=mean) absolute deviation (AAD).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    d : float
        Mean absolute deviation.
    """
    return np.mean(np.abs(x - y))


def mad(x, y):
    """
    Median absolute deviation (MAD).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    d : float
        Median absolute deviation.
    """
    return np.median(np.abs(x - y))


def rmsd(x, y, ddof=0):
    """
    Root-mean-square deviation (RMSD). It is implemented for an unbiased
    estimator, which means the RMSD is the square root of the variance, also
    known as the standard error. The delta degree of freedom keyword (ddof) can
    be used to correct for the case the true variance is unknown and estimated
    from the population. Concretely, the naive sample variance estimator sums
    the squared deviations and divides by n, which is biased. Dividing instead
    by n -1 yields an unbiased estimator

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
    rmsd : float
        Root-mean-square deviation.
    """
    return np.sqrt(RSS(x, y) / (len(x) - ddof))


def nrmsd(x, y, ddof=0):
    """
    Normalized root-mean-square deviation (nRMSD).

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
    nrmsd : float
        Normalized root-mean-square deviation (nRMSD).
    """
    return rmsd(x, y, ddof) / (np.max([x, y]) - np.min([x, y]))


def ubrmsd(x, y, ddof=0):
    """
    Unbiased root-mean-square deviation (uRMSD).

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
    urmsd : float
        Unbiased root-mean-square deviation (uRMSD).
    """
    return np.sqrt(np.sum(((x - np.mean(x)) -
                           (y - np.mean(y))) ** 2) / (len(x) - ddof))


def mse(x, y, ddof=0):
    """
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
    mse_corr = 2 * np.std(x, ddof=ddof) * \
        np.std(y, ddof=ddof) * (1 - pearsonr(x, y)[0])
    mse_bias = bias(x, y) ** 2
    mse_var = (np.std(x, ddof=ddof) - np.std(y, ddof=ddof)) ** 2
    mse = mse_corr + mse_bias + mse_var

    return mse, mse_corr, mse_bias, mse_var


def tcol_error(x, y, z):
    """
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
    .. [Scipal2008] Scipal, K., Holmes, T., De Jeu, R., Naeimi, V., & Wagner, W. (2008). A
       possible solution for the problem of estimating the error structure of global
       soil moisture data sets. Geophysical Research Letters, 35(24), .
    """
    e_x = np.sqrt(np.abs(np.mean((x - y) * (x - z))))
    e_y = np.sqrt(np.abs(np.mean((y - x) * (y - z))))
    e_z = np.sqrt(np.abs(np.mean((z - x) * (z - y))))

    return e_x, e_y, e_z


def tcol_snr(x, y, z, ref_ind=0):
    """
    triple collocation based estimation of signal-to-noise ratio, absolute errors,
    and rescaling coefficients

    Parameters
    ----------
    x: 1D numpy.ndarray
        first input dataset
    y: 1D numpy.ndarray
        second input dataset
    z: 1D numpy.ndarray
        third input dataset
    ref_ind: int
        index of reference data set for estimating scaling coefficients. default: 0 (x)

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
    covariances of the dataset. For a general overview and how this function and
    :py:func:`pytesmo.metrics.tcol_error` are related please see [Gruber2015]_.

    Estimation of the error variances from the covariances of the datasets
    (e.g. :math:`\\sigma_{XY}` for the covariance between :math:`x` and
    :math:`y`) is done using the following formula:

    .. math:: \\sigma_{\\varepsilon_x}^2 = \\sigma_{X}^2 - \\frac{\\sigma_{XY}\\sigma_{XZ}}{\\sigma_{YZ}}
    .. math:: \\sigma_{\\varepsilon_y}^2 = \\sigma_{Y}^2 - \\frac{\\sigma_{YX}\\sigma_{YZ}}{\\sigma_{XZ}}
    .. math:: \\sigma_{\\varepsilon_z}^2 = \\sigma_{Z}^2 - \\frac{\\sigma_{ZY}\\sigma_{ZX}}{\\sigma_{YX}}

    :math:`\\beta` can also be estimated from the covariances:

    .. math:: \\beta_x = 1
    .. math:: \\beta_y = \\frac{\\sigma_{XZ}}{\\sigma_{YZ}}
    .. math:: \\beta_z=\\frac{\\sigma_{XY}}{\\sigma_{ZY}}

    The signal to noise ratio (SNR) is also calculated from the variances
    and covariances:

    .. math:: \\text{SNR}_X[dB] = -10\\log\\left(\\frac{\\sigma_{X}^2\\sigma_{YZ}}{\\sigma_{XY}\\sigma_{XZ}}-1\\right)
    .. math:: \\text{SNR}_Y[dB] = -10\\log\\left(\\frac{\\sigma_{Y}^2\\sigma_{XZ}}{\\sigma_{YX}\\sigma_{YZ}}-1\\right)
    .. math:: \\text{SNR}_Z[dB] = -10\\log\\left(\\frac{\\sigma_{Z}^2\\sigma_{XY}}{\\sigma_{ZX}\\sigma_{ZY}}-1\\right)

    It is given in dB to make it symmetric around zero. If the value is zero
    it means that the signal variance and the noise variance are equal. +3dB
    means that the signal variance is twice as high as the noise variance.

    References
    ----------
    .. [Gruber2015] Gruber, A., Su, C., Zwieback, S., Crow, W., Dorigo, W., Wagner, W.
       (2015). Recent advances in (soil moisture) triple collocation analysis.
       International Journal of Applied Earth Observation and Geoinformation,
       in review
    """

    cov = np.cov(np.vstack((x, y, z)))

    ind = (0, 1, 2, 0, 1, 2)
    no_ref_ind = np.where(np.arange(3) != ref_ind)[0]

    snr = 10 * np.log10([((cov[i, i] * cov[ind[i + 1], ind[i + 2]]) /
                          (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) - 1) ** (-1)
                         for i in np.arange(3)])
    err_var = np.array([
        cov[i, i] -
        (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]]
        for i in np.arange(3)])

    beta = np.array([cov[ref_ind, no_ref_ind[no_ref_ind != i][0]] /
                     cov[i, no_ref_ind[no_ref_ind != i][0]] if i != ref_ind
                     else 1 for i in np.arange(3)])

    return snr, np.sqrt(err_var) * beta, beta

def check_if_biased(combs,correlated):
    """
    Supporting function for extended collocation
    Checks whether the estimators are biased by checking of not
    too manny data sets (are assumed to have cross-correlated errors)
    """
    for corr in correlated:
        for comb in combs:
            if (np.array_equal(comb,corr))|(np.array_equal(comb,corr[::-1])):
                return True
    return False

def ecol(data, correlated=None, err_cov=None, abs_est=True):
    """
    Extended collocation analysis to obtain estimates of:
        - signal variances
        - error variances
        - signal-to-noise ratios [dB]
        - error cross-covariances (and -correlations)
    based on an arbitrary number of N>3 data sets.

    !!! EACH DATA SET MUST BE MEMBER OF >= 1 TRIPLET THAT FULFILLS THE CLASSICAL TRIPLE COLLOCATION ASSUMPTIONS !!!

    Parameters
    ----------
    data : pd.DataFrame
        Temporally matched input data sets in each column
    correlated : tuple of tuples (string)
        A tuple containing tuples of data set names (column names), between
        which the error cross-correlation shall be estimated.
        e.g. [['AMSR-E','SMOS'],['GLDAS','ERA']] estimates error cross-correlations
        between (AMSR-E and SMOS), and (GLDAS and ERA), respectively.
    err_cov :
        A priori known error cross-covariances that shall be included
        in the estimation (to obtain unbiased estimates)
    abs_est :
        Force absolute values for signal and error variance estimates
        (to mitiate the issue of estimation uncertainties)

    Returns
    -------
    A dictionary with the following entries (<name> correspond to data set (df column's) names:
    - sig_<name> : signal variance of <name>
    - err_<name> : error variance of <name>
    - snr_<name> : SNR (in dB) of <name>
    - err_cov_<name1>_<name2> : error covariance between <name1> and <name2>
    - err_corr_<name1>_<name2> : error correlation between <name1> and <name2>

    Notes
    -----
    Rescaling parameters can be derived from the signal variances
    e.g., scaling <src> against <ref>:
    beta =  np.sqrt(sig_<ref> / sig_<src>)
    rescaled = (data[<src>] - data[<src>].mean()) * beta + data[<ref>].mean()

    Examples
    --------
    # Just random numbers for demonstrations
    ds1 = np.random.normal(0,1,500)
    ds2 = np.random.normal(0,1,500)
    ds3 = np.random.normal(0,1,500)
    ds4 = np.random.normal(0,1,500)
    ds5 = np.random.normal(0,1,500)

    # Three data sets without cross-correlated errors: This is equivalent
    # to standard triple collocation.
    df = pd.DataFrame({'ds1':ds1,'ds2':ds2,'ds3':ds3},
                      index=np.arange(500))
    res = ecol(df)

    # Five data sets, where data sets (1 and 2), and (3 and 4), are assumed
    # to have cross-correlated errors.
    df = pd.DataFrame({'ds1':ds1,'ds2':ds2,'ds3':ds3,'ds4':ds4,'ds5':ds5},
                      index=np.arange(500),)
    correlated = [['ds1','ds2'],['ds3','ds4']]
    res = ecol(df,correlated=correlated)

    References
    ----------
    .. [Gruber2016] Gruber, A., Su, C. H., Crow, W. T., Zwieback, S., Dorigo, W. A., & Wagner, W. (2016). Estimating error
    cross-correlations in soil moisture data sets using extended collocation analysis. Journal of Geophysical
    Research: Atmospheres, 121(3), 1208-1219.
    """

    data.dropna(inplace=True)

    cols = data.columns.values
    cov = data.cov()

    # subtract a-priori known error covariances to obtained unbiased estimates
    if err_cov is not None:
        cov[err_cov[0]][err_cov[1]] -= err_cov[2]
        cov[err_cov[1]][err_cov[0]] -= err_cov[2]

    # ----- Building up the observation vector y and the design matrix A -----

    # Initial lenght of the parameter vector x:
    # n error variances + n signal variances
    n_x = 2 * len(cols)

    # First n elements in y: variances of all data sets
    y = cov.values.diagonal()

    # Extend y if data sets with correlated errors exist
    if correlated is not None:
        # additionally estimated in x:
        # k error covariances, and k cross-biased signal variances
        # (biased with the respective beta_i*beta_j)
        n_x += 2 * len(correlated)

        # add covariances between the correlated data sets to the y vector
        y = np.hstack((y, [cov[ds[0]][ds[1]] for ds in correlated]))

    # Generate the first part of the design matrix A (total variance = signal variance + error variance)
    A = np.hstack((np.matrix(np.identity(int(n_x / 2))), np.matrix(np.identity(int(n_x / 2))))).astype('int')

    # build up A and y components for estimating signal variances (biased with beta_i**2 only)
    # i.e., the classical TC based signal variance estimators cov(a,c)*cov(a,d)/cov(c,d)
    for col in cols:

        others = cols[cols != col]
        combs = list(combinations(others, 2))

        for comb in combs:
            if correlated is not None:
                if check_if_biased([[col, comb[0]], [col, comb[1]], [comb[0], comb[1]]], correlated):
                    continue

            A_line = np.zeros(n_x).astype('int')
            A_line[np.where(cols == col)[0][0]] = 1
            A = np.vstack((A, A_line))

            y = np.append(y, cov[col][comb[0]] * cov[col][comb[1]] / cov[comb[0]][comb[1]])

    # build up A and y components for the cross-biased signal variabilities (with beta_i*beta_j)
    # i.e., the cross-biased signal variance estimators (cov(a,c)*cov(b,d)/cov(c,d))
    if correlated is not None:
        for i in np.arange(len(correlated)):
            others = cols[(cols != correlated[i][0]) & (cols != correlated[i][1])]
            combs = list(permutations(others, 2))

            for comb in combs:
                if check_if_biased([[correlated[i][0], comb[0]], [correlated[i][1], comb[1]], comb], correlated):
                    continue

                A_line = np.zeros(n_x).astype('int')
                A_line[len(cols) + i] = 1
                A = np.vstack((A, A_line))

                y = np.append(y,
                              cov[correlated[i][0]][comb[0]] * cov[correlated[i][1]][comb[1]] / cov[comb[0]][comb[1]])

    y = np.matrix(y).T

    # ----- Solving for the parameter vector x -----

    x = (A.T * A).I * A.T * y
    x = np.squeeze(np.array(x))

    # ----- Building up the result dictionary -----

    tags = np.hstack(('sig_' + cols, 'err_' + cols, 'snr_' + cols))

    if correlated is not None:

        # remove the cross-biased signal variabilities (with beta_i*beta_j) as they are not useful
        x = np.delete(x, np.arange(len(correlated)) + len(cols))

        # Derive error cross-correlations from error covariances and error variances
        for i in np.arange(len(correlated)):
            x = np.append(x, x[2 * len(cols) + i] / np.sqrt(x[len(cols) + np.where(cols == correlated[i][0])[0][0]] * x[
                len(cols) + np.where(cols == correlated[i][1])[0][0]]))

        # add error covariances and correlations to the result dictionary
        tags = np.append(tags, np.array(['err_cov_' + ds[0] + '_' + ds[1] for ds in correlated]))
        tags = np.append(tags, np.array(['err_corr_' + ds[0] + '_' + ds[1] for ds in correlated]))

    # force absolute signal and error variance estimates to compensate for estimation uncertainties
    if abs_est is True:
        x[0:2 * len(cols)] = np.abs(x[0:2 * len(cols)])

    # calculate and add SNRs (in decibel units)
    x = np.insert(x, 2 * len(cols), 10 * np.log10(x[0:len(cols)] / x[len(cols):2 * len(cols)]))

    return dict(zip(tags, x))


def ecol(data, correlated=None, err_cov=None, abs_est=True):
    """
    Extended collocation analysis to obtain estimates of:
        - signal variances
        - error variances
        - signal-to-noise ratios [dB]
        - error cross-covariances (and -correlations)
    based on an arbitrary number of N>3 data sets.

    !!! EACH DATA SET MUST BE MEMBER OF >= 1 TRIPLET THAT FULFILLS THE CLASSICAL TRIPLE COLLOCATION ASSUMPTIONS !!!

    Parameters
    ----------
    data : pd.DataFrame
        Temporally matched input data sets in each column
    correlated : tuple of tuples (string)
        A tuple containing tuples of data set names (column names), between
        which the error cross-correlation shall be estimated.
        e.g. [['AMSR-E','SMOS'],['GLDAS','ERA']] estimates error cross-correlations
        between (AMSR-E and SMOS), and (GLDAS and ERA), respectively.
    err_cov :
        A priori known error cross-covariances that shall be included
        in the estimation (to obtain unbiased estimates)
    abs_est :
        Force absolute values for signal and error variance estimates
        (to mitiate the issue of estimation uncertainties)

    Returns
    -------
    A dictionary with the following entries (<name> correspond to data set (df column's) names:
    - sig_<name> : signal variance of <name>
    - err_<name> : error variance of <name>
    - snr_<name> : SNR (in dB) of <name>
    - err_cov_<name1>_<name2> : error covariance between <name1> and <name2>
    - err_corr_<name1>_<name2> : error correlation between <name1> and <name2>

    Notes
    -----
    Rescaling parameters can be derived from the signal variances
    e.g., scaling <src> against <ref>:
    beta =  np.sqrt(sig_<ref> / sig_<src>)
    rescaled = (data[<src>] - data[<src>].mean()) * beta + data[<ref>].mean()

    References
    ----------
    .. [Gruber2016] Gruber, A., Su, C. H., Crow, W. T., Zwieback, S., Dorigo, W. A., & Wagner, W. (2016). Estimating error
    cross-correlations in soil moisture data sets using extended collocation analysis. Journal of Geophysical
    Research: Atmospheres, 121(3), 1208-1219.
    """

    data.dropna(inplace=True)

    cols = data.columns.values
    cov = data.cov()

    # subtract a-priori known error covariances to obtained unbiased estimates
    if err_cov is not None:
        cov[err_cov[0]][err_cov[1]] -= err_cov[2]
        cov[err_cov[1]][err_cov[0]] -= err_cov[2]

    # ----- Building up the observation vector y and the design matrix A -----

    # Initial lenght of the parameter vector x:
    # n error variances + n signal variances
    n_x = 2 * len(cols)

    # First n elements in y: variances of all data sets
    y = cov.values.diagonal()

    # Extend y if data sets with correlated errors exist
    if correlated is not None:
        # additionally estimated in x:
        # k error covariances, and k cross-biased signal variances
        # (biased with the respective beta_i*beta_j)
        n_x += 2 * len(correlated)

        # add covariances between the correlated data sets to the y vector
        y = np.hstack((y, [cov[ds[0]][ds[1]] for ds in correlated]))

    # Generate the first part of the design matrix A (total variance = signal variance + error variance)
    A = np.hstack((np.matrix(np.identity(int(n_x / 2))), np.matrix(np.identity(int(n_x / 2))))).astype('int')

    # build up A and y components for estimating signal variances (biased with beta_i**2 only)
    # i.e., the classical TC based signal variance estimators cov(a,c)*cov(a,d)/cov(c,d)
    for col in cols:

        others = cols[cols != col]
        combs = list(combinations(others, 2))

        for comb in combs:
            if correlated is not None:
                if check_if_biased([[col, comb[0]], [col, comb[1]], [comb[0], comb[1]]], correlated):
                    continue

            A_line = np.zeros(n_x).astype('int')
            A_line[np.where(cols == col)[0][0]] = 1
            A = np.vstack((A, A_line))

            y = np.append(y, cov[col][comb[0]] * cov[col][comb[1]] / cov[comb[0]][comb[1]])

    # build up A and y components for the cross-biased signal variabilities (with beta_i*beta_j)
    # i.e., the cross-biased signal variance estimators (cov(a,c)*cov(b,d)/cov(c,d))
    if correlated is not None:
        for i in np.arange(len(correlated)):
            others = cols[(cols != correlated[i][0]) & (cols != correlated[i][1])]
            combs = list(permutations(others, 2))

            for comb in combs:
                if check_if_biased([[correlated[i][0], comb[0]], [correlated[i][1], comb[1]], comb], correlated):
                    continue

                A_line = np.zeros(n_x).astype('int')
                A_line[len(cols) + i] = 1
                A = np.vstack((A, A_line))

                y = np.append(y,
                              cov[correlated[i][0]][comb[0]] * cov[correlated[i][1]][comb[1]] / cov[comb[0]][comb[1]])

    y = np.matrix(y).T

    # ----- Solving for the parameter vector x -----

    x = (A.T * A).I * A.T * y
    x = np.squeeze(np.array(x))

    # ----- Building up the result dictionary -----

    tags = np.hstack(('sig_' + cols, 'err_' + cols, 'snr_' + cols))

    if correlated is not None:

        # remove the cross-biased signal variabilities (with beta_i*beta_j) as they are not useful
        x = np.delete(x, np.arange(len(correlated)) + len(cols))

        # Derive error cross-correlations from error covariances and error variances
        for i in np.arange(len(correlated)):
            x = np.append(x, x[2 * len(cols) + i] / np.sqrt(x[len(cols) + np.where(cols == correlated[i][0])[0][0]] * x[
                len(cols) + np.where(cols == correlated[i][1])[0][0]]))

        # add error covariances and correlations to the result dictionary
        tags = np.append(tags, np.array(['err_cov_' + ds[0] + '_' + ds[1] for ds in correlated]))
        tags = np.append(tags, np.array(['err_corr_' + ds[0] + '_' + ds[1] for ds in correlated]))

    # force absolute signal and error variance estimates to compensate for estimation uncertainties
    if abs_est is True:
        x[0:2 * len(cols)] = np.abs(x[0:2 * len(cols)])

    # calculate and add SNRs (in decibel units)
    x = np.insert(x, 2 * len(cols), 10 * np.log10(x[0:len(cols)] / x[len(cols):2 * len(cols)]))

    return dict(zip(tags, x))


def nash_sutcliffe(o, p):
    """
    Nash Sutcliffe model efficiency coefficient E. The range of E lies between
    1.0 (perfect fit) and -inf.

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    E : float
        Nash Sutcliffe model efficiency coefficient E.
    """
    return 1 - (np.sum((o - p) ** 2)) / (np.sum((o - np.mean(o)) ** 2))


def RSS(o, p):
    """
    Residual sum of squares.

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    res : float
        Residual sum of squares.
    """
    return np.sum((o - p) ** 2)


def pearsonr(x, y):
    """
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
    return sc_stats.pearsonr(x, y)


def pearsonr_recursive(x, y, n_old=0, sum_xi_yi=0,
                       sum_xi=0, sum_yi=0, sum_x2=0,
                       sum_y2=0):
    """
    Calculate pearson correlation in a recursive manner based on

    .. math:: r_{xy} = \\frac{n\\sum x_iy_i-\\sum x_i\\sum y_i} {\\sqrt{n\\sum x_i^2-(\\sum x_i)^2}~\\sqrt{n\\sum y_i^2-(\\sum y_i)^2}}

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
    sum_x2 = sum_x2 + np.sum(x**2)
    sum_y2 = sum_y2 + np.sum(y**2)

    r = ((n_new * sum_xi_yi - sum_xi * sum_yi) /
         (np.sqrt(n_new * sum_x2 - sum_xi**2) *
          np.sqrt(n_new * sum_y2 - sum_yi**2)))

    return r, (n_new, sum_xi_yi, sum_xi, sum_yi, sum_x2, sum_y2)


def pearson_conf(r, n, c=95):
    """
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
    norm_z_value = sc_stats.norm.ppf(n_quant)
    # calculate upper and lower limit for normally distributed z
    z_upper = z + std_err * norm_z_value
    z_lower = z - std_err * norm_z_value
    # inverse fisher transform using the tanh
    return np.tanh(z_lower), np.tanh(z_upper)


def spearmanr(x, y):
    """
    Wrapper for scipy.stats.spearmanr. Calculates a Spearman
    rank-order correlation coefficient and the p-value to
    test for non-correlation.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
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
    return sc_stats.spearmanr(x, y)


def kendalltau(x, y):
    """
    Wrapper for scipy.stats.kendalltau

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
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
    return sc_stats.kendalltau(x.tolist(), y.tolist())


def index_of_agreement(o, p):
    """
    Index of agreement was proposed by Willmot (1981), to overcome the
    insenstivity of Nash-Sutcliffe efficiency E and R^2 to differences in the
    observed and predicted means and variances (Legates and McCabe, 1999).
    The index of agreement represents the ratio of the mean square error and
    the potential error (Willmot, 1984). The potential error in the denominator
    represents the largest value that the squared difference of each pair can
    attain. The range of d is similar to that of R^2 and lies between
    0 (no correlation) and 1 (perfect fit).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    d : float
        Index of agreement.
    """
    denom = np.sum((np.abs(p - np.mean(o)) + np.abs(o - np.mean(o)))**2)
    d = 1 - RSS(o, p) / denom

    return d
