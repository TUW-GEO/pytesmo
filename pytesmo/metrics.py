# Copyright (c) 2015, Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology,
#      Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

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

import numpy as np
import scipy.stats as sc_stats


def bias(o, p):
    """
    Difference of the mean values.

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    bias : float
        Bias between observations and predictions.
    """
    return np.mean(o) - np.mean(p)


def aad(o, p):
    """
    Average (=mean) absolute deviation (AAD).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predicitions.

    Returns
    -------
    d : float
        Mean absolute deviation.
    """
    return np.mean(np.abs(o - p))


def mad(o, p):
    """
    Median absolute deviation (MAD).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predicitions.

    Returns
    -------
    d : float
        Median absolute deviation.
    """
    return np.median(np.abs(o - p))


def rmsd(o, p):
    """
    Root-mean-square deviation (RMSD).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation.
    """
    return np.sqrt(RSS(o, p) / len(o))


def nrmsd(o, p):
    """
    Normalized root-mean-square deviation (nRMSD).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    nrmsd : float
        Normalized root-mean-square deviation (nRMSD).
    """
    return rmsd(o, p) / (np.max([o, p]) - np.min([o, p]))


def ubrmsd(o, p):
    """
    Unbiased root-mean-square deviation (uRMSD).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    urmsd : float
        Unbiased root-mean-square deviation (uRMSD).
    """
    return np.sqrt(np.sum(((o - np.mean(o)) - (p - np.mean(p))) ** 2) / len(o))


def mse(o, p):
    """
    Mean square error (MSE) as a decomposition of the RMSD into individual
    error components.

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

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
    mse_corr = 2 * np.std(o, ddof=1) * \
        np.std(p, ddof=1) * (1 - pearsonr(o, p)[0])
    mse_bias = bias(o, p) ** 2
    mse_var = (np.std(o, ddof=1) - np.std(p, ddof=1)) ** 2
    mse = mse_corr + mse_bias + mse_var

    return mse, mse_corr, mse_bias, mse_var


def tcol_error(x, y, z):
    """
    Triple collocation error estimate

    Parameters
    ----------
    x: numpy.ndarray
        1D numpy array to calculate the errors
    y: numpy.ndarray
        1D numpy array to calculate the errors
    z: numpy.ndarray
        1D numpy array to calculate the errors

    Returns
    -------
    e_x : float
        Triple collocation error for x.
    e_y : float
        Triple collocation error for y.
    e_z : float
        Triple collocation error for z.
    """
    e_x = np.sqrt(np.abs(np.mean((x - y) * (x - z))))
    e_y = np.sqrt(np.abs(np.mean((y - x) * (y - z))))
    e_z = np.sqrt(np.abs(np.mean((z - x) * (z - y))))

    return e_x, e_y, e_z


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


def pearsonr(o, p):
    """
    Wrapper for scipy.stats.pearsonr. Calculates a Pearson correlation
    coefficient and the p-value for testing non-correlation.

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

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
    return sc_stats.pearsonr(o, p)


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


def spearmanr(o, p):
    """
    Wrapper for scipy.stats.spearmanr. Calculates a Spearman
    rank-order correlation coefficient and the p-value to
    test for non-correlation.

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

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
    return sc_stats.spearmanr(o, p)


def kendalltau(o, p):
    """
    Wrapper for scipy.stats.kendalltau

    Parameters
    ----------
    o : numpy.array
        Observations.
    p : numpy.array
        Predictions.

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
    return sc_stats.kendalltau(o.tolist(), p.tolist())


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
