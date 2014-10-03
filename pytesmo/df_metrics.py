# Copyright (c) 2013,Vienna University of Technology,
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

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Module contains wrappers for methods in pytesmo.metrics
which can be given pandas.DataFrames
instead of single numpy.arrays . If the DataFrame has more columns
than the function has input parameters
the function will be applied pairwise

Created on Aug 14, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''

import numpy as np
import pytesmo.metrics as metrics
from collections import namedtuple


class DataFrameDimensionError(Exception):
    pass


def bias(df):
    """Bias

    Returns
    -------
    bias : pandas.Dataframe
        of shape (len(df.columns),len(df.columns))
    See Also
    --------
    pytesmo.metrics.bias
    """
    return _to_namedtuple(pairwise_apply(df, metrics.bias), 'bias')


def rmsd(df):
    """Root-mean-square deviation

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.rmsd
    """
    return _to_namedtuple(pairwise_apply(df, metrics.rmsd, comm=True), 'rmsd')


def nrmsd(df):
    """Normalized root-mean-square deviation

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.nrmsd
    """
    return _to_namedtuple(pairwise_apply(df, metrics.nrmsd,
                                         comm=True), 'nrmsd')


def ubrmsd(df):
    """Unbiased root-mean-square deviation

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.ubrmsd
    """
    return _to_namedtuple(pairwise_apply(df, metrics.ubrmsd,
                                         comm=True), 'ubrmsd')


def mse(df):
    """Mean square error (MSE) as a decomposition of the RMSD into
    individual error components

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.mse

    """
    MSE, MSEcorr, MSEbias, MSEvar = pairwise_apply(df, metrics.mse, comm=True)
    return (_to_namedtuple(MSE, 'MSE'),
            _to_namedtuple(MSEcorr, 'MSEcorr'),
            _to_namedtuple(MSEbias, 'MSEbias'),
            _to_namedtuple(MSEvar, 'MSEvar'))


def tcol_error(df):
    """Triple collocation error estimate
    In this case df has to have exactly 3 columns, since triple wise
    application of a function is not yet implemented and
    would probably return a complicated structure

    Returns
    -------
    result : namedtuple
        with column names of df

    See Also
    --------
    pytesmo.metrics.tcol_error
    """

    if len(df.columns) != 3:
        raise DataFrameDimensionError("DataFrame has to have 3 columns")

    tcol_result = namedtuple('triple_collocation_error', df.columns)

    return tcol_result._make(metrics.tcol_error(df.ix[:, 0].values,
                                                df.ix[:, 1].values,
                                                df.ix[:, 2].values))


def nash_sutcliffe(df):
    """Nash Sutcliffe model efficiency coefficient

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.nash_sutcliffe
    """
    return _to_namedtuple(pairwise_apply(df, metrics.nash_sutcliffe,
                                         comm=True), 'Nash_Sutcliffe')


def RSS(df):
    """Redidual sum of squares

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.RSS
    """
    return _to_namedtuple(pairwise_apply(df, metrics.RSS, comm=True), 'RSS')


def pearsonr(df):
    """
    Wrapper for scipy.stats.pearsonr

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.pearsonr
    scipy.stats.pearsonr
    """
    r, p = pairwise_apply(df, metrics.pearsonr, comm=True)
    return _to_namedtuple(r, 'Pearsons_r'), _to_namedtuple(p, 'p_value')


def spearmanr(df):
    """
    Wrapper for scipy.stats.spearmanr

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.spearmenr
    scipy.stats.spearmenr
    """
    r, p = pairwise_apply(df, metrics.spearmanr, comm=True)
    return _to_namedtuple(r, 'Spearman_r'), _to_namedtuple(p, 'p_value')


def kendalltau(df):
    """
    Wrapper for scipy.stats.kendalltau

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.kendalltau
    scipy.stats.kendalltau
    """
    r, p = pairwise_apply(df, metrics.kendalltau, comm=True)
    return _to_namedtuple(r, 'Kendall_tau'), _to_namedtuple(p, 'p_value')


def pairwise_apply(df, method, comm=False):
    """
    Compute given method pairwise for all columns, excluding NA/null values

    Parameters
    ----------
    df : pandas.DataFrame
        input data, method will be applied to each column pair
    method : function
        method to apply to each column pair. has to take 2 input arguments of
        type numpy.array and return one value or tuple of values

    Returns
    -------
    results : pandas.DataFrame
    """
    numeric_df = df._get_numeric_data()
    cols = numeric_df.columns
    mat = numeric_df.values
    mat = mat.T
    applyf = method
    K = len(cols)
    result_empty = np.empty((K, K), dtype=float)
    result_empty.fill(np.nan)

    # find out how many variables the applyf returns
    c = applyf(mat[0], mat[0])
    result = []
    for index, value in enumerate(np.atleast_1d(c)):
        result.append(result_empty)
    result = np.array(result)
    mask = np.isfinite(mat)
    for i, ac in enumerate(mat):
        for j, bc in enumerate(mat):
            if i == j:
                continue
            if comm and np.isfinite(result[0][i, j]):
                continue
            valid = mask[i] & mask[j]
            if not valid.any():
                continue
            if not valid.all():
                c = applyf(ac[valid], bc[valid])
            else:
                c = applyf(ac, bc)

            for index, value in enumerate(np.atleast_1d(c)):
                result[index][i, j] = value
                if comm:
                    result[index][j, i] = value
    return_list = []
    for data in result:
        return_list.append(df._constructor(data, index=cols, columns=cols))

    if len(return_list) == 1:
        return return_list[0]
    else:
        return tuple(return_list)


def _to_namedtuple(df, name):
    """
    takes df produced by pairwise apply and produces named tuple
    of the non duplicate values for commutative operations(the triangle
    above the diagonal)
    """

    names = []
    values = []
    for i, column in enumerate(df.columns[:-1]):
        for column_names in df.columns[i + 1:]:
            names.append('_and_'.join([df.index[i], column_names]))
        values.extend(df[column].values[i + 1:])

    result = namedtuple(name, names)
    return result._make(values)
