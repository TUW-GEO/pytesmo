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

"""
Module contains wrappers for methods in pytesmo.metrics which can be given
pandas.DataFrames instead of single numpy.arrays.
If the DataFrame has more columns than the function has input parameters
the function will be applied pairwise, resp. to triples.

Created on Aug 14, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
"""

import numpy as np
import pytesmo.metrics as metrics
from collections import namedtuple, OrderedDict
import itertools
import pandas as pd

class DataFrameDimensionError(Exception):
    pass

def n_combinations(iterable, n, must_include=None, with_replacement=False):
    """
    Create possible combinations of an input iterable.

    Parameters
    ---------
    iterable: iterable
        Elements from this iterable are combined.
    n : int
        Number of elements per combination.
    must_include : list, optional (default: None)
        One or more element(s) of iterable that MUST be in each combination.
    with_replacement : bool, optional (default: False)
        Add combinations of elements with itself.
        A,B -> AA, AB, BB

    Returns:
    ---------
    combs: iterable
        The possible combinations of n elements.
    """
    if with_replacement:
        combs = list(itertools.combinations_with_replacement(iterable, n))
    else:
        combs = list(itertools.combinations(iterable, n))
    if must_include:
        combs_filtered = []
        for comb in combs:
            if all([i in comb for i in must_include]):
                combs_filtered.append(comb)
        combs = combs_filtered
    return combs

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
    """
    Triple collocation error estimate, applied to triples of columns of the
    passed data frame.

    Returns
    -------
    triple_collocation_error_x : namedtuple
        Error for the first dataset
    triple_collocation_error_y : namedtuple
        Error for the second dataset
    triple_collocation_error_z : namedtuple
        Error for the third dataset

    See Also
    --------
    pytesmo.metrics.tcol_error
    """
    err0, err1, err2 = nwise_apply(df, metrics.tcol_error, comm=True, n=3)
    trips = list(err0.keys()) # triples in all err are equal
    assert trips == list(err0.keys()) == list(err1.keys()) == list(err2.keys())

    #Tcol_result = namedtuple('triple_collocation_error', ['_and_'.join(trip) for trip in trips])
    errors = []
    for trip in trips:
        #inner_name = '_and_'.join(trip)
        res = [err0[trip], err1[trip], err2[trip]]
        Inner = namedtuple('triple_collocation_error', dict(zip(trip, res)))
        errors.append(Inner(*res))

    return tuple(errors)


def tcol_snr(df, ref_ind=0):
    """
    Triple Collocation based SNR estimation.
    The first column in df is the scaling reference.

    Parameters
    ----------
    df : pd.DataFrame
        Contains the input values as time series in the df columns
    ref_ind : int, optional (default: 0)
        The index of the column in df that contains the reference data set.

    Returns
    -------
    snr : namedtuple
        signal-to-noise (variance) ratio [dB] from the named columns.
    err_std_dev : namedtuple
        **SCALED** error standard deviation from the named columns
    beta : namedtuple
        Scaling coefficients (i_scaled = i * beta_i)
    """
    snr, err, beta = nwise_apply(df, metrics.tcol_snr, comm=True, n=3, ref_ind=ref_ind)

    results = {}
    for var_name, var_vals in {'snr': snr, 'err' : err, 'beta' : beta}.items():
        results[var_name] = []
        for trip, res in var_vals.items():
            inner_name = '_and_'.join(trip)
            Inner = namedtuple(inner_name, dict(zip(trip, res)))
            r = Inner(*res)
            results[var_name].append(r)

    return (results['snr'], results['err'], results['beta'])





def old_tcol_error(df):
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

def nwise_apply(df, method, comm=False, n=2, as_df=False, ds_names=True, **kwargs):
    """
    Compute given method pairwise for all columns, excluding NA/null values

    Parameters
    ----------
    df : pd.DataFrame
        input data, method will be applied to each column pair
    method : function
        method to apply to each column pair. Has to take 2 input arguments of
        type numpy.array and return one value or tuple of values
    comm : bool, optional (default: False)
        #todo: what is this for?
    n : int, optional (default: 2)
        Number of datasets that are combined. e.g. n=2 is the same as the
        pairwise apply function.
    as_df : bool, optional (default: False)
        Return matrix structure, same as for pairwise_apply, only available for
        n=2. By default, the return value will be a list of ordered dicts.
    ds_names : bool, optional (default: True)
        Use the column names of df to identify the dataset instead of using their
        index.
    kwargs :
        Keyword arguments that are passed to method.

    Returns
    -------
    results : pd.DataFrame
    """

    numeric_df = df._get_numeric_data()
    cols = numeric_df.columns.values
    mat = numeric_df.values
    mat = mat.T
    applyf = method

    # find out how many variables the applyf returns
    result = []
    # apply the method using the first data set to find out the shape of c,
    # we add a bias (i) to avoid raining warnings.
    c = applyf(*[mat[i] for i in range(n)])
    for index, value in enumerate(np.atleast_1d(c)):
        result.append(OrderedDict())
    result = np.array(result)    # array of OrderedDicts
    # each return value result is a dict that gets filled with dicts that have
    # the cols and keys and the results as values

    mask = np.isfinite(mat)

    # create the possible combinations of lines
    counter = list(range(mat.shape[0])) # get the number of lines?
    combs = n_combinations(counter, n) # ALL possible combinations of lines?

    lut_comb_cols = dict()

    for comb in combs:
        valid = np.logical_and(*[mask[i] for i in comb]) # where all are True
        if not valid.any():
            continue
        if not valid.all():
            c = applyf(*[mat[i,:][valid] for i in comb], **kwargs)
        else:
            c = applyf(*[mat[i,:] for i in comb], **kwargs)

        lut_comb_cols.update(dict(zip(comb, tuple(cols[[*comb]]))))


        for index, value in enumerate(np.atleast_1d(c)):
            result[index][comb] = value
            if comm:
                result[index][comb] = value

    if as_df:
        # this replicates the old resuls form of the _pairwise_apply()
        if n != 2:
            raise ValueError('Array result only available for n=2')
        else:
            if not ds_names:
                lut_comb_cols = None
            result = [_to_df(r, lut_comb_cols, fill_upper=False) for r in result]
    else:
        if ds_names:
            formatted_results = []
            for r in result:
                formatted = OrderedDict()
                for k, v in r.items():
                    formatted[tuple([lut_comb_cols[i] for i in k])] = v
                formatted_results.append(formatted)
            result = formatted_results

        if len(result) == 1:
            result = result[0]
        else:
            result = tuple(result)

    return result

def _to_df(result, lut_names=None, fill_upper=False):
    # copy the upper diagonal elements to the lower diagonal
    imax = max([max(r) for r in list(result.keys())])
    res = np.full((imax+1, imax+1), np.nan)
    for k, v in result.items():
        res[k[::-1]] = v # create a lower diagonal matrix

    if fill_upper:
        i_upper = np.triu_indices(res.shape[0], 1)
        i_lower = np.tril_indices(res.shape[0], -1)
        res[i_lower] = res[i_upper]

    if lut_names is not None:
        res = pd.DataFrame(data={lut_names[i]: res[:, i] for i in list(range(max(res.shape)))})
    else:
        res = pd.DataFrame(data={i : res[:, i] for i in list(range(max(res.shape)))})
    res.index = res.columns
    return res

def _dict_to_namedtuple(res_dict, name):
    """
    Takes the OrderedDictionary produced by nwise_apply and produces named
    tuples, using the dictionary keys.
    """

    names = []
    values = []

    for k, v in res_dict.items():
        names.append('_and_'.join(k))
        values.append(v)

    result = namedtuple(name, names)
    return result._make(values)


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
