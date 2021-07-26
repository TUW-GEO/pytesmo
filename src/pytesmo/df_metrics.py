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

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
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
"""

from collections import namedtuple, OrderedDict
from collections.abc import Iterable
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import warnings

import pytesmo.metrics as metrics
from pytesmo.utils import array_dropna, deprecated


def n_combinations(iterable, n, must_include=None, permutations=False):
    """
    Create possible combinations of an input iterable.

    Parameters
    ---------
    iterable: Iterable
        Elements from this iterable are combined.
    n : int
        Number of elements per combination.
    must_include : Iterable, optional (default: None)
        One or more element(s) of iterable that MUST be in each combination.
    permutations : bool, optional (default: False)
        Create combinations of n elements, order matters: e.g. AB -> AB, BA
        If this is False, the output combinations will be sorted.

    Returns:
    ---------
    combs: iterable
        The possible combinations of n elements.
    """
    if must_include:
        if (not isinstance(must_include, Iterable)) or isinstance(
            must_include, str
        ):
            must_include = [must_include]

    if permutations:
        combs = [c for c in itertools.permutations(iterable, n)]
    else:
        combs = list(itertools.combinations(iterable, n))
    if must_include:
        combs_filtered = []
        for comb in combs:
            if all([i in comb for i in must_include]):
                combs_filtered.append(comb)
        combs = combs_filtered
    return combs


def _wrap_metric(metric, symmetric=True, name=None):
    """
    Wraps a metric function to be called by only providing a dataframe.

    Parameters
    ----------
    metric : callable
        Metric function from pytesmo.metrics
    symmetric : bool, optional
        Whether the metric is symmetrical w.r.t to the order of input
        arguments. Default is ``True``.
    name : str or None, optional
        The name of the namedtuple. If it is None (default), the name of the
        metric will be used.

    Returns
    -------
    wrapped : callable
        New function that takes a dataframe as input and returns the metric
        value(s) as named tuple.
        The name of the function is the same as the input function name.
    """
    metric_name = metric.__name__
    if name is None:
        name = metric_name

    def wrapped(df):
        return _dict_to_namedtuple(
            nwise_apply(df, metric, n=2, comm=symmetric), metric_name
        )

    # add name and docstring
    wrapped.__name__ = name
    wrapped.__doc__ = f"""
    Wrapper to call :py:func:`pytesmo.metrics.{metric_name}` on a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe for whose columns combinations the metric should be
        evaluated.

    Returns
    -------
    result : namedtuple
        Metric values for the different combinations. Member names are `df`'s
        column names separated by '_and_'.

    See also :py:func:`pytesmo.metrics.{metric_name}` docstring.
    """
    return wrapped


# add functions that are simple to wrap to the module
bias = _wrap_metric(metrics.bias, symmetric=False)
msd = _wrap_metric(metrics.msd)
rmsd = _wrap_metric(metrics.rmsd)
nrmsd = _wrap_metric(metrics.nrmsd)
ubrmsd = _wrap_metric(metrics.ubrmsd)
mse_corr = _wrap_metric(metrics.mse_corr)
mse_var = _wrap_metric(metrics.mse_var)
mse_bias = _wrap_metric(metrics.mse_bias)
pearson_r = _wrap_metric(metrics.pearson_r)
spearman_r = _wrap_metric(metrics.spearman_r)
kendall_tau = _wrap_metric(metrics.kendall_tau)
nash_sutcliffe = _wrap_metric(metrics.nash_sutcliffe, name="Nash_Sutcliffe")
RSS = _wrap_metric(metrics.RSS)


def mse_decomposition(df):
    """
    Mean square error (MSE) and decomposition of the MSE into individual
    error components.

    Returns
    -------
    result : namedtuple
        with column names of df for which the calculation
        was done as name of the
        element separated by '_and_'

    See Also
    --------
    pytesmo.metrics.mse_decomposition
    """
    mse, mse_corr, mse_bias, mse_var = nwise_apply(
        df, metrics.mse_decomposition, n=2, comm=True
    )
    return (
        _dict_to_namedtuple(mse, "MSE"),
        _dict_to_namedtuple(mse_corr, "MSEcorr"),
        _dict_to_namedtuple(mse_bias, "MSEbias"),
        _dict_to_namedtuple(mse_var, "MSEvar"),
    )


@deprecated()
def mse(df):
    """
    Deprecated: use :py:func:`pytesmo.df_metrics.msd` and the functions for the
    individual components instead, or
    :py:func:`pytesmo.df_metrics.msd_decomposition` for the old functionality
    with better performance.

    Mean square error (MSE) as a decomposition of the RMSD into individual
    error components

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
    MSE, MSEcorr, MSEbias, MSEvar = nwise_apply(
        df, metrics.mse, n=2, comm=True
    )
    return (
        _dict_to_namedtuple(MSE, "MSE"),
        _dict_to_namedtuple(MSEcorr, "MSEcorr"),
        _dict_to_namedtuple(MSEbias, "MSEbias"),
        _dict_to_namedtuple(MSEvar, "MSEvar"),
    )


@deprecated()
def tcol_error(df):
    """
    Deprecated: use :py:func:`pytesmo.df_metrics.tcol_metrics` instead.

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
    # For TC, the input order has NO effect --> comm=True
    err0, err1, err2 = nwise_apply(df, metrics.tcol_error, n=3, comm=True)
    trips = list(err0.keys())  # triples in all err are equal
    assert trips == list(err0.keys()) == list(err1.keys()) == list(err2.keys())

    errors = []
    for trip in trips:
        res = [err0[trip], err1[trip], err2[trip]]
        Inner = namedtuple(
            "triple_collocation_error", OrderedDict(zip(trip, res))
        )
        errors.append(Inner(*res))

    return tuple(errors)


@deprecated()
def tcol_snr(df, ref_ind=0):
    """DEPRECATED: use `tcol_metrics` instead."""
    return tcol_metrics(df, ref_ind=0)


def tcol_metrics(df, ref_ind=0):
    """
    Triple Collocation metrics applied to triples of dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        Contains the input values as time series in the df columns
    ref_ind : int or None, optional (default: 0)
        The index of the column in df that contains the reference data set.
        If None is passed, we use the first column of each triple as the
        reference, otherwise only triples that contain the reference
        dataset are considered during processing.

    Returns
    -------
    snr : namedtuple
        signal-to-noise (variance) ratio [dB] from the named columns.
    err_std_dev : namedtuple
        **SCALED** error standard deviation from the named columns
    beta : namedtuple
        Scaling coefficients (i_scaled = i * beta_i)
    """
    # For TC, the input order has NO effect --> comm=True
    if ref_ind is not None:
        # This column must be part of each triple and is always used as the
        # reference
        incl = [ref_ind]
    else:
        # All unique triples are processed, the first dataset of a triple is
        # the reference.
        incl = None
        ref_ind = 0
    snr, err, beta = nwise_apply(
        df,
        metrics.tcol_metrics,
        n=3,
        comm=True,
        must_include=incl,
        ref_ind=ref_ind,
    )

    results = {}
    var_dict = {"snr": snr, "err_std_dev": err, "beta": beta}
    for var_name, var_vals in var_dict.items():
        results[var_name] = []
        for trip, res in var_vals.items():
            Inner = namedtuple(var_name, OrderedDict(zip(trip, res)))
            results[var_name].append(Inner(*res))

    return (results["snr"], results["err_std_dev"], results["beta"])


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
    r, p = nwise_apply(df, stats.pearsonr, n=2, comm=True)
    return (
        _dict_to_namedtuple(r, "Pearsons_r"),
        _dict_to_namedtuple(p, "p_value"),
    )


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
    r, p = nwise_apply(df, stats.spearmanr, n=2, comm=True)
    return (
        _dict_to_namedtuple(r, "Spearman_r"),
        _dict_to_namedtuple(p, "p_value"),
    )


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
    r, p = nwise_apply(df, stats.kendalltau, n=2, comm=True)
    return (
        _dict_to_namedtuple(r, "Kendall_tau"),
        _dict_to_namedtuple(p, "p_value"),
    )


def pairwise_apply(df, method, comm=False):
    """
    Compute given method pairwise for all columns, excluding NA/null values

    Parameters
    ----------
    df : pd.DataFrame
        input data, method will be applied to each column pair
    method : function
        method to apply to each column pair. has to take 2 input arguments of
        type np.array and return one value or tuple of values
    comm : bool, optional (default: False)
        Also fills the lower part of the results matrix

    Returns
    -------
    results : pd.DataFrame
    """
    warnings.warn(
        "pairwise_apply() is deprecated, use nwise_apply(..., n=2) instead",
        DeprecationWarning,
    )
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


def nwise_apply(
    df,
    method,
    n=2,
    comm=False,
    as_df=False,
    ds_names=True,
    must_include=None,
    **method_kwargs,
):
    """
    Compute given method for column combinations of a data frame, excluding
    NA/null values.

    Parameters
    ----------
    df : pd.DataFrame
        Input data, method will be applied to combinations of columns of this
        df.
    method : function
        method to apply to each column pair. Has to take 2 input arguments of
        type numpy.array and return one value or tuple of values
    n : int, optional (default: 2)
        Number of columns that are combined. The default n=2 is the same as the
        previous pairwise_apply() function.
    comm : bool, optional (default: False)
        Metrics do NOT depend on the order of input values. In these cases we
        can skip unnecessary calculations and simply copy the results if
        necessary (faster).
    as_df : bool, optional (default: False)
        Return matrix structure, same as for previous pairwise_apply(), only
        available for n=2. By default, the return value will be a list of
        ordered dicts.
    ds_names : bool, optional (default: True)
        Use the column names of df to identify the dataset instead of using
        their index.
    must_include : list, optional (default: None)
        The index of one or multiple columns in df that MUST be in part of each
        combination that is processed.
    method_kwargs :
        Keyword arguments that are passed to method.

    Returns
    -------
    results : pd.DataFrame or dict or tuple
    """

    numeric_df = df._get_numeric_data()
    cols = numeric_df.columns.values
    mat = numeric_df.values
    mat = mat.T
    applyf = method

    mask = np.isfinite(mat)

    # create the possible combinations of lines
    counter = list(range(mat.shape[0]))  # get the number of lines?
    # ALL possible combinations of lines?
    perm = True if not comm else False
    combs = n_combinations(
        counter, n, must_include=must_include, permutations=perm
    )

    # find out how many variables the applyf returns
    result = []
    # apply the method using the first data set to find out the shape of c,
    c = applyf(*array_dropna(*[mat[i] for i in range(n)]))
    for index, value in enumerate(np.atleast_1d(c)):
        result.append(OrderedDict([(c, np.nan) for c in combs]))
    result = np.array(result)  # array of OrderedDicts
    # each return value result is a dict that gets filled with dicts that have
    # the cols and keys and the results as values

    lut_comb_cols = dict()

    for comb in combs:
        valid = np.logical_and(*[mask[i] for i in comb])  # where all are True

        lut_comb_cols.update(dict(zip(comb, tuple(np.take(cols, comb)))))

        if not valid.any():
            continue
        if not valid.all():
            c = applyf(*[mat[i, :][valid] for i in comb], **method_kwargs)
        else:
            c = applyf(*[mat[i, :] for i in comb], **method_kwargs)

        for index, value in enumerate(np.atleast_1d(c)):
            result[index][comb] = value

    if as_df:
        if n != 2:
            raise ValueError("Array structure only available for n=2")
        else:
            if not ds_names:
                lut_comb_cols = None
            result = [
                _to_df(r, comm=comm, lut_names=lut_comb_cols) for r in result
            ]
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


def _to_df(result, comm=False, lut_names=None):
    """
    Create a 2d results matrix/dataframe from the result dictionary to
    reproduce the output structure of the previous pairwise_apply() function.

    Parameters
    ---------
    result : OrderedDict
        The results as the are calculated in nwise_apply()
    comm : bool, optional (default: False)
        Copy elements from the upper diagonal matrix in the lower diagonal.
    lut_names: dict, optional (default: None)
        A LUT that applies nice names to the columns and lines in the data
        frame, e.g. {1:'ds1', 2:'ds2', 3:'ds3')
    """

    # find out how large the matrix is
    imax = max([max(r) for r in list(result.keys())])
    # create and fill the matrix
    res = np.full((imax + 1, imax + 1), np.nan)
    for k, v in result.items():
        res[k[::-1]] = v
    res = res.transpose()

    if comm:
        i_upper = np.triu_indices(res.shape[0], 1)
        i_lower = np.tril_indices(res.shape[0], -1)
        res[i_lower] = res[i_upper]

    if lut_names is not None:
        res = pd.DataFrame(
            data={lut_names[i]: res[:, i] for i in list(range(max(res.shape)))}
        )
    else:
        res = pd.DataFrame(
            data={i: res[:, i] for i in list(range(max(res.shape)))}
        )
    res.index = res.columns
    return res


def _dict_to_namedtuple(res_dict, name):
    """
    Takes the OrderedDictionary produced by nwise_apply(..., as_df=False) and
    produces named tuples, using the dictionary keys.
    """

    names = []
    values = []

    for k, v in res_dict.items():
        names.append("_and_".join(k))
        values.append(v)

    result = namedtuple(name, names)
    return result._make(values)
