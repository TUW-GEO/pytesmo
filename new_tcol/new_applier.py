# -*- coding: utf-8 -*-
"""
Created on Sep 15 18:50 2019

@author: wolfgang
"""
import numpy as np
import pandas as pd
from pytesmo.df_metrics import pairwise_apply, _to_namedtuple
from pytesmo.metrics import bias, pearsonr
import itertools
from collections import OrderedDict

def n_combinations(iterable, n, must_include=None, with_replacement=False):
    """
    Create possible combinations of an input iterable.
    # todo: for e.g. bias we must also allow the elements AB and BA!!
    Parameters
    ---------
    iterable: iterable
        Elements from this iterable are combined.
    n : int
        Number of elements that are one combination
    must_include : list of element(s) of iterable, optional (default: None)
        One or multiple elements that MUST be in each combination
    with_replacement : bool, optional (default: False)
        Add combinations of elements with itself.

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

def nwise_apply(df, method, comm=False, n=2, as_df=False):
    """
    Compute given method pairwise for all columns, excluding NA/null values

    Parameters
    ----------
    df : pandas.DataFrame
        input data, method will be applied to each column pair
    method : function
        method to apply to each column pair. has to take 2 input arguments of
        type numpy.array and return one value or tuple of values
    as_df : bool, optional (default: False)
        Return array structure, same as for pairwise apply, only available for
        n=2

    Returns
    -------
    results : pandas.DataFrame
    """

    numeric_df = df._get_numeric_data()
    cols = numeric_df.columns.values
    mat = numeric_df.values
    mat = mat.T
    applyf = method

    # find out how many variables the applyf returns
    result = []
    c = applyf(mat[0], mat[0]) # each return val has an empty result
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
            c = applyf(*[mat[i,:][valid] for i in comb])
        else:
            c = applyf(*[mat[i,:] for i in comb])

        lut_comb_cols.update(dict(zip(comb, tuple(cols[[*comb]]))))


        for index, value in enumerate(np.atleast_1d(c)):
            result[index][comb] = value
            if comm:
                result[index][comb] = value

    if as_df:
        if n != 2:
            raise ValueError('Array result only available for n=2')
        else:
            result = [_to_df(r, lut_comb_cols) for r in result]
    else:
        if len(result) == 1:
            result = result[0]
        else:
            result = tuple(result)

    return result

def _to_df(result, lut_names=None):
    imax = max([max(r) for r in list(result.keys())])
    res = np.full((imax+1, imax+1), np.nan)
    for k, v in result.items():
        res[k] = v

    res = pd.DataFrame(data={lut_names[i]: res[:, i] for i in list(range(max(res.shape)))})
    res.index = res.columns
    return res

def create_testdata(n=3, biases=np.array([0.2, 0.3, 0.4])):
    np.random.seed = 12345

    df = []
    for i in range(n):
        ds = pd.Series(index=range(100), data=np.random.rand(100))
        ds = ds + biases[i]
        ds.name = 'ds{}'.format(i)
        df.append(ds)

    df = pd.concat(df, axis=1)
    return df

def new_pairwise_apply():
    df = create_testdata(3, np.array([1., 2., 3.]))
    method = pearsonr
    result = nwise_apply(df, method, comm=False, n=2, as_df=True)
    named_tupel = _to_namedtuple(result[0], 'R')

def old_pairwise_apply():
    df = create_testdata(3, np.array([1., 2., 3.]))
    method = pearsonr
    oldresult = pairwise_apply(df, method, comm=False)

if __name__ == '__main__':
    new_pairwise_apply()