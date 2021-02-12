# -*- coding: utf-8 -*-

"""
Test functions from the df_metrics module that applies metrics to combinations
of columns from a data frame.
"""

import numpy as np
import pandas as pd
import pytesmo.df_metrics as df_metrics
from pytesmo.metrics import bias, pearsonr
import pytest
from scipy import stats

def test_n_combinations():
    coll = [1,2,3,4]
    combs = df_metrics.n_combinations(coll, n=2, must_include=[1], permutations=False)
    assert combs == [(1,2), (1,3), (1,4)]

    coll = [1, 2, 3, 4]
    combs = df_metrics.n_combinations(coll, n=3, permutations=False)
    assert combs == [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]

def test_apply():

    df = pd.DataFrame(index=pd.date_range(start='2000-01-01', end='2000-12-31', freq='D'),
                      data={'ds0': np.repeat(0, 366), 'ds1': np.repeat(1, 366)})
    df.loc[df.index[np.random.choice(range(366), 10)], 'ds0'] = np.nan
    df.loc[df.index[np.random.choice(range(366), 10)], 'ds1'] = np.nan
    with pytest.deprecated_call():
        bias_matrix_old = df_metrics.pairwise_apply(df, bias)
    bias_matrix_new = df_metrics.nwise_apply(df, bias,  n=2, as_df=True)
    r_matrix_new = df_metrics.nwise_apply(df, stats.pearsonr,  n=2, as_df=True)
    assert bias_matrix_old.equals(bias_matrix_new)

    # check if dict implementation and matrix implementation have same result
    bias_new = df_metrics.nwise_apply(df, bias,  n=2, as_df=False)
    for i, v in bias_new.items():
        assert bias_matrix_new.loc[i] == v

def test_dict_to_namedtuple():
    d = {'a': 1, 'b': 2}
    d_named = df_metrics._dict_to_namedtuple(d, 'name')
    assert d_named._fields == ('a', 'b')
    assert type(d_named).__name__ == 'name'

if __name__ == '__main__':
    test_apply()
