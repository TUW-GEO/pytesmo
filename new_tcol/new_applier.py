# -*- coding: utf-8 -*-
"""
Functions to apply triple collocation metrics to triples of data sets.
"""
import numpy as np
import pandas as pd
import pytesmo.df_metrics as df_metrics
import pytesmo.metrics as metrics
from pprint import pprint

def create_testdata(n=3, biases=np.array([0.2, 0.3, 0.4]), seed=12345):
    np.random.seed(seed)

    df = []
    for i in range(n):
        ds = pd.Series(index=range(100), data=np.random.rand(100))
        ds = ds + biases[i]
        ds.name = 'ds{}'.format(i)
        df.append(ds)

    df = pd.concat(df, axis=1)
    return df

def read_realdata(n=4):
    df = pd.read_csv('cci44_merra2_gldas21_era5_at_-91.625_32.875.csv', index_col=0)
    if n > len(df.columns):
        raise ValueError('There are only {} columns available'.format(len(df.columns)))
    return df.iloc[:, :n]

def df_snr(realdata=False, n=3):
    if realdata:
        df = read_realdata(n=n).dropna()
    else:
        df = create_testdata(n, np.array(list(range(n))))
    res = df_metrics.tcol_snr(df, ref_ind=0)

def df_err(realdata=False, n=3):
    if realdata:
        df = read_realdata(n).dropna()
    else:
        df = create_testdata(n, np.array(list(range(n))))
    err = df_metrics.tcol_error(df)
    print('new_err')
    pprint(err)
    if n == 3:
        err_old = df_metrics.old_tcol_error(df)
        print('old_err')
        pprint(err_old)


def new_pairwise_apply(method=metrics.pearsonr):
    df = create_testdata(3, np.array([1., 2., 3.]))
    result = df_metrics.nwise_apply(df, method, comm=False, n=2, as_df=False)
    named_tupel = df_metrics._dict_to_namedtuple(result[0], 'R')
    return named_tupel

def old_pairwise_apply():
    df = create_testdata(3, np.array([1., 2., 3.]))
    method = metrics.pearsonr
    result = df_metrics.pairwise_apply(df, method, comm=False)
    named_tupel = df_metrics._to_namedtuple(result[0], 'R')
    return named_tupel

if __name__ == '__main__':
    snr = df_snr(True, n=4)

    err = df_err(True, n=4)



    new_result = new_pairwise_apply()
    old_result = old_pairwise_apply()
