# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
from pytesmo.validation_framework.metric_calculators import *

def make_some_data():
    """
    Create a data frame with 3 columns and a pre defined bias.
    """
    startdate = datetime(2000,1,1)
    enddate = datetime(2000,12,31)
    dt_index = pd.date_range(start=startdate, end=enddate, freq='D')

    names = ['ref', 'k1', 'k2']
    # always 0.5
    df = pd.DataFrame(index=dt_index, data={name: np.repeat(0.5, dt_index.size) for name in names})

    df['k1'] += 0.2 # some positive bias
    df['k2'] -= 0.2 # some negative bias

    return df

def test_BasicMetrics_calculator():

    df = make_some_data()
    data = df[['ref', 'k1']]

    metrics = BasicMetrics(other_name='k1', calc_tau=False)
    res = metrics.calc_metrics(data, gpi_info=(0,0,0))

    should = dict(n_obs=np.array([366]), RMSD=np.array([0.2], dtype='float32'),
                  BIAS=np.array([-0.2], dtype='float32'), p_R=np.array([1.], dtype='float32'))

    assert res['n_obs'] == should['n_obs']
    assert np.isnan(res['rho'])
    assert res['RMSD'] == should['RMSD']
    assert res['BIAS'] == should['BIAS']
    assert np.isnan(res['R'])
    assert res['p_R'] == should['p_R']



def test_IntercompMetrics_calculator():

    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metrics = IntercomparisonMetrics(other_names=['k1', 'k2'], calc_tau=False)
    res = metrics.calc_metrics(data, gpi_info=(0,0,0))

    assert res['n_obs'] == np.array([366])

    assert np.isnan(res['R_between_ref_k1'])
    assert np.isnan(res['R_between_ref_k2'])

    assert np.isnan(res['rho_between_ref_k1'])
    assert np.isnan(res['rho_between_ref_k2'])

    assert np.isnan(res['mse_between_ref_k1'])
    assert np.isnan(res['mse_between_ref_k2'])

    assert np.isnan(res['mse_corr_between_ref_k1'])
    assert np.isnan(res['mse_corr_between_ref_k2'])

    assert res['mse_bias_between_ref_k1'], np.array([0.04], dtype='float32')
    assert res['mse_bias_between_ref_k2'], np.array([0.04], dtype='float32')

    assert res['p_R_between_ref_k1'] == np.array([1.], dtype='float32')
    assert res['p_R_between_ref_k2'] == np.array([1.], dtype='float32')

    assert res['rmsd_between_ref_k1'] == np.array([0.2], dtype='float32')
    assert res['rmsd_between_ref_k2'] == np.array([0.2], dtype='float32')

    assert res['bias_between_ref_k1'] == np.array([0.2], dtype='float32')
    assert res['bias_between_ref_k2'] == np.array([-0.2], dtype='float32')

    np.testing.assert_almost_equal(res['ubRMSD_between_ref_k1'], np.array([0.],dtype='float32'))
    np.testing.assert_almost_equal(res['ubRMSD_between_ref_k2'], np.array([0.],dtype='float32'))


def test_TC_metrics_calculator():
    # todo: choos example data that returns tc variables.
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]


    metrics = TCMetrics(other_name1='k1', other_name2= 'k2', calc_tau=False,
                        dataset_names=['ref', 'k1', 'k2'])

    res = metrics.calc_metrics(data, gpi_info=(0,0,0))

    assert res['n_obs'] == np.array([366])

    assert res['rmsd_between_ref_k1'] == np.array([0.2], dtype='float32')
    assert res['rmsd_between_ref_k2'] == np.array([0.2], dtype='float32')

    assert res['bias_between_ref_k1'] == np.array([0.2], dtype='float32')
    assert res['bias_between_ref_k2'] == np.array([-0.2], dtype='float32')

    assert res['p_R_between_ref_k1'] == np.array([1.], dtype='float32')
    assert res['p_R_between_ref_k2'] == np.array([1.], dtype='float32')

    assert res['mse_bias_between_ref_k1'], np.array([0.04], dtype='float32')
    assert res['mse_bias_between_ref_k2'], np.array([0.04], dtype='float32')

    assert np.isnan(res['R_between_ref_k1'])
    assert np.isnan(res['R_between_ref_k2'])

    assert np.isnan(res['rho_between_ref_k1'])
    assert np.isnan(res['rho_between_ref_k2'])

    assert np.isnan(res['mse_between_ref_k1'])
    assert np.isnan(res['mse_between_ref_k2'])

    assert np.isnan(res['mse_corr_between_ref_k1'])
    assert np.isnan(res['mse_corr_between_ref_k2'])

    np.testing.assert_almost_equal(res['ubRMSD_between_ref_k1'], np.array([0.],dtype='float32'))
    np.testing.assert_almost_equal(res['ubRMSD_between_ref_k2'], np.array([0.],dtype='float32'))



if __name__ == '__main__':
    test_TC_metrics_calculator()