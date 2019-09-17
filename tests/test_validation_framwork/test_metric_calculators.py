# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
from pytesmo.validation_framework.metric_calculators import *

import warnings
warnings.filterwarnings("ignore")

def make_some_data():
    """
    Create a data frame with 3 columns and a pre defined bias.
    """
    startdate = datetime(2000, 1, 1)
    enddate = datetime(2000, 12, 31)
    dt_index = pd.date_range(start=startdate, end=enddate, freq='D')

    names = ['ref', 'k1', 'k2']
    # always 0.5
    df = pd.DataFrame(index=dt_index, data={name: np.repeat(0.5, dt_index.size) for name in names})

    df['k1'] += 0.2  # some positive bias
    df['k2'] -= 0.2  # some negative bias

    return df

def test_MetadataMetrics_calculator():
    df = make_some_data()
    data = df[['ref', 'k1']]

    metriccalc = MetadataMetrics(other_name='k1')
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))
    assert sorted(list(res.keys())) == sorted(['gpi', 'lon', 'lat'])

    metadata_dict_template = {'network' : np.array(['None'], dtype='U256'),
                              'station' : np.array(['None'], dtype='U256'),
                              'landcover' : np.int32([-1]),
                              'climate' : np.array(['None'], dtype='U4')}

    metadata_dict = {'network' : 'SOILSCAPE',
                      'station' : 'node1200',
                      'landcover' : 110,
                      'climate' : 'Csa'}

    metriccalc = MetadataMetrics(other_name='k1', metadata_template=metadata_dict_template)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, metadata_dict))
    for key, value in metadata_dict.items():
        assert res[key] == metadata_dict[key]

def test_BasicMetrics_calculator():
    df = make_some_data()
    data = df[['ref', 'k1']]

    metriccalc = BasicMetrics(other_name='k1', calc_tau=False)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(n_obs=np.array([366]), RMSD=np.array([0.2], dtype='float32'),
                  BIAS=np.array([-0.2], dtype='float32'))

    assert res['n_obs'] == should['n_obs']
    assert np.isnan(res['rho'])
    assert res['RMSD'] == should['RMSD']
    assert res['BIAS'] == should['BIAS']
    assert np.isnan(res['R'])
    # scipy 1.3.0 is not built for python 2.7 so we allow both for now
    assert (np.isnan(res['p_R']) or res['p_R'] == 1.0)

def test_BasicMetrics_calculator_metadata():
    df = make_some_data()
    data = df[['ref', 'k1']]

    metadata_dict_template = {'network' : np.array(['None'], dtype='U256')}

    metriccalc = BasicMetrics(other_name='k1', calc_tau=False, metadata_template=metadata_dict_template)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    should = dict(network=np.array(['SOILSCAPE'], dtype='U256'),
                  n_obs=np.array([366]), RMSD=np.array([0.2], dtype='float32'),
                  BIAS=np.array([-0.2], dtype='float32'), dtype='float32')

    assert res['n_obs'] == should['n_obs']
    assert np.isnan(res['rho'])
    assert res['RMSD'] == should['RMSD']
    assert res['BIAS'] == should['BIAS']
    assert res['network'] == should['network']
    assert np.isnan(res['R'])
    # depends on scipy version changed after v1.2.1
    assert res['p_R'] == np.array([1.]) or np.isnan(res['R'])

def test_BasicMetricsPlusMSE_calculator():
    df = make_some_data()
    data = df[['ref', 'k1']]

    metadata_dict_template = {'network' : np.array(['None'], dtype='U256')}

    metriccalc = BasicMetricsPlusMSE(other_name='k1')
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(network=np.array(['SOILSCAPE'], dtype='U256'),
                  n_obs=np.array([366]), RMSD=np.array([0.2], dtype='float32'),
                  BIAS=np.array([-0.2], dtype='float32'), dtype='float32')

    assert res['n_obs'] == should['n_obs']
    assert np.isnan(res['rho'])
    assert res['RMSD'] == should['RMSD']
    assert res['BIAS'] == should['BIAS']
    assert np.isnan(res['R'])
    # depends on scipy version changed after v1.2.1
    assert res['p_R'] == np.array([1.]) or np.isnan(res['R'])

def test_BasicMetricsPlusMSE_calculator_metadata():
    df = make_some_data()
    data = df[['ref', 'k1']]

    metadata_dict_template = {'network' : np.array(['None'], dtype='U256')}

    metriccalc = BasicMetricsPlusMSE(other_name='k1', metadata_template=metadata_dict_template)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    should = dict(network=np.array(['SOILSCAPE'], dtype='U256'),
                  n_obs=np.array([366]), RMSD=np.array([0.2], dtype='float32'),
                  BIAS=np.array([-0.2], dtype='float32'), dtype='float32')

    assert res['n_obs'] == should['n_obs']
    assert np.isnan(res['rho'])
    assert res['RMSD'] == should['RMSD']
    assert res['BIAS'] == should['BIAS']
    assert res['network'] == should['network']
    assert np.isnan(res['R'])
    # depends on scipy version changed after v1.2.1
    assert res['p_R'] == np.array([1.]) or np.isnan(res['R'])

def test_IntercompMetrics_calculator():
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metriccalc = IntercomparisonMetrics(other_names=['k1', 'k2'], calc_tau=False)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    assert res['n_obs'] == np.array([366])

    assert np.isnan(res['R_between_ref_and_k1'])
    assert np.isnan(res['R_between_ref_and_k2'])

    assert np.isnan(res['rho_between_ref_and_k1'])
    assert np.isnan(res['rho_between_ref_and_k2'])

    assert np.isnan(res['mse_between_ref_and_k1'])
    assert np.isnan(res['mse_between_ref_and_k2'])

    assert np.isnan(res['mse_corr_between_ref_and_k1'])
    assert np.isnan(res['mse_corr_between_ref_and_k2'])

    assert res['mse_bias_between_ref_and_k1'], np.array([0.04], dtype='float32')
    assert res['mse_bias_between_ref_and_k2'], np.array([0.04], dtype='float32')

    # scipy 1.3.0 is not built for python 2.7 so we allow both for now
    assert (np.isnan(res['p_R_between_ref_and_k1']) or res['p_R_between_ref_and_k1'] == 1.0)
    assert (np.isnan(res['p_R_between_ref_and_k2']) or res['p_R_between_ref_and_k2'] == 1.0)

    assert res['RMSD_between_ref_and_k1'] == np.array([0.2], dtype='float32')
    assert res['RMSD_between_ref_and_k2'] == np.array([0.2], dtype='float32')

    assert res['BIAS_between_ref_and_k1'] == np.array([0.2], dtype='float32')
    assert res['BIAS_between_ref_and_k2'] == np.array([-0.2], dtype='float32')

    np.testing.assert_almost_equal(res['urmsd_between_ref_and_k1'], np.array([0.], dtype='float32'))
    np.testing.assert_almost_equal(res['urmsd_between_ref_and_k2'], np.array([0.], dtype='float32'))

    assert 'RSS_between_ref_and_k1' in res.keys()
    assert 'RSS_between_ref_and_k2' in res.keys()

def test_IntercompMetrics_calculator_metadata():
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metadata_dict_template = {'network' : np.array(['None'], dtype='U256')}

    metriccalc = IntercomparisonMetrics(other_names=['k1', 'k2'], calc_tau=False,
                                        metadata_template=metadata_dict_template)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')

def test_TC_metrics_calculator():
    # this calculator uses a reference data set that is part of ALL triples.
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metriccalc = TCMetrics(other_name1='k1', other_name2='k2', calc_tau=False,
                           dataset_names=['ref', 'k1', 'k2'])

    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    assert res['n_obs'] == np.array([366])

    assert res['rmsd_between_ref_k1'] == np.array([0.2], dtype='float32')
    assert res['rmsd_between_ref_k2'] == np.array([0.2], dtype='float32')

    assert res['bias_between_ref_k1'] == np.array([0.2], dtype='float32')
    assert res['bias_between_ref_k2'] == np.array([-0.2], dtype='float32')

    # scipy 1.3.0 is not built for python 2.7 so we allow both for now
    assert (np.isnan(res['p_R_between_ref_k1']) or res['p_R_between_ref_k1'] == 1.0)
    assert (np.isnan(res['p_R_between_ref_k2']) or res['p_R_between_ref_k1'] == 1.0)

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

    np.testing.assert_almost_equal(res['ubRMSD_between_ref_k1'], np.array([0.], dtype='float32'))
    np.testing.assert_almost_equal(res['ubRMSD_between_ref_k2'], np.array([0.], dtype='float32'))

def test_TC_metrics_calculator_metadata():
    # todo: choose example data that returns tc variables.
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metadata_dict_template = {'network' : np.array(['None'], dtype='U256')}

    metriccalc = TCMetrics(other_name1='k1', other_name2='k2', calc_tau=False,
                           dataset_names=['ref', 'k1', 'k2'], metadata_template=metadata_dict_template)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')

def test_FTMetrics():
    df = make_some_data()
    data = df[['ref', 'k1']]

    metriccalc = FTMetrics(frozen_flag=2, other_name='k1')
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(n_obs=np.array([366]), ssf_fr_temp_un=np.array([0.0], dtype='float32'), dtype='float32')

    assert res['n_obs'] == should['n_obs']
    assert res['ssf_fr_temp_un'] == should['ssf_fr_temp_un']

def test_FTMetrics_metadata():
    df = make_some_data()
    data = df[['ref', 'k1']]

    metadata_dict_template = {'network' : np.array(['None'], dtype='U256')}

    metriccalc = FTMetrics(frozen_flag=2, other_name='k1', metadata_template=metadata_dict_template)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')

def test_BasicSeasonalMetrics():
    df = make_some_data()
    data = df[['ref', 'k1']]

    metriccalc = BasicSeasonalMetrics(other_name='k1')
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(ALL_n_obs=np.array([366]), dtype='float32')

    assert res['ALL_n_obs'] == should['ALL_n_obs']
    assert np.isnan(res['ALL_rho'])

def test_BasicSeasonalMetrics_metadata():
    df = make_some_data()
    data = df[['ref', 'k1']]

    metadata_dict_template = {'network' : np.array(['None'], dtype='U256')}

    metriccalc = BasicSeasonalMetrics(other_name='k1', metadata_template=metadata_dict_template)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')

def test_HSAF_Metrics():
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metriccalc = HSAF_Metrics(other_name1='k1', other_name2='k2')
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(ALL_n_obs=np.array([366]), dtype='float32')

    assert res['ALL_n_obs'] == should['ALL_n_obs']
    assert np.isnan(res['ref_k1_ALL_rho'])
    assert np.isnan(res['ref_k2_ALL_rho'])


def test_HSAF_Metrics_metadata():
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metadata_dict_template = {'network' : np.array(['None'], dtype='U256')}

    metriccalc = HSAF_Metrics(other_name1='k1', metadata_template=metadata_dict_template)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')
