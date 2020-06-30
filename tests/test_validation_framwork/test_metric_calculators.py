# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of TU Wien, Department of Geodesy and Geoinformation nor
#     the names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.

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

import numpy as np
import pandas as pd
from datetime import datetime

from pytesmo.validation_framework.metric_calculators import MetadataMetrics
from pytesmo.validation_framework.metric_calculators import BasicMetrics
from pytesmo.validation_framework.metric_calculators import BasicMetricsPlusMSE
from pytesmo.validation_framework.metric_calculators import IntercomparisonMetrics
from pytesmo.validation_framework.metric_calculators import TCMetrics
from pytesmo.validation_framework.metric_calculators import FTMetrics
from pytesmo.validation_framework.metric_calculators import HSAF_Metrics
from pytesmo.validation_framework.metric_calculators import RollingMetrics
from pytesmo.validation_framework.metric_calculators import MonthsMetricsAdapter
import pytesmo.metrics as metrics

import warnings
# warnings.filterwarnings("ignore")

def make_some_data():
    """
    Create a data frame with 3 columns and a pre defined bias.
    """
    startdate = datetime(2000, 1, 1)
    enddate = datetime(2000, 12, 31)
    dt_index = pd.date_range(start=startdate, end=enddate, freq='D')

    names = ['ref', 'k1', 'k2', 'k3']
    # always 0.5
    df = pd.DataFrame(index=dt_index, data={
                      name: np.repeat(0.5, dt_index.size) for name in names})

    df['k1'] += 0.2  # some positive bias
    df['k2'] -= 0.2  # some negative bias
    df['k3'] -= 0.3  # some more negative bias

    return df


def test_MetadataMetrics_calculator():
    """
    Test MetadataMetrics.
    """
    df = make_some_data()
    data = df[['ref', 'k1']]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = MetadataMetrics(other_name='k1')
        res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    assert sorted(list(res.keys())) == sorted(['gpi', 'lon', 'lat'])

    metadata_dict_template = {'network': np.array(['None'], dtype='U256'),
                              'station': np.array(['None'], dtype='U256'),
                              'landcover': np.int32([-1]),
                              'climate': np.array(['None'], dtype='U4')}

    metadata_dict = {'network': 'SOILSCAPE',
                     'station': 'node1200',
                     'landcover': 110,
                     'climate': 'Csa'}

    metriccalc = MetadataMetrics(
        other_name='k1', metadata_template=metadata_dict_template)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, metadata_dict))
    for key, value in metadata_dict.items():
        assert res[key] == metadata_dict[key]


def test_BasicMetrics_calculator():
    """
    Test BasicMetrics.
    """
    df = make_some_data()
    data = df[['ref', 'k1']]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

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
    """
    Test BasicMetrics with metadata.
    """
    df = make_some_data()
    data = df[['ref', 'k1']]

    metadata_dict_template = {'network': np.array(['None'], dtype='U256')}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = BasicMetrics(other_name='k1', calc_tau=False,
                                  metadata_template=metadata_dict_template)

        res = metriccalc.calc_metrics(
            data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

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
    """
    Test BasicMetricsPlusMSE.
    """
    df = make_some_data()
    data = df[['ref', 'k1']]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

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
    """
    Test BasicMetricsPlusMSE with metadata.
    """
    df = make_some_data()
    data = df[['ref', 'k1']]

    metadata_dict_template = {'network': np.array(['None'], dtype='U256')}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = BasicMetricsPlusMSE(
            other_name='k1', metadata_template=metadata_dict_template)
        res = metriccalc.calc_metrics(
            data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

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
    """
    Test IntercompMetrics.
    """
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = IntercomparisonMetrics(
            other_names=('k1', 'k2'), calc_tau=True)

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

    assert res['mse_bias_between_ref_and_k1'], np.array(
        [0.04], dtype='float32')
    assert res['mse_bias_between_ref_and_k2'], np.array(
        [0.04], dtype='float32')

    # scipy 1.3.0 is not built for python 2.7 so we allow both for now
    assert (np.isnan(res['p_R_between_ref_and_k1'])
            or res['p_R_between_ref_and_k1'] == 1.0)
    assert (np.isnan(res['p_R_between_ref_and_k2'])
            or res['p_R_between_ref_and_k2'] == 1.0)

    assert res['RMSD_between_ref_and_k1'] == np.array([0.2], dtype='float32')
    assert res['RMSD_between_ref_and_k2'] == np.array([0.2], dtype='float32')

    assert res['BIAS_between_ref_and_k1'] == np.array([-0.2], dtype='float32')
    assert res['BIAS_between_ref_and_k2'] == np.array([0.2], dtype='float32')

    np.testing.assert_almost_equal(
        res['urmsd_between_ref_and_k1'], np.array([0.], dtype='float32'))
    np.testing.assert_almost_equal(
        res['urmsd_between_ref_and_k2'], np.array([0.], dtype='float32'))

    assert 'RSS_between_ref_and_k1' in res.keys()
    assert 'RSS_between_ref_and_k2' in res.keys()


def test_IntercompMetrics_calculator_metadata():
    """
    Test IntercompMetrics with metadata.
    """
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metadata_dict_template = {'network': np.array(['None'], dtype='U256')}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = IntercomparisonMetrics(other_names=('k1', 'k2'), calc_tau=True,
                                            metadata_template=metadata_dict_template)
        res = metriccalc.calc_metrics(
            data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')


def test_TC_metrics_calculator():
    """
    Test TC metrics.
    """
    # this calculator uses a reference data set that is part of ALL triples.
    df = make_some_data()
    data = df[['ref', 'k1', 'k2', 'k3']]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = TCMetrics(other_names=('k1', 'k2', 'k3'), calc_tau=True,
                               dataset_names=('ref', 'k1', 'k2', 'k3'))

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

    assert res['mse_bias_between_ref_and_k1'], np.array(
        [0.04], dtype='float32')
    assert res['mse_bias_between_ref_and_k2'], np.array(
        [0.04], dtype='float32')

    # scipy 1.3.0 is not built for python 2.7 so we allow both for now
    assert (np.isnan(res['p_R_between_ref_and_k1'])
            or res['p_R_between_ref_and_k1'] == 1.0)
    assert (np.isnan(res['p_R_between_ref_and_k2'])
            or res['p_R_between_ref_and_k2'] == 1.0)

    assert res['RMSD_between_ref_and_k1'] == np.array([0.2], dtype='float32')
    assert res['RMSD_between_ref_and_k2'] == np.array([0.2], dtype='float32')

    assert res['BIAS_between_ref_and_k1'] == np.array([-0.2], dtype='float32')
    assert res['BIAS_between_ref_and_k2'] == np.array([0.2], dtype='float32')

    np.testing.assert_almost_equal(
        res['urmsd_between_ref_and_k1'], np.array([0.], dtype='float32'))
    np.testing.assert_almost_equal(
        res['urmsd_between_ref_and_k2'], np.array([0.], dtype='float32'))

    assert 'RSS_between_ref_and_k1' in res.keys()
    assert 'RSS_between_ref_and_k2' in res.keys()
    # each non-ref dataset has a snr, err and beta

    assert np.isnan(res['snr_k1_between_ref_and_k1_and_k2'])
    assert np.isnan(res['snr_k2_between_ref_and_k1_and_k2'])
    assert np.isnan(res['snr_k2_between_ref_and_k2_and_k3'])
    assert np.isnan(res['err_std_k1_between_ref_and_k1_and_k2'])
    np.testing.assert_almost_equal(
        res['beta_k1_between_ref_and_k1_and_k2'][0], 0.)
    np.testing.assert_almost_equal(
        res['beta_k2_between_ref_and_k1_and_k2'][0], 0.)
    np.testing.assert_almost_equal(
        res['beta_k3_between_ref_and_k1_and_k3'][0], 0.)


def test_TC_metrics_calculator_metadata():
    """
    Test TC metrics with metadata.
    """
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metadata_dict_template = {'network': np.array(['None'], dtype='U256')}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = TCMetrics(other_names=('k1', 'k2'), calc_tau=True,
                               dataset_names=['ref', 'k1', 'k2'], metadata_template=metadata_dict_template)
        res = metriccalc.calc_metrics(
            data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')


def test_FTMetrics():
    """
    Test FT metrics.
    """
    df = make_some_data()
    data = df[['ref', 'k1']]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = FTMetrics(frozen_flag=2, other_name='k1')
        res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(n_obs=np.array([366]), ssf_fr_temp_un=np.array(
        [0.0], dtype='float32'), dtype='float32')

    assert res['n_obs'] == should['n_obs']
    assert res['ssf_fr_temp_un'] == should['ssf_fr_temp_un']


def test_FTMetrics_metadata():
    """
    Test FT metrics with metadata.
    """
    df = make_some_data()
    data = df[['ref', 'k1']]

    metadata_dict_template = {'network': np.array(['None'], dtype='U256')}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data
        metriccalc = FTMetrics(frozen_flag=2, other_name='k1',
                               metadata_template=metadata_dict_template)
        res = metriccalc.calc_metrics(
            data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')


def test_BasicSeasonalMetrics():
    """
    Test BasicSeasonalMetrics.
    """
    df = make_some_data()
    data = df[['ref', 'k1']]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = MonthsMetricsAdapter(BasicMetrics(other_name='k1'))
        res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(ALL_n_obs=np.array([366]), dtype='float32')

    assert res['ALL_n_obs'] == should['ALL_n_obs']
    assert np.isnan(res['ALL_rho'])


def test_BasicSeasonalMetrics_metadata():
    """
    Test BasicSeasonalMetrics with metadata.
    """
    df = make_some_data()
    data = df[['ref', 'k1']]

    metadata_dict_template = {'network': np.array(['None'], dtype='U256')}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = MonthsMetricsAdapter(BasicMetrics(
            other_name='k1', metadata_template=metadata_dict_template))
        res = metriccalc.calc_metrics(
            data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')


def test_HSAF_Metrics():
    """
    Test HSAF Metrics
    """
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = HSAF_Metrics(other_name1='k1', other_name2='k2')
        res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(ALL_n_obs=np.array([366]), dtype='float32')

    assert res['ALL_n_obs'] == should['ALL_n_obs']
    assert np.isnan(res['ref_k1_ALL_rho'])
    assert np.isnan(res['ref_k2_ALL_rho'])


def test_HSAF_Metrics_metadata():
    """
    Test HSAF Metrics with metadata.
    """
    df = make_some_data()
    data = df[['ref', 'k1', 'k2']]

    metadata_dict_template = {'network': np.array(['None'], dtype='U256')}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = HSAF_Metrics(
            other_name1='k1', metadata_template=metadata_dict_template)
        res = metriccalc.calc_metrics(
            data, gpi_info=(0, 0, 0, {'network': 'SOILSCAPE'}))

    assert res['network'] == np.array(['SOILSCAPE'], dtype='U256')


def test_RollingMetrics():
    """
    Test RollingMetrics.
    """
    df = make_some_data()
    df['ref'] += np.random.rand(len(df))
    df['k1'] += np.random.rand(len(df))
    data = df[['ref', 'k1']]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # many warnings due to test data

        metriccalc = RollingMetrics(other_name='k1')
        dataset = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0), center=False)

    # test pearson r
    ref_array = df['ref'].rolling('30d').corr(df['k1'])
    np.testing.assert_almost_equal(dataset['R'][0], ref_array.values)

    # test rmsd
    indexer = np.arange(30)[None, :] + np.arange(len(df)-30)[:, None]
    rmsd_arr = []
    for i in range(indexer.shape[0]):
        rmsd_arr.append(metrics.rmsd(df['ref'][indexer[i, :]],
                                     df['k1'][indexer[i, :]]))

    rmsd_arr = np.array(rmsd_arr)
    np.testing.assert_almost_equal(dataset['RMSD'][0][29:-1], rmsd_arr)
