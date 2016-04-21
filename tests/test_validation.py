# coding: utf-8
# Copyright (c) 2015,Vienna University of Technology,
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
'''
Tests for the validation framework
Created on Mon Jul  6 12:49:07 2015
'''

import os
import tempfile
import netCDF4 as nc
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.util.testing as pdtest
import pytest

import pygeogrids.grids as grids
from pygeobase.io_base import GriddedTsBase

import pytesmo.validation_framework.temporal_matchers as temporal_matchers
import pytesmo.validation_framework.metric_calculators as metrics_calculators
from pytesmo.validation_framework.results_manager import netcdf_results_manager
from pytesmo.validation_framework.data_manager import DataManager
from pytesmo.validation_framework.data_manager import get_result_names

from datetime import datetime

from pytesmo.io.sat.ascat import AscatH25_SSM
from pytesmo.io.ismn.interface import ISMN_Interface
from pytesmo.validation_framework.validation import Validation
from pytesmo.validation_framework.validation import args_to_iterable


def test_ascat_ismn_validation():
    """
    Test processing framework with some ISMN and ASCAT sample data
    """
    ascat_data_folder = os.path.join(os.path.dirname(__file__), 'test-data',
                                     'sat', 'ascat', 'netcdf', '55R22')

    ascat_grid_folder = os.path.join(os.path.dirname(__file__), 'test-data',
                                     'sat', 'ascat', 'netcdf', 'grid')

    ascat_reader = AscatH25_SSM(ascat_data_folder, ascat_grid_folder)
    ascat_reader.read_bulk = True
    ascat_reader._load_grid_info()

    # Initialize ISMN reader

    ismn_data_folder = os.path.join(os.path.dirname(__file__), 'test-data',
                                    'ismn', 'multinetwork', 'header_values')
    ismn_reader = ISMN_Interface(ismn_data_folder)

    jobs = []

    ids = ismn_reader.get_dataset_ids(
        variable='soil moisture',
        min_depth=0,
        max_depth=0.1)
    for idx in ids:
        metadata = ismn_reader.metadata[idx]
        jobs.append((idx, metadata['longitude'], metadata['latitude']))

    # Create the variable ***save_path*** which is a string representing the
    # path where the results will be saved. **DO NOT CHANGE** the name
    # ***save_path*** because it will be searched during the parallel
    # processing!

    save_path = tempfile.mkdtemp()

    # Create the validation object.

    datasets = {
        'ISMN': {
            'class': ismn_reader,
            'columns': ['soil moisture']
        },
        'ASCAT': {
            'class': ascat_reader,
            'columns': ['sm'],
            'kwargs': {'mask_frozen_prob': 80,
                       'mask_snow_prob': 80,
                       'mask_ssf': True}
        }}

    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

    process = Validation(
        datasets, 'ISMN',
        temporal_ref='ASCAT',
        scaling='lin_cdf_match',
        scaling_ref='ASCAT',
        metrics_calculators={
            (2, 2): metrics_calculators.BasicMetrics(other_name='k1').calc_metrics},
        period=period)

    for job in jobs:
        results = process.calc(*job)
        netcdf_results_manager(results, save_path)

    results_fname = os.path.join(
        save_path, 'ASCAT.sm_with_ISMN.soil moisture.nc')

    vars_should = [u'n_obs', u'tau', u'gpi', u'RMSD', u'lon', u'p_tau',
                   u'BIAS', u'p_rho', u'rho', u'lat', u'R', u'p_R']
    n_obs_should = [360, 385, 1644, 1881, 1927, 479, 140, 251]
    rho_should = np.array([0.546187,
                           0.717398,
                           0.620892,
                           0.532465,
                           0.302997,
                           0.694713,
                           0.840592,
                           0.742065],
                          dtype=np.float32)

    rmsd_should = np.array([11.536263,
                            7.545650,
                            17.451935,
                            21.193714,
                            14.246680,
                            14.494674,
                            13.173215,
                            12.903898],
                           dtype=np.float32)
    with nc.Dataset(results_fname) as results:
        assert sorted(results.variables.keys()) == sorted(vars_should)
        assert sorted(results.variables['n_obs'][:].tolist()) == sorted(
            n_obs_should)
        nptest.assert_allclose(sorted(rho_should),
                               sorted(results.variables['rho'][:]),
                               rtol=1e-4)
        nptest.assert_allclose(sorted(rmsd_should),
                               sorted(results.variables['RMSD'][:]),
                               rtol=1e-4)


class TestDataset(object):
    """Test dataset that acts as a fake object for the base classes."""

    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode

    def read(self, *args):

        n = 1000
        x = np.arange(n)
        y = np.arange(n) * 0.5
        index = pd.date_range(start="2000-01-01", periods=n, freq="D")

        df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
        return df

    def write(self, gpi, data):
        return None

    def read_ts(self, *args, **kwargs):
        return self.read(*args, **kwargs)

    def read_ts_other(self, *args, **kwargs):
        return self.read(*args, **kwargs) * 2

    def write_ts(self, gpi, data):
        return None

    def close(self):
        pass

    def flush(self):
        pass


class MaskingTestDataset(TestDataset):

    def read(self, *args, **kwargs):
        limit = kwargs.pop("limit")
        data = super(MaskingTestDataset, self).read(*args)
        data = data[['x']]
        data = data < limit
        return data


def test_masking_testdataset():

    ds = MaskingTestDataset("")
    data = ds.read(1, limit=500)
    data_should = np.concatenate([np.ones((500), dtype=bool),
                                  np.zeros((500), dtype=bool)])
    nptest.assert_almost_equal(data['x'].values, data_should)
    data = ds.read(1, limit=250)
    data_should = np.concatenate([np.ones((250), dtype=bool),
                                  np.zeros((750), dtype=bool)])
    nptest.assert_almost_equal(data['x'].values, data_should)


def setup_TestDatasets():
    grid = grids.CellGrid(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]),
                          np.array([4, 4, 2, 1]), gpis=np.array([1, 2, 3, 4]))

    ds1 = GriddedTsBase("", grid, TestDataset)
    ds2 = GriddedTsBase("", grid, TestDataset)
    ds3 = GriddedTsBase("", grid, TestDataset)

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['x'],
            'args': [],
            'kwargs': {}
        },
        'DS2': {
            'class': ds2,
            'columns': ['y'],
            'args': [],
            'kwargs': {},
            'use_lut': False,
            'grids_compatible': True
        },
        'DS3': {
            'class': ds3,
            'columns': ['x', 'y'],
            'args': [],
            'kwargs': {},
            'use_lut': False,
            'grids_compatible': True
        }
    }
    return datasets


def test_validation_n2_k2():

    tst_results = {
        (('DS1', 'x'), ('DS3', 'y')): {
            'n_obs': np.array([1000], dtype=np.int32),
            'tau': np.array([np.nan], dtype=np.float32),
            'gpi': np.array([4], dtype=np.int32),
            'RMSD': np.array([0.], dtype=np.float32),
            'lon': np.array([4.]),
            'p_tau': np.array([np.nan], dtype=np.float32),
            'BIAS': np.array([0.], dtype=np.float32),
            'p_rho': np.array([0.], dtype=np.float32),
            'rho': np.array([1.], dtype=np.float32),
            'lat': np.array([4.]),
            'R': np.array([1.], dtype=np.float32),
            'p_R': np.array([0.], dtype=np.float32)},
        (('DS1', 'x'), ('DS2', 'y')): {
            'n_obs': np.array([1000], dtype=np.int32),
            'tau': np.array([np.nan], dtype=np.float32),
            'gpi': np.array([4], dtype=np.int32),
            'RMSD': np.array([0.], dtype=np.float32),
            'lon': np.array([4.]),
            'p_tau': np.array([np.nan], dtype=np.float32),
            'BIAS': np.array([0.], dtype=np.float32),
            'p_rho': np.array([0.], dtype=np.float32),
            'rho': np.array([1.], dtype=np.float32),
            'lat': np.array([4.]),
            'R': np.array([1.], dtype=np.float32),
            'p_R': np.array([0.], dtype=np.float32)},
        (('DS1', 'x'), ('DS3', 'x')): {
            'n_obs': np.array([1000], dtype=np.int32),
            'tau': np.array([np.nan], dtype=np.float32),
            'gpi': np.array([4], dtype=np.int32),
            'RMSD': np.array([0.], dtype=np.float32),
            'lon': np.array([4.]),
            'p_tau': np.array([np.nan], dtype=np.float32),
            'BIAS': np.array([0.], dtype=np.float32),
            'p_rho': np.array([0.], dtype=np.float32),
            'rho': np.array([1.], dtype=np.float32),
            'lat': np.array([4.]),
            'R': np.array([1.], dtype=np.float32),
            'p_R': np.array([0.], dtype=np.float32)}}

    datasets = setup_TestDatasets()

    process = Validation(
        datasets, 'DS1',
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0).combinatory_matcher,
        scaling='lin_cdf_match',
        metrics_calculators={
            (2, 2): metrics_calculators.BasicMetrics(other_name='k1').calc_metrics})

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n3_k2():

    tst_results = {
        (('DS1', 'x'), ('DS3', 'y')): {
            'n_obs': np.array([1000], dtype=np.int32),
            'tau': np.array([np.nan], dtype=np.float32),
            'gpi': np.array([4], dtype=np.int32),
            'RMSD': np.array([0.], dtype=np.float32),
            'lon': np.array([4.]),
            'p_tau': np.array([np.nan], dtype=np.float32),
            'BIAS': np.array([0.], dtype=np.float32),
            'p_rho': np.array([0.], dtype=np.float32),
            'rho': np.array([1.], dtype=np.float32),
            'lat': np.array([4.]),
            'R': np.array([1.], dtype=np.float32),
            'p_R': np.array([0.], dtype=np.float32)},
        (('DS1', 'x'), ('DS2', 'y')): {
            'n_obs': np.array([1000], dtype=np.int32),
            'tau': np.array([np.nan], dtype=np.float32),
            'gpi': np.array([4], dtype=np.int32),
            'RMSD': np.array([0.], dtype=np.float32),
            'lon': np.array([4.]),
            'p_tau': np.array([np.nan], dtype=np.float32),
            'BIAS': np.array([0.], dtype=np.float32),
            'p_rho': np.array([0.], dtype=np.float32),
            'rho': np.array([1.], dtype=np.float32),
            'lat': np.array([4.]),
            'R': np.array([1.], dtype=np.float32),
            'p_R': np.array([0.], dtype=np.float32)},
        (('DS1', 'x'), ('DS3', 'x')): {
            'n_obs': np.array([1000], dtype=np.int32),
            'tau': np.array([np.nan], dtype=np.float32),
            'gpi': np.array([4], dtype=np.int32),
            'RMSD': np.array([0.], dtype=np.float32),
            'lon': np.array([4.]),
            'p_tau': np.array([np.nan], dtype=np.float32),
            'BIAS': np.array([0.], dtype=np.float32),
            'p_rho': np.array([0.], dtype=np.float32),
            'rho': np.array([1.], dtype=np.float32),
            'lat': np.array([4.]),
            'R': np.array([1.], dtype=np.float32),
            'p_R': np.array([0.], dtype=np.float32)}}

    datasets = setup_TestDatasets()

    process = Validation(
        datasets, 'DS1',
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0).combinatory_matcher,
        scaling='lin_cdf_match',
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(other_name='k1').calc_metrics})

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n3_k2_masking():

    # test result for one gpi in a cell
    tst_results_one = {
        (('DS1', 'x'), ('DS3', 'y')): {
            'n_obs': np.array([250], dtype=np.int32)},
        (('DS1', 'x'), ('DS2', 'y')): {
            'n_obs': np.array([250], dtype=np.int32)},
        (('DS1', 'x'), ('DS3', 'x')): {
            'n_obs': np.array([250], dtype=np.int32)}}

    # test result for two gpis in a cell
    tst_results_two = {
        (('DS1', 'x'), ('DS3', 'y')): {
            'n_obs': np.array([250, 250], dtype=np.int32)},
        (('DS1', 'x'), ('DS2', 'y')): {
            'n_obs': np.array([250, 250], dtype=np.int32)},
        (('DS1', 'x'), ('DS3', 'x')): {
            'n_obs': np.array([250, 250], dtype=np.int32)}}

    # cell 4 in this example has two gpis so it returns different results.
    tst_results = {1: tst_results_one,
                   1: tst_results_one,
                   2: tst_results_two}

    datasets = setup_TestDatasets()

    # setup masking datasets

    grid = grids.CellGrid(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]),
                          np.array([4, 4, 2, 1]), gpis=np.array([1, 2, 3, 4]))

    mds1 = GriddedTsBase("", grid, MaskingTestDataset)
    mds2 = GriddedTsBase("", grid, MaskingTestDataset)

    mds = {
        'masking1': {
            'class': mds1,
            'columns': ['x'],
            'args': [],
            'kwargs': {'limit': 500},
            'use_lut': False,
            'grids_compatible': True},
        'masking2': {
            'class': mds2,
            'columns': ['x'],
            'args': [],
            'kwargs': {'limit': 750},
            'use_lut': False,
            'grids_compatible': True}
    }

    process = Validation(
        datasets, 'DS1',
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0).combinatory_matcher,
        scaling='lin_cdf_match',
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(other_name='k1').calc_metrics},
        masking_datasets=mds)

    gpi_info = (1, 1, 1)
    ref_df = datasets['DS1']['class'].read_ts(1)
    new_ref_df = process.mask_dataset(ref_df, gpi_info)
    assert len(new_ref_df) == 250
    nptest.assert_allclose(new_ref_df.x.values, np.arange(750, 1000))
    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        tst = tst_results[len(job[0])]
        assert sorted(list(results)) == sorted(list(tst))
        for key, tst_key in zip(sorted(results),
                                sorted(tst)):
            nptest.assert_almost_equal(results[key]['n_obs'],
                                       tst[tst_key]['n_obs'])


class TestDatasetRuntimeError(object):
    """Test dataset that acts as a fake object for the base classes."""

    def __init__(self, filename, mode='r', message="No such file or directory"):
        self.filename = filename
        self.mode = mode
        raise RuntimeError(message)
        open(filename, mode)

    def read(self, gpi):
        return None

    def write(self, gpi, data):
        return None

    def read_ts(self, gpi):
        return None

    def write_ts(self, gpi, data):
        return None

    def close(self):
        pass

    def flush(self):
        pass


def setup_TestDataManager():

    grid = grids.CellGrid(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]),
                          np.array([4, 4, 2, 1]), gpis=np.array([1, 2, 3, 4]))

    ds1 = GriddedTsBase("", grid, TestDatasetRuntimeError)
    ds2 = GriddedTsBase("", grid, TestDatasetRuntimeError)
    ds3 = GriddedTsBase("", grid, TestDatasetRuntimeError,
                        ioclass_kws={'message': 'Other RuntimeError'})

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['soil moisture'],
            'args': [],
            'kwargs': {}
        },
        'DS2': {
            'class': ds2,
            'columns': ['sm'],
            'args': [],
            'kwargs': {},
            'grids_compatible': True
        },
        'DS3': {
            'class': ds3,
            'columns': ['sm', 'sm2'],
            'args': [],
            'kwargs': {},
            'grids_compatible': True
        }
    }

    dm = DataManager(datasets, 'DS1')
    return dm


def test_DataManager_default_add():

    grid = grids.CellGrid(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]),
                          np.array([4, 4, 2, 1]), gpis=np.array([1, 2, 3, 4]))

    ds1 = GriddedTsBase("", grid, TestDataset)

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['soil moisture'],
        },
        'DS2': {
            'class': ds1,
            'columns': ['soil moisture'],
        }
    }

    dm = DataManager(datasets, 'DS1')
    assert dm.datasets == {
        'DS1': {
            'class': ds1,
            'columns': ['soil moisture'],
            'args': [],
            'kwargs': {},
            'use_lut': False,
            'lut_max_dist': None,
            'grids_compatible': False
        },
        'DS2': {
            'class': ds1,
            'columns': ['soil moisture'],
            'args': [],
            'kwargs': {},
            'use_lut': False,
            'lut_max_dist': None,
            'grids_compatible': False
        }}


def test_DataManager_read_ts_method_names():

    ds1 = TestDataset("")

    datasets = {
        'DS1': {
            'class': ds1,
            'columns': ['soil moisture'],
        },
        'DS2': {
            'class': ds1,
            'columns': ['soil moisture'],
        }
    }

    read_ts_method_names = {'DS1': 'read_ts',
                            'DS2': 'read_ts_other'}
    dm = DataManager(datasets, 'DS1',
                     read_ts_names=read_ts_method_names)
    data = dm.read_ds('DS1', 1)
    data_other = dm.read_ds('DS2', 1)
    pdtest.assert_frame_equal(data, ds1.read_ts(1))
    pdtest.assert_frame_equal(data_other, ds1.read_ts_other(1))


def test_DataManager_RuntimeError():
    """
    Test DataManager with some fake Datasets that throw RuntimeError
    instead of IOError if a file does not exist like netCDF4

    """

    dm = setup_TestDataManager()
    with pytest.warns(UserWarning):
        dm.read_reference(1)
    with pytest.warns(UserWarning):
        dm.read_other('DS2', 1)
    with pytest.raises(RuntimeError):
        dm.read_other('DS3', 1)


def test_DataManager_dataset_names():

    dm = setup_TestDataManager()
    result_names = dm.get_results_names(3)
    assert result_names == [(('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm')),
                            (('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm2'))]

    result_names = dm.get_results_names(2)
    assert result_names == [(('DS1', 'soil moisture'), ('DS2', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm2'))]


def test_DataManager_get_data():

    datasets = setup_TestDatasets()
    dm = DataManager(datasets, 'DS1')
    data = dm.get_data(1, 1, 1)
    assert sorted(list(data)) == ['DS1', 'DS2', 'DS3']


def test_get_result_names():

    tst_ds_dict = {'DS1': ['soil moisture'],
                   'DS2': ['sm'],
                   'DS3': ['sm', 'sm2']}
    result_names = get_result_names(tst_ds_dict, 'DS1', 3)
    assert result_names == [(('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm')),
                            (('DS1', 'soil moisture'), ('DS2', 'sm'), ('DS3', 'sm2'))]

    result_names = get_result_names(tst_ds_dict, 'DS1', 2)
    assert result_names == [(('DS1', 'soil moisture'), ('DS2', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm')),
                            (('DS1', 'soil moisture'), ('DS3', 'sm2'))]

    result_names = get_result_names(tst_ds_dict, 'DS2', 2)
    assert result_names == [(('DS2', 'sm'), ('DS1', 'soil moisture')),
                            (('DS2', 'sm'), ('DS3', 'sm')),
                            (('DS2', 'sm'), ('DS3', 'sm2'))]


def test_combinatory_matcher_n2():

    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5
    index = pd.date_range(start="2000-01-01", periods=n, freq="D")

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df2 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df3 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)

    df_dict = {'data1': df,
               'data2': df2,
               'data3': df3}

    temp_matcher = temporal_matchers.BasicTemporalMatching()
    matched = temp_matcher.combinatory_matcher(df_dict, 'data1')
    assert sorted(list(matched)) == sorted([('data1', 'data2'),
                                            ('data1', 'data3')])
    assert sorted(list(matched[('data1',
                                'data2')].columns)) == sorted([('data1', 'x'),
                                                               ('data1', 'y'),
                                                               ('data2', 'x'),
                                                               ('data2', 'y')])

    assert sorted(list(matched[('data1',
                                'data3')].columns)) == sorted([('data1', 'x'),
                                                               ('data1', 'y'),
                                                               ('data3', 'x'),
                                                               ('data3', 'y')])


def test_combinatory_matcher_n3():

    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5
    index = pd.date_range(start="2000-01-01", periods=n, freq="D")

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df2 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df3 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df4 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)

    df_dict = {'data1': df,
               'data2': df2,
               'data3': df3}

    temp_matcher = temporal_matchers.BasicTemporalMatching()
    matched = temp_matcher.combinatory_matcher(df_dict, 'data1', n=3)
    assert list(matched) == [('data1', 'data2', 'data3')]
    assert sorted(list(matched[('data1',
                                'data2',
                                'data3')].columns)) == sorted([('data1', 'x'),
                                                               ('data1', 'y'),
                                                               ('data2', 'x'),
                                                               ('data2', 'y'),
                                                               ('data3', 'x'),
                                                               ('data3', 'y')])

    df_dict = {'data1': df,
               'data2': df2,
               'data3': df3,
               'data4': df4}

    temp_matcher = temporal_matchers.BasicTemporalMatching()
    matched = temp_matcher.combinatory_matcher(df_dict, 'data1', n=3)
    assert sorted(list(matched)) == sorted([('data1', 'data2', 'data3'),
                                            ('data1', 'data2', 'data4'),
                                            ('data1', 'data3', 'data4')])
    assert sorted(list(matched[('data1',
                                'data2',
                                'data3')].columns)) == sorted([('data1', 'x'),
                                                               ('data1', 'y'),
                                                               ('data2', 'x'),
                                                               ('data2', 'y'),
                                                               ('data3', 'x'),
                                                               ('data3', 'y')])


def test_add_name_to_df_columns():

    n = 10
    x = np.arange(n)
    y = np.arange(n) * 0.5
    index = pd.date_range(start="2000-01-01", periods=n, freq="D")

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df = temporal_matchers.df_name_multiindex(df, 'test')
    assert list(df.columns) == [('test', 'x'), ('test', 'y')]


def test_args_to_iterable_non_iterables():

    gpis = 1
    lons = 1
    lats = 1
    arg1 = 1
    arg2 = 2
    arg3 = 3
    gpis_, lons_, lats_, args = args_to_iterable(gpis, lons, lats,
                                                 arg1, arg2, arg3, n=3)

    assert gpis_ == [gpis]
    assert lons_ == [lons]
    assert lats_ == [lats]
    assert args == ([arg1], [arg2], [arg3])


def test_args_to_iterable_n3():

    gpis = [1, 2, 3]
    lons = [2, 3, 4]
    lats = [3, 4, 5]
    arg1 = [1, 1, 1]
    arg2 = [1, 1, 1]
    gpis_, lons_, lats_, args = args_to_iterable(gpis, lons, lats,
                                                 arg1, arg2, n=3)

    assert gpis_ == gpis
    assert lons_ == lons
    assert lats_ == lats
    assert args == (arg1, arg2)

    zipped_should = [(1, 2, 3, 1, 1),
                     (2, 3, 4, 1, 1),
                     (3, 4, 5, 1, 1)]

    for i, t in enumerate(zip(gpis_, lons_, lats_, *args)):
        assert zipped_should[i] == t


def test_args_to_iterable_mixed():

    gpis = [1, 2, 3]
    lons = [2, 3, 4]
    lats = 1
    arg1 = 1
    gpis_, lons_, lats_, args = args_to_iterable(gpis, lons, lats,
                                                 arg1)

    assert gpis_ == gpis
    assert lons_ == lons
    assert lats_ == [lats]
    assert args == [arg1]
