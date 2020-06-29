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
import pytest

import pygeogrids.grids as grids
from pygeobase.io_base import GriddedTsBase

import pytesmo.validation_framework.temporal_matchers as temporal_matchers
import pytesmo.validation_framework.metric_calculators as metrics_calculators
from pytesmo.validation_framework.results_manager import netcdf_results_manager
from pytesmo.validation_framework.data_manager import DataManager
from pytesmo.validation_framework.results_manager import PointDataResults

from datetime import datetime

from ascat.read_native.cdr import AscatSsmCdr
from ismn.interface import ISMN_Interface

from pytesmo.validation_framework.validation import Validation
from pytesmo.validation_framework.validation import args_to_iterable

from tests.test_validation_framwork.test_datasets import setup_TestDatasets
from tests.test_validation_framwork.test_datasets import setup_two_without_overlap
from tests.test_validation_framwork.test_datasets import setup_three_with_two_overlapping
from tests.test_validation_framwork.test_datasets import MaskingTestDataset

import warnings

@pytest.mark.full_framework
def test_ascat_ismn_validation():
    """
    Test processing framework with some ISMN and ASCAT sample data
    """
    ascat_data_folder = os.path.join(os.path.dirname(__file__), '..', 'test-data',
                                     'sat', 'ascat', 'netcdf', '55R22')

    ascat_grid_folder = os.path.join(os.path.dirname(__file__), '..', 'test-data',
                                     'sat', 'ascat', 'netcdf', 'grid')

    static_layers_folder = os.path.join(os.path.dirname(__file__),
                                        '..', 'test-data', 'sat',
                                        'h_saf', 'static_layer')

    ascat_reader = AscatSsmCdr(ascat_data_folder, ascat_grid_folder,
                               grid_filename='TUW_WARP5_grid_info_2_1.nc',
                               static_layer_path=static_layers_folder)
    ascat_reader.read_bulk = True

    # Initialize ISMN reader

    ismn_data_folder = os.path.join(os.path.dirname(__file__), '..', 'test-data',
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

    read_ts_names = {'ASCAT': 'read', 'ISMN': 'read_ts'}
    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

    datasets = DataManager(datasets, 'ISMN', period, read_ts_names=read_ts_names)

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
                   u'BIAS', u'p_rho', u'rho', u'lat', u'R', u'p_R', u'time',
                   u'idx', u'_row_size']
    n_obs_should = [384,  357,  482,  141,  251, 1927, 1887, 1652]
    rho_should = np.array([0.70022893, 0.53934574,
                           0.69356072, 0.84189808,
                           0.74206454, 0.30299741,
                           0.53143877, 0.62204134], dtype=np.float32)

    rmsd_should = np.array([7.72966719, 11.58347607,
                            14.57700157, 13.06224251,
                            12.90389824, 14.24668026,
                            21.19682884, 17.3883934], dtype=np.float32)
    with nc.Dataset(results_fname, mode='r') as results:
        assert sorted(list(results.variables.keys())) == sorted(vars_should)
        assert sorted(results.variables['n_obs'][:].tolist()) == sorted(
            n_obs_should)
        nptest.assert_allclose(sorted(rho_should),
                               sorted(results.variables['rho'][:]),
                               rtol=1e-4)
        nptest.assert_allclose(sorted(rmsd_should),
                               sorted(results.variables['RMSD'][:]),
                               rtol=1e-4)

@pytest.mark.full_framework
def test_ascat_ismn_validation_metadata():
    """
    Test processing framework with some ISMN and ASCAT sample data
    """
    ascat_data_folder = os.path.join(os.path.dirname(__file__), '..', 'test-data',
                                     'sat', 'ascat', 'netcdf', '55R22')

    ascat_grid_folder = os.path.join(os.path.dirname(__file__), '..', 'test-data',
                                     'sat', 'ascat', 'netcdf', 'grid')

    static_layers_folder = os.path.join(os.path.dirname(__file__),
                                        '..', 'test-data', 'sat',
                                        'h_saf', 'static_layer')

    ascat_reader = AscatSsmCdr(ascat_data_folder, ascat_grid_folder,
                               grid_filename='TUW_WARP5_grid_info_2_1.nc',
                               static_layer_path=static_layers_folder)
    ascat_reader.read_bulk = True

    # Initialize ISMN reader

    ismn_data_folder = os.path.join(os.path.dirname(__file__), '..', 'test-data',
                                    'ismn', 'multinetwork', 'header_values')
    ismn_reader = ISMN_Interface(ismn_data_folder)

    jobs = []

    ids = ismn_reader.get_dataset_ids(
        variable='soil moisture',
        min_depth=0,
        max_depth=0.1)

    metadata_dict_template = {'network': np.array(['None'], dtype='U256'),
                              'station': np.array(['None'], dtype='U256'),
                              'landcover': np.float32([np.nan]),
                              'climate': np.array(['None'], dtype='U4')}

    for idx in ids:
        metadata = ismn_reader.metadata[idx]
        metadata_dict = [{'network': metadata['network'],
                          'station': metadata['station'],
                          'landcover': metadata['landcover_2010'],
                          'climate': metadata['climate']}]
        jobs.append((idx, metadata['longitude'],
                     metadata['latitude'], metadata_dict))

    # Create the variable ***save_path*** which is a string representing the
    # path where the results will be saved. **DO NOT CHANGE** the name
    # ***save_path*** because it will be searched during the parallel
    # processing!

    save_path = tempfile.mkdtemp()

    # Create the validation object.

    datasets = {
        'ISMN': {
            'class': ismn_reader,
            'columns': ['soil moisture'],
        },
        'ASCAT': {
            'class': ascat_reader,
            'columns': ['sm'],
            'kwargs': {'mask_frozen_prob': 80,
                       'mask_snow_prob': 80,
                       'mask_ssf': True},
        }}

    read_ts_names = {'ASCAT': 'read', 'ISMN': 'read_ts'}
    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

    datasets = DataManager(datasets, 'ISMN', period, read_ts_names=read_ts_names)
    process = Validation(
        datasets, 'ISMN',
        temporal_ref='ASCAT',
        scaling='lin_cdf_match',
        scaling_ref='ASCAT',
        metrics_calculators={
            (2, 2): metrics_calculators.BasicMetrics(other_name='k1', metadata_template=metadata_dict_template).calc_metrics},
        period=period)

    for job in jobs:
        results = process.calc(*job)
        netcdf_results_manager(results, save_path)

    results_fname = os.path.join(
        save_path, 'ASCAT.sm_with_ISMN.soil moisture.nc')

    vars_should = [u'n_obs', u'tau', u'gpi', u'RMSD', u'lon', u'p_tau',
                   u'BIAS', u'p_rho', u'rho', u'lat', u'R', u'p_R', u'time',
                   u'idx', u'_row_size']
    for key, value in metadata_dict_template.items():
        vars_should.append(key)

    n_obs_should = [384,  357,  482,  141,  251, 1927, 1887, 1652]
    rho_should = np.array([0.70022893, 0.53934574,
                           0.69356072, 0.84189808,
                           0.74206454, 0.30299741,
                           0.53143877, 0.62204134], dtype=np.float32)

    rmsd_should = np.array([7.72966719, 11.58347607,
                            14.57700157, 13.06224251,
                            12.90389824, 14.24668026,
                            21.19682884, 17.3883934], dtype=np.float32)

    network_should = np.array(['MAQU', 'MAQU', 'SCAN', 'SCAN', 'SCAN',
                               'SOILSCAPE', 'SOILSCAPE', 'SOILSCAPE'], dtype='U256')

    with nc.Dataset(results_fname, mode='r') as results:
        assert sorted(results.variables.keys()) == sorted(vars_should)
        assert sorted(results.variables['n_obs'][:].tolist()) == sorted(
            n_obs_should)

        nptest.assert_allclose(sorted(rho_should),
                               sorted(results.variables['rho'][:]),
                               rtol=1e-4)
        nptest.assert_allclose(sorted(rmsd_should),
                               sorted(results.variables['RMSD'][:]),
                               rtol=1e-4)
        nptest.assert_equal(sorted(network_should),
                            sorted(results.variables['network'][:]))


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

    dm = DataManager(datasets, 'DS1', read_ts_names={d: 'read' for d in ['DS1', 'DS2', 'DS3']})

    process = Validation(
        dm, 'DS1',
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0).combinatory_matcher,
        scaling='lin_cdf_match',
        metrics_calculators={
            (2, 2): metrics_calculators.BasicMetrics(other_name='k1').calc_metrics})

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n2_k2_temporal_matching_no_matches():

    tst_results = {}

    datasets = setup_two_without_overlap()

    dm = DataManager(datasets, 'DS1', read_ts_names={d: 'read' for d in ['DS1', 'DS2', 'DS3']})


    process = Validation(
        dm, 'DS1',
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0).combinatory_matcher,
        scaling='lin_cdf_match',
        metrics_calculators={
            (2, 2): metrics_calculators.BasicMetrics(other_name='k1').calc_metrics})

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n2_k2_data_manager_argument():

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
    dm = DataManager(datasets, 'DS1', read_ts_names={d: 'read' for d in ['DS1', 'DS2', 'DS3']})

    process = Validation(dm, 'DS1',
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
    dm = DataManager(datasets, 'DS1', read_ts_names={d: 'read' for d in ['DS1', 'DS2', 'DS3']})

    process = Validation(
        dm, 'DS1',
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0).combinatory_matcher,
        scaling='lin_cdf_match',
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(other_name='k1').calc_metrics})

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n3_k2_temporal_matching_no_matches():

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

    datasets = setup_three_with_two_overlapping()
    dm = DataManager(datasets, 'DS1', read_ts_names={d: 'read' for d in ['DS1', 'DS2', 'DS3']})

    process = Validation(
        dm, 'DS1',
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0).combinatory_matcher,
        scaling='lin_cdf_match',
        metrics_calculators={
            (2, 2): metrics_calculators.BasicMetrics(other_name='k1').calc_metrics})

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n3_k2_masking_no_data_remains():

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
            'kwargs': {'limit': 1000},
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
    ref_df = datasets['DS1']['class'].read(1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        new_ref_df = process.mask_dataset(ref_df, gpi_info)
    assert len(new_ref_df) == 0
    nptest.assert_allclose(new_ref_df.x.values, np.arange(1000, 1000))
    jobs = process.get_processing_jobs()
    for job in jobs:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            results = process.calc(*job)
        tst = []
        assert sorted(list(results)) == sorted(list(tst))
        for key, tst_key in zip(sorted(results),
                                sorted(tst)):
            nptest.assert_almost_equal(results[key]['n_obs'],
                                       tst[tst_key]['n_obs'])


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
    ref_df = datasets['DS1']['class'].read(1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning) # read_ts is hard coded when using mask_data
        new_ref_df = process.mask_dataset(ref_df, gpi_info)
    assert len(new_ref_df) == 250
    nptest.assert_allclose(new_ref_df.x.values, np.arange(750, 1000))
    jobs = process.get_processing_jobs()
    for job in jobs:

        with warnings.catch_warnings():
            # most warnings here are caused by the read_ts function that cannot
            # be changed when using a masking data set
            warnings.simplefilter('ignore', category=DeprecationWarning)
            results = process.calc(*job)

        tst = tst_results[len(job[0])]
        assert sorted(list(results)) == sorted(list(tst))
        for key, tst_key in zip(sorted(results),
                                sorted(tst)):
            nptest.assert_almost_equal(results[key]['n_obs'],
                                       tst[tst_key]['n_obs'])

@pytest.mark.full_framework
def test_ascat_ismn_validation_metadata_rolling():
    """
    Test processing framework with some ISMN and ASCAT sample data
    """
    ascat_data_folder = os.path.join(os.path.dirname(__file__), '..', 'test-data',
                                     'sat', 'ascat', 'netcdf', '55R22')

    ascat_grid_folder = os.path.join(os.path.dirname(__file__), '..', 'test-data',
                                     'sat', 'ascat', 'netcdf', 'grid')

    static_layers_folder = os.path.join(os.path.dirname(__file__),
                                        '..', 'test-data', 'sat',
                                        'h_saf', 'static_layer')

    ascat_reader = AscatSsmCdr(ascat_data_folder, ascat_grid_folder,
                               grid_filename='TUW_WARP5_grid_info_2_1.nc',
                               static_layer_path=static_layers_folder)
    ascat_reader.read_bulk = True

    # Initialize ISMN reader

    ismn_data_folder = os.path.join(os.path.dirname(__file__), '..', 'test-data',
                                    'ismn', 'multinetwork', 'header_values')
    ismn_reader = ISMN_Interface(ismn_data_folder)

    jobs = []

    ids = ismn_reader.get_dataset_ids(
        variable='soil moisture',
        min_depth=0,
        max_depth=0.1)

    metadata_dict_template = {'network': np.array(['None'], dtype='U256'),
                              'station': np.array(['None'], dtype='U256'),
                              'landcover': np.float32([np.nan]),
                              'climate': np.array(['None'], dtype='U4')}

    for idx in ids:
        metadata = ismn_reader.metadata[idx]
        metadata_dict = [{'network': metadata['network'],
                          'station': metadata['station'],
                          'landcover': metadata['landcover_2010'],
                          'climate': metadata['climate']}]
        jobs.append((idx, metadata['longitude'],
                     metadata['latitude'], metadata_dict))

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

    read_ts_names = {'ASCAT': 'read', 'ISMN': 'read_ts'}
    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

    datasets = DataManager(datasets, 'ISMN', period, read_ts_names=read_ts_names)

    process = Validation(
        datasets, 'ISMN',
        temporal_ref='ASCAT',
        scaling='lin_cdf_match',
        scaling_ref='ASCAT',
        metrics_calculators={
            (2, 2): metrics_calculators.RollingMetrics(other_name='k1',
                                                       metadata_template=metadata_dict_template).calc_metrics},
        period=period)

    for job in jobs:
        results = process.calc(*job)
        netcdf_results_manager(results, save_path, ts_vars=[
                               'R', 'p_R', 'RMSD'])

    results_fname = os.path.join(
        save_path, 'ASCAT.sm_with_ISMN.soil moisture.nc')

    vars_should = [u'gpi', u'lon',  u'lat', u'R', u'p_R', u'time',
                   u'idx', u'_row_size']

    for key, value in metadata_dict_template.items():
        vars_should.append(key)

    network_should = np.array(['MAQU', 'MAQU', 'SCAN', 'SCAN', 'SCAN',
                               'SOILSCAPE', 'SOILSCAPE', 'SOILSCAPE'], dtype='U256')

    reader = PointDataResults(results_fname, read_only=True)
    df = reader.read_loc(None)
    nptest.assert_equal(sorted(network_should), sorted(df['network'].values))
    assert np.all(df.gpi.values == np.arange(8))
    assert(reader.read_ts(0).index.size == 357)
    assert np.all(reader.read_ts(1).columns.values ==
                  np.array(['R', 'p_R', 'RMSD']))

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


def test_args_to_iterable_mixed_strings():

    gpis = [1, 2, 3]
    lons = [2, 3, 4]
    lats = 1
    arg1 = 'test'
    gpis_, lons_, lats_, args = args_to_iterable(gpis, lons, lats,
                                                 arg1)

    assert gpis_ == gpis
    assert lons_ == lons
    assert lats_ == [lats]
    assert args == [arg1]