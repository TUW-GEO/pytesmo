# Copyright (c) 2017,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without specific
#     prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''
Tests for the data scalers.
'''

from pytesmo.validation_framework.data_scalers import CDFStoreParamsScaler
from pygeogrids.grids import genreg_grid
import numpy as np
import pandas as pd


def get_dataframe_2_col():
    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'])
    return df


def get_dataframe_3_col():
    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5
    z = np.arange(n) * 0.2 + 5

    df = pd.DataFrame({'x': x, 'y': y, 'z': z}, columns=['x', 'y', 'z'])
    return df


def test_CDF_storage_scaler_scale_df_2_columns(tmpdir):

    df = get_dataframe_2_col()
    path = str(tmpdir)
    grid = genreg_grid(1, 1).to_cell_grid(10, 10)
    scaler = CDFStoreParamsScaler(path, grid)
    scaled_data = scaler.scale(df, 0, (1, 10, 10))
    np.testing.assert_almost_equal(scaled_data['x'].values,
                                   scaled_data['y'].values)


def test_CDF_storage_scaler_scale_df_3_columns(tmpdir):

    df = get_dataframe_3_col()
    path = str(tmpdir)
    grid = genreg_grid(1, 1).to_cell_grid(10, 10)
    scaler = CDFStoreParamsScaler(path, grid)
    scaled_data = scaler.scale(df, 0, (1, 10, 10))
    np.testing.assert_almost_equal(scaled_data['x'].values,
                                   scaled_data['y'].values)
    np.testing.assert_almost_equal(scaled_data['x'].values,
                                   scaled_data['z'].values)


def test_CDF_storage_scaler_scale_df_3_columns_load_precomputed(tmpdir):

    df = get_dataframe_3_col()
    path = str(tmpdir)
    grid = genreg_grid(1, 1).to_cell_grid(10, 10)
    scaler = CDFStoreParamsScaler(path, grid)
    scaler.get_parameters(df, 0, 1)
    scaler.get_parameters(df, 0, 100)
    scaled_data = scaler.scale(df, 0, (1, 10, 10))
    np.testing.assert_almost_equal(scaled_data['x'].values,
                                   scaled_data['y'].values)
    np.testing.assert_almost_equal(scaled_data['x'].values,
                                   scaled_data['z'].values)


def test_CDF_storage_scaler_calc_parameters():

    df = get_dataframe_2_col()
    path = 'test'
    grid = genreg_grid(1, 1).to_cell_grid(10, 10)
    scaler = CDFStoreParamsScaler(path, grid)
    parameters = scaler.calc_parameters(df, 0)
    parameters_should = {'y_x': np.array([
        [0., 24.75, 49.75, 149.75, 249.75, 349.75, 449.75, 474.75, 499.5],
        [0., 49.5, 99.5, 299.5, 499.5, 699.5, 899.5, 949.5, 999.]
    ])}
    assert sorted(
        list(parameters_should.keys())) == sorted(list(parameters.keys()))
    for key in parameters_should:
        np.testing.assert_allclose(parameters[key],
                                   parameters_should[key], atol=1e-10)


def test_CDF_storage_scaler_store_load_parameters(tmpdir):

    path = str(tmpdir)
    grid = genreg_grid(1, 1).to_cell_grid(10, 10)
    scaler = CDFStoreParamsScaler(path, grid)
    data_written = {'y_x': np.array([
        [0., 24.75, 49.75, 149.75, 249.75, 349.75, 449.75, 474.75, 499.5],
        [0., 49.5, 99.5, 299.5, 499.5, 699.5, 899.5, 949.5, 999.]
    ])}
    scaler.store_parameters(1, data_written)
    params_loaded = scaler.load_parameters(1)
    assert sorted(
        list(data_written.keys())) == sorted(list(params_loaded.keys()))
    for key in params_loaded:
        np.testing.assert_allclose(params_loaded[key],
                                   data_written[key], rtol=1e-10)


def test_CDF_storage_scaler_get_parameters(tmpdir):

    df = get_dataframe_2_col()
    path = str(tmpdir)
    grid = genreg_grid(1, 1).to_cell_grid(10, 10)
    scaler = CDFStoreParamsScaler(path, grid)
    data_written = {'y_x': np.array([
        [0., 24.75, 49.75, 149.75, 249.75, 349.75, 449.75, 474.75, 499.5],
        [0., 49.5, 99.5, 299.5, 499.5, 699.5, 899.5, 949.5, 999.]
    ])}
    params_loaded = scaler.get_parameters(df, 0, 1)
    assert sorted(
        list(data_written.keys())) == sorted(list(params_loaded.keys()))
    for key in params_loaded:
        np.testing.assert_allclose(params_loaded[key],
                                   data_written[key], atol=1e-10)

    # ensure that get_parameters has also stored them
    params_loaded = scaler.load_parameters(1)
    assert sorted(
        list(data_written.keys())) == sorted(list(params_loaded.keys()))
    for key in params_loaded:
        np.testing.assert_allclose(params_loaded[key],
                                   data_written[key], atol=1e-10)
