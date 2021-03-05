# Copyright (c) 2016,Vienna University of Technology,
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
Tests for the results managers.
'''

import tempfile
import numpy as np
import os
import netCDF4
import pandas as pd

from pytesmo.validation_framework.results_manager import netcdf_results_manager, PointDataResults


def test_netcdf_result_manager_n2():

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

    tempdir = tempfile.mkdtemp()
    netcdf_results_manager(tst_results, tempdir)
    assert sorted(os.listdir(tempdir)) == sorted(['DS1.x_with_DS3.x.nc',
                                                  'DS1.x_with_DS3.y.nc',
                                                  'DS1.x_with_DS2.y.nc'])

    # check a few variable in the file
    with netCDF4.Dataset(os.path.join(tempdir, 'DS1.x_with_DS3.x.nc')) as ds:
        assert ds.variables['lon'][:] == np.array([4])
        assert ds.variables['n_obs'][:] == np.array([1000])


def test_netcdf_result_manager_n3():

    tst_results = {
        (('DS1', 'x'), ('DS2', 'y'), ('DS3', 'x')): {
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
        (('DS1', 'x'), ('DS2', 'y'), ('DS3', 'y')): {
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

    tempdir = tempfile.mkdtemp()
    netcdf_results_manager(tst_results, tempdir)
    assert (
        sorted(os.listdir(tempdir))
        == sorted(['DS1.x_with_DS2.y_with_DS3.x.nc',
                   'DS1.x_with_DS2.y_with_DS3.y.nc'])
    )

    # check a few variable in the file
    with netCDF4.Dataset(
            os.path.join(tempdir, 'DS1.x_with_DS2.y_with_DS3.x.nc')
    ) as ds:
        assert ds.variables['lon'][:] == np.array([4])
        assert ds.variables['n_obs'][:] == np.array([1000])


def test_netcdf_results_manager_ts():
    results = {
        (('a1', 'x1'), ('b1', 'y1')): {
            'time': np.array([
                pd.date_range('2000-01-01', '2000-01-03', freq='D'),
                pd.date_range('2000-05-01', '2000-05-05', freq='D')
            ], dtype="object"),
            'tvar': np.array([
                np.array([1, 2, 3]),
                np.array([1, 2, 3, 4, 5])
            ], dtype="object"),
            'lvar': np.array([99, 100]),
            'lon': np.array([1., 2.]),
            'lat': np.array([1., 2.])
        },
        (('a2', 'x2'), ('b2', 'y2')): {
            'time': np.array([
                pd.date_range('2003-01-01', '2003-01-02', freq='D')]),
            'tvar': np.array([np.array([1, 2])]),
            'lvar': np.array([99]),
            'lon': np.array([1.]),
            'lat': np.array([1.])
        }
    }

    tempdir = tempfile.mkdtemp()
    netcdf_results_manager(
        results=results, save_path=tempdir, ts_vars=['tvar'],
        attr={'tvar': {'long_name': 'Time var'},
              'lvar': {'long_name': 'Loc var'}}
    )

    ds = PointDataResults(
        os.path.join(tempdir, 'a1.x1_with_b1.y1.nc'), read_only=True
    )
    ts = ds.read_ts(0)
    assert ts.loc['2000-01-02', 'tvar'] == 2
    df = ds.read_loc(None)
    assert np.all(df.loc[0, :] == ds.read_loc(0))
    assert df.loc[1, 'lvar'] == 100
