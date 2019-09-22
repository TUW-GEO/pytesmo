# -*- coding: utf-8 -*-

'''
Module description
'''
# TODO:
#   (+) 
#---------
# NOTES:
#   -

import tempfile
import numpy as np
import os
import netCDF4
import shutil

from pytesmo.validation_framework.results_manager import netcdf_results_manager
from local_scripts.new_results_manager import *


def test_netcdf_result_manager_new_n2():

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
    manag = NcResultsManager(tempdir)
    manag.add(tst_results)
    var_attr = {'R':{'name': 'Pearson'}}
    glob_attr = {'test_attribute': 'test'}
    manag.to_netcdf(var_attr=var_attr, glob_attr=glob_attr)
    assert sorted(os.listdir(tempdir)) == sorted(['DS1.x_with_DS3.x.nc',
                                                  'DS1.x_with_DS3.y.nc',
                                                  'DS1.x_with_DS2.y.nc'])

    # check a few variable in the file
    with netCDF4.Dataset(os.path.join(tempdir, 'DS1.x_with_DS3.x.nc')) as ds:
        assert ds.variables['lon'][:] == np.array([4])
        assert ds.variables['n_obs'][:] == np.array([1000])
        assert(ds.variables['R'].getncattr('name') == 'Pearson')
        assert(ds.getncattr('test_attribute') == 'test')


def test_netcdf_result_manager_new_n3():

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
    manag = NcResultsManager(tempdir)
    manag.add(tst_results)
    var_attr = {'R':{'name': 'Pearson'}}
    glob_attr = {'test_attribute': 'test'}
    manag.to_netcdf(var_attr=var_attr, glob_attr=glob_attr)
    assert sorted(os.listdir(tempdir)) == sorted(['DS1.x_with_DS2.y_with_DS3.x.nc',
                                                  'DS1.x_with_DS2.y_with_DS3.y.nc'])

    # check a few variable in the file
    with netCDF4.Dataset(os.path.join(tempdir, 'DS1.x_with_DS2.y_with_DS3.x.nc')) as ds:
        assert ds.variables['lon'][:] == np.array([4])
        assert ds.variables['n_obs'][:] == np.array([1000])
        assert(ds.variables['R'].getncattr('name') == 'Pearson')
        assert(ds.getncattr('test_attribute') == 'test')