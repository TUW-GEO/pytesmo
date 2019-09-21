# -*- coding: utf-8 -*-

'''
Module description
'''
# TODO:
#   (+) 
#---------
# NOTES:
#   -

"""
Created on 01.06.2015
@author: Andreea Plocon andreea.plocon@geo.tuwien.ac.at
"""

import os
import netCDF4

from datetime import datetime
import pandas as pd
import numpy as np
import copy
import warnings
import xarray as xr
from collections import OrderedDict

def build_filename(root, key):
    """
    Create savepath/filename that does not exceed 255 characters

    Parameters
    ----------
    root : str
        Directory where the file should be stored
    key : list of tuples
        The keys are joined to create a filename from them. If the length of the
        joined keys is too long we shorten it.

    Returns
    -------
    fname : str
        Full path to the netcdf file to store
    """
    ds_names = []
    for ds in key:
        if isinstance(ds, tuple):
            ds_names.append('.'.join(ds))
        else:
            ds_names.append(ds)

    fname = '_with_'.join(ds_names)
    ext = 'nc'

    if len(os.path.join(root, '.'.join([fname, ext]))) > 255:
        ds_names = [str(ds[0]) for ds in key]
        fname = '_with_'.join(ds_names)

        if len(os.path.join(root, '.'.join([fname, ext]))) > 255:
            fname = 'validation'

    return os.path.join(root, '.'.join([fname, ext]))

def netcdf_results_manager(results, save_path, zlib=True):
    """
    Function for writing the results of the validation process as NetCDF file.

    Parameters
    ----------
    results : dict of dicts
        Keys: Combinations of (referenceDataset.column, otherDataset.column)
        Values: dict containing the results from metric_calculator
    save_path : string
        Path where the file/files will be saved.
    """
    for key in results.keys():
        fname = build_filename(save_path, key)
        filename = os.path.join(save_path, fname)
        if not os.path.exists(filename):
            ncfile = netCDF4.Dataset(filename, 'w')

            global_attr = {}
            s = "%Y-%m-%d %H:%M:%S"
            global_attr['date_created'] = datetime.now().strftime(s)
            ncfile.setncatts(global_attr)

            ncfile.createDimension('dim', None)
        else:
            ncfile = netCDF4.Dataset(filename, 'a')

        index = len(ncfile.dimensions['dim'])
        for field in results[key]:

            if field in ncfile.variables.keys():
                var = ncfile.variables[field]
            else:
                var_type = results[key][field].dtype
                kwargs = {'fill_value': -99999}
                # if dtype is a object the assumption is that the data is a
                # string
                if var_type == object:
                    var_type = str
                    kwargs = {}

                if zlib:
                    kwargs['zlib'] = True,
                    kwargs['complevel'] = 6

                var = ncfile.createVariable(field, var_type,
                                            'dim', **kwargs)
            var[index:] = results[key][field]

        ncfile.close()


class NcResultsManager(object):
    """
    Stores validation results on a regular or irregular grid.
    """
    # def __enter__(self):
    #     return self
    # 
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     pass


    def __init__(self, save_path, glob_attrs=None, var_attrs=None, zlib=True):
        '''
        Create a data frame from the jobs.
        Add options for compression and for buffer storing?

        Parameters
        -------
        save_path : str
            Root folder where we store the results.
        zlib : bool, optional (default: True)
            Activate compression of all results.
        '''

        self.save_path = save_path
        self.zlib = zlib
        self.dfs = dict()

    def add(self, results, lat_name='lat', lon_name='lon'):
        """
        Add results to the respective job in the data frame
        Returns
        -------

        """
        for k, res in results.items():
            if k not in self.dfs.keys():
                self.dfs[k] = pd.DataFrame()
            gpi_results = pd.DataFrame.from_dict(results[k]).set_index([lat_name, lon_name])
            self.dfs[k] = pd.concat([self.dfs[k], gpi_results], axis=0)

    def _global_attr(self, global_attr=None):
        """
        Create global attributes that are passed when writing a netcdf file.

        Parameters
        ----------
        attr: dict, optional (default: None)
            Attributes that the user passed. Will be added to the file

        Returns
        -------
        global_attr : dict
            Global attributes for the netcdf results file.
        """
        if global_attr is None:
            global_attr = {}
        else:
            global_attr = copy.deepcopy(global_attr)

        s = "%Y-%m-%d %H:%M:%S"
        global_attr['date_created'] = datetime.now().strftime(s)
        return global_attr

    def to_netcdf(self, glob_attr=None, var_attr=None):
        """
        Write the data from memory to disk as a netcdf file.

        ---------
        """
        for key, df in self.dfs.items():
            try:
                filename = build_filename(self.save_path, key)
                dataset = df.to_xarray() # type: xr.Dataset

                dataset = dataset.assign_attrs(self._global_attr(glob_attr))

                if var_attr is not None:
                    for varname, var_meta in var_attr.items():
                        if varname in dataset.variables:
                            if isinstance(var_meta, dict):
                                var_meta = OrderedDict(var_meta)
                            dataset[varname].attrs = var_meta

                try:
                    if self.zlib:
                        encoding = {}
                        for var in dataset.variables:
                            if var not in ['lat', 'lon']:
                                encoding[var] = {'complevel': 6, 'zlib': True}
                    else:
                        encoding = None
                    dataset.to_netcdf(filename, engine='netcdf4', encoding=encoding)
                except:
                    warnings.warn('Compression failed, store uncompressed results.')
                    dataset.to_netcdf(filename, engine='netcdf4')

                dataset.close()
            except:
                data = {}
                for col in df:
                    data[col] = df[col].values
                data['lat'] = df.index.get_level_values('lat').values
                data['lon'] = df.index.get_level_values('lon').values
                data = {key: data}
                netcdf_results_manager(data, self.save_path, zlib=self.zlib)


if __name__ == '__main__':
    gpis = list(range(1,10))
    lat = [30] + list(range(30,30+len(gpis)-1,1))
    lon = [-119]+list(range(-119,-119+len(gpis)-1,1))
    n_obs = np.random.randint(0,1000,len(gpis))
    s = ['s%i' %i for i in gpis]
    n = np.random.random_sample(len(lon))


    results = {('test1','test2') : dict(lat=lat, lon=lon, gpi=gpis, n_obs=n_obs, s=s, n=n)}

    path = r'C:\Temp\nc_compress'

    var_attrs = {'n_obs': {'name':'Number of Observations', 'smthg_else':1}}

    results_manager = NcResultsManager(save_path=path, var_attrs=var_attrs,
                                       glob_attrs={'global': 'Test'})
    with results_manager as writer:
        writer.add(results)
