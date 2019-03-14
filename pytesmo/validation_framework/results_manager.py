"""
Created on 01.06.2015
@author: Andreea Plocon andreea.plocon@geo.tuwien.ac.at
"""

import os
import netCDF4

from datetime import datetime


def build_filename(key):
    ''' Create a filename that does not exceed the max number of characters (144)?'''
    ds_names = ['.'.join(ds) for ds in key]
    fname = '_with_'.join(ds_names) + '.nc'
    if len(fname) > 144:
        ds_names = [str(ds[0]) for ds in key]
        fname = '_with_'.join(ds_names) + '.nc'
    if len(fname) > 144:
        fname = 'validation.nc'
    return fname

def netcdf_results_manager(results, save_path):
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
        fname = build_filename(key)
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
                var = ncfile.createVariable(field, var_type,
                                            'dim', **kwargs)
            var[index:] = results[key][field]

        ncfile.close()
