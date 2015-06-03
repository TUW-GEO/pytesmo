"""
Created on 01.06.2015
@author: Andreea Plocon andreea.plocon@geo.tuwien.ac.at
"""

import os
import netCDF4

from datetime import datetime


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
        filename = os.path.join(save_path, key[0] + '_with_' + key[1] + '.nc')
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
                var = ncfile.createVariable(field, results[key][field].dtype,
                                            'dim', fill_value=-99999)
            var[index:] = results[key][field]

        ncfile.close()
