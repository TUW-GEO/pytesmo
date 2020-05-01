# -*- coding: utf-8 -*-

"""
The results manager stores validation results in netcdf format.
"""

import os
import netCDF4
from datetime import datetime

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
