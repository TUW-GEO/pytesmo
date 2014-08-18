# Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

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

'''
Created on Jan 21, 2014

Module for saving grid to netCDF

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''
from netCDF4 import Dataset
import numpy as np
import os
from datetime import datetime

from pytesmo.grid.grids import CellGrid, BasicGrid


def save_lonlat(filename, arrlon, arrlat, arrcell=None,
                gpis=None, subset_points=None, subset_name='land_points',
                subset_meaning="water land",
                global_attrs=None):
    """
    saves grid information to netCDF file

    Parameters
    ----------
    filename : string
        name of file
    arrlon : numpy.array
        array of longitudes
    arrlat : numpy.array
        array of latitudes
    arrcell : numpy.array, optional
        array of cell numbers
    gpis : numpy.array, optional
        gpi numbers if not index of arrlon, arrlat
    subset_points : numpy.array, optional
        and indication if the given grid point is over land
        must be an indices into arrlon, arrlat
    subset_name : string, optional
        long_name of the variable 'subset_flag'
        if the subset symbolises something other than a land/sea mask
    subset_meaning : string, optional
        will be written into flag_meanings metadata of variable 'subset_flag'
    global_attrs : dict, optional
        if given will be written as global attributs into netCDF file
    """

    nc_name = filename

    with Dataset(nc_name, 'w', format='NETCDF4') as ncfile:

        if (global_attrs is not None and 'shape' in global_attrs and
            type(global_attrs['shape']) is not int):

            latsize = global_attrs['shape'][0]
            lonsize = global_attrs['shape'][1]
            ncfile.createDimension("lat", latsize)
            ncfile.createDimension("lon", lonsize)
            arrlat = np.unique(arrlat)[::-1]  # sorts arrlat descending
            arrlon = np.unique(arrlon)

            gpisize = global_attrs['shape'][0] * global_attrs['shape'][1]
            if gpis is None:
                gpivalues = np.arange(gpisize,
                                      dtype=np.int32).reshape(latsize,
                                                              lonsize)
                gpivalues = gpivalues[::-1]
            else:
                gpivalues = gpis.reshape(latsize, lonsize)
        else:
            ncfile.createDimension("gp", arrlon.size)
            gpisize = arrlon.size
            if gpis is None:
                gpivalues = np.arange(arrlon.size, dtype=np.int32)
            else:
                gpivalues = gpis

        dim = ncfile.dimensions.keys()

        gpi = ncfile.createVariable('gpi', np.dtype('int32').char, dim)

        if gpis is None:
            gpi[:] = gpivalues
            setattr(gpi, 'long_name', 'Grid point index')
            setattr(gpi, 'units', '')
            setattr(gpi, 'valid_range', [0, gpisize])
            gpidirect = 0x1b
        else:
            gpi[:] = gpivalues
            setattr(gpi, 'long_name', 'Grid point index')
            setattr(gpi, 'units', '')
            setattr(gpi, 'valid_range', [np.min(gpivalues), np.max(gpivalues)])
            gpidirect = 0x0b

        latitude = ncfile.createVariable('lat', np.dtype('float32').char,
                                         dim[0])
        latitude[:] = arrlat
        setattr(latitude, 'long_name', 'Latitude')
        setattr(latitude, 'units', 'degree_north')
        setattr(latitude, 'standard_name', 'latitude')
        setattr(latitude, 'valid_range', [-90.0, 90.0])

        if len(dim) == 2:
            londim = dim[1]
        else:
            londim = dim[0]
        longitude = ncfile.createVariable('lon', np.dtype('float32').char,
                                          londim)
        longitude[:] = arrlon
        setattr(longitude, 'long_name', 'Longitude')
        setattr(longitude, 'units', 'degree_east')
        setattr(longitude, 'standard_name', 'longitude')
        setattr(longitude, 'valid_range', [-180.0, 180.0])

        if arrcell is not None:
            cell = ncfile.createVariable('cell', np.dtype('int16').char, dim)
            cell[:] = arrcell
            setattr(cell, 'long_name', 'Cell')
            setattr(cell, 'units', '')
            setattr(cell, 'valid_range', [np.min(arrcell), np.max(arrcell)])

        if subset_points is not None:
            land_flag = ncfile.createVariable('subset_flag',
                                              np.dtype('int8').char,
                                              dim)

            # create landflag array based on shape of data
            lf = np.zeros_like(gpivalues)
            if len(dim) == 2:
                lf = lf.flatten()
            lf[subset_points] = 1
            if len(dim) == 2:
                lf = lf.reshape(latsize, lonsize)

            land_flag[:] = lf
            setattr(land_flag, 'long_name', subset_name)
            setattr(land_flag, 'units', '')
            setattr(land_flag, 'coordinates', 'lat lon')
            setattr(land_flag, 'flag_values', np.arange(2, dtype=np.int8))
            setattr(land_flag, 'flag_meanings', 'water land')
            setattr(land_flag, 'valid_range', [0, 1])

        s = "%Y-%m-%d %H:%M:%S"
        date_created = datetime.now().strftime(s)

        attr = {'Conventions': 'CF-1.6',
                'id': os.path.split(filename)[1],  # file name
                'date_created': date_created,
                'geospatial_lat_min': np.round(np.min(arrlat), 4),
                'geospatial_lat_max': np.round(np.max(arrlat), 4),
                'geospatial_lon_min': np.round(np.min(arrlon), 4),
                'geospatial_lon_max': np.round(np.max(arrlon), 4),
                'gpidirect': gpidirect
                }

        ncfile.setncatts(attr)
        if global_attrs is not None and type(global_attrs) is dict:
            ncfile.setncatts(global_attrs)


def save_grid(filename, grid, subset_name='land_points',
              subset_meaning="water land", global_attrs=None):
    """
    save a BasicGrid or CellGrid to netCDF
    it is assumed that a subset should be used as land_points

    Parameters
    ----------
    filename : string
        name of file
    grid : BasicGrid or CellGrid object
        grid whose definition to save to netCDF
    subset_name : string, optional
        long_name of the variable 'subset_flag'
        if the subset symbolises something other than a land/sea mask
    subset_meaning : string, optional
        will be written into flag_meanings metadata of variable 'subset_flag'
    global_attrs : dict, optional
        if given will be written as global attributs into netCDF file
    """

    try:
        arrcell = grid.arrcell
    except AttributeError:
        arrcell = None

    if grid.gpidirect is True:
        gpis = None
    else:
        gpis = grid.gpis

    if grid.shape is not None:
        if global_attrs is None:
            global_attrs = {}
        global_attrs['shape'] = grid.shape

    save_lonlat(filename, grid.arrlon, grid.arrlat, arrcell=arrcell,
                gpis=gpis, subset_points=grid.subset, subset_name=subset_name,
                subset_meaning=subset_meaning,
                global_attrs=global_attrs)


def load_grid(filename):
    """
    load a grid from netCDF file

    Parameters
    ----------
    filename : string
        filename

    Returns
    -------
    grid : BasicGrid or CellGrid instance
        grid instance initialized with the loaded data
    """

    with Dataset(filename, 'r') as nc_data:
        # determine if it is a cell grid or a basic grid
        arrcell = None
        if 'cell' in nc_data.variables.keys():
            arrcell = nc_data.variables['cell'][:]

        # determine if gpis are in order or custom order
        if nc_data.gpidirect == 0x1b:
            gpis = None  # gpis can be calculated through np.arange..
        else:
            gpis = nc_data.variables['gpi'][:]

        shape = None
        if hasattr(nc_data, 'shape'):
            try:
                shape = tuple(nc_data.shape)
            except TypeError as e:
                try:
                    length = len(nc_data.shape)
                except TypeError:
                    length = nc_data.shape.size
                if length == 1:
                    shape = tuple([nc_data.shape])
                else:
                    raise e

        subset = None

        # check if grid has regular shape
        if len(shape) == 2:
            lons, lats = np.meshgrid(nc_data.variables['lon'][:],
                                     nc_data.variables['lat'][::-1])
            lons = lons.flatten('F')
            lats = lats.flatten('F')

            if 'subset_flag' in nc_data.variables.keys():
                subset = np.where(nc_data.variables['subset_flag'][:].flatten() == 1)[0]

        elif len(shape) == 1:
            lons = nc_data.variables['lon'][:]
            lats = nc_data.variables['lat'][:]

            # determine if it has a subset
            if 'subset_flag' in nc_data.variables.keys():
                subset = np.where(nc_data.variables['subset_flag'][:] == 1)[0]

        if arrcell is None:
            # BasicGrid
            return BasicGrid(lons,
                             lats,
                             gpis=gpis,
                             subset=subset,
                             shape=shape)
        else:
            # CellGrid
            return CellGrid(nc_data.variables['lon'][:],
                            nc_data.variables['lat'][:],
                            arrcell,
                            gpis=gpis,
                            subset=subset,
                            shape=shape)
