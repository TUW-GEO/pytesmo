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
Data scaler classes to be used together with the validation framework.
'''

import numpy as np
import pandas as pd
import pytesmo.scaling as scaling
from pytesmo.scaling import lin_cdf_match_stored_params
from pytesmo.utils import unique_percentiles_interpolate
from pynetcf.point_data import GriddedPointData


class DefaultScaler(object):
    """
    Scaling class that implements the scaling based on a
    given method from the pytesmo.scaling module.

    Parameters
    ----------
    method: string
        The data will be scaled into the reference space using the
        method specified by this string.
    """

    def __init__(self, method):
        self.method = method

    def scale(self, data, reference_index, gpi_info):
        """
        Scale all columns in data to the
        column at the reference_index.

        Parameters
        ----------
        data: pandas.DataFrame
            temporally matched dataset
        reference_index: int
            Which column of the data contains the
            scaling reference.
        gpi_info: tuple
            tuple of at least, (gpi, lon, lat)
            Where gpi has to be the grid point indices
            of the grid of this scaler.

        Raises
        ------
        ValueError
            if scaling is not successful
        """
        return scaling.scale(data,
                             method=self.method,
                             reference_index=reference_index)


class CDFStoreParamsScaler(object):
    """
    CDF scaling using stored parameters if available.
    If stored parameters are not available they are calculated
    and written to disk.

    Parameters
    ----------
    path: string
        Path where the data is/should be stored
    grid: :py:class:`pygeogrids.grids.CellGrid` instance
        Grid on which the data is stored.
        Should be the same as the spatial reference grid
        of the validation framework instance in which this
        scaler is used.
    percentiles: list or np.ndarray
        Percentiles to use for CDF matching
    """

    def __init__(self, path, grid,
                 percentiles=[0, 5, 10, 30, 50, 70, 90, 95, 100]):
        self.path = path
        self.grid = grid
        self.percentiles = np.asanyarray(percentiles)
        self.io = GriddedPointData(path, grid, mode='a',
                                   ioclass_kws={'add_dims': {'percentiles': self.percentiles.size}})

    def scale(self, data, reference_index, gpi_info):
        """
        Scale all columns in data to the
        column at the reference_index.

        Parameters
        ----------
        data: pandas.DataFrame
            temporally matched dataset
        reference_index: int
            Which column of the data contains the
            scaling reference.
        gpi_info: tuple
            tuple of at least, (gpi, lon, lat)
            Where gpi has to be the grid point indices
            of the grid of this scaler.

        Raises
        ------
        ValueError
            if scaling is not successful
        """
        gpi = gpi_info[0]
        parameters = self.get_parameters(data, gpi)

        reference_name = data.columns.values[reference_index]
        reference = data[reference_name]
        data = data.drop([reference_name], axis=1)
        for series in data:
            src_percentiles = parameters[series]
            ref_percentiles = parameters[reference_name]
            data[series] = pd.Series(
                lin_cdf_match_stored_params(data[series].values,
                                            src_percentiles,
                                            ref_percentiles),
                index=data.index)

        data.insert(reference_index, reference.name, reference)
        return data

    def calc_parameters(self, data):
        """
        Calculate the percentiles used for CDF matching.

        Parameters
        ----------
        data: pandas.DataFrame
            temporally matched dataset

        Returns
        -------
        parameters: dictionary
            keys -> Names of columns in the input data frame
            values -> numpy.ndarrays with the percentiles
        """

        parameters = {}
        for column in data.columns:
            c_data = data[column].values
            perc = np.percentile(c_data, self.percentiles)
            perc = unique_percentiles_interpolate(perc,
                                                  percentiles=self.percentiles)
            parameters[column] = perc

        return parameters

    def get_parameters(self, data, gpi):
        """
        Function to get scaling parameters.
        Try to load them, if they are not found we
        calculate them and store them.

        Parameters
        ----------
        data: pandas.DataFrame
            temporally matched dataset
        gpi: int
            grid point index of self.grid

        Returns
        -------
        params: dictionary
            keys -> Names of columns in the input data frame
            values -> numpy.ndarrays with the percentiles
        """

        params = self.load_parameters(gpi)
        if params is None:
            params = self.calc_parameters(data)
            self.store_parameters(gpi, params)
        return params

    def load_parameters(self, gpi):
        data = self.io.read(gpi)
        if data is not None:
            unwanted_keys = ['lat', 'lon', 'alt', 'time', 'location_id']
            for key in unwanted_keys:
                del data[key]

            # remove extra dimension from reading from netCDF
            for key in data:
                data[key] = np.squeeze(data[key])

        return data

    def store_parameters(self, gpi, parameters):
        """
        Store parameters for gpi into netCDF file.

        Parameters
        ----------
        gpi: int
            grid point index of self.grid
        params: dictionary
            keys -> Names of columns in the input data frame
            values -> numpy.ndarrays with the percentiles
        """
        data = []
        dtypes = []
        dim_info = {'dims': {}}
        for key in parameters:
            dim_info['dims'][key] = ('obs', 'percentiles')
            dtypes.append(
                (key, parameters[key].dtype, (parameters[key].size, )))
            data.append(parameters[key])

        data = np.core.records.fromarrays(data, dtype=np.dtype(dtypes,
                                                               metadata=dim_info))
        self.io.write(gpi, data)
