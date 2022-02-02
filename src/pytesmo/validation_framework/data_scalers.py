# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the TU Wien, Department of Geodesy and
#      Geoinformation nor the names of its contributors may be used to endorse
#      or promote products derived from this software without specific prior
#      written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Data scaler classes to be used together with the validation framework.
"""

import numpy as np
import pandas as pd
import pytesmo.scaling as scaling
from pytesmo.cdf_matching import CDFMatching
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
        return scaling.scale(
            data, method=self.method, reference_index=reference_index
        )


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

    def __init__(
            self, path, grid, percentiles=[0, 5, 10, 30, 50, 70, 90, 95, 100],
            linear_edge_scaling=True
    ):
        self.path = path
        self.grid = grid
        self.percentiles = np.asanyarray(percentiles)
        self.io = GriddedPointData(
            path,
            grid,
            mode="a",
            ioclass_kws={
                "add_dims": {"src_ref": 2, "percentiles": len(percentiles)}
            },
        )
        self.percentiles = percentiles
        self.linear_edge_scaling = linear_edge_scaling

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
        parameters = self.get_parameters(data, reference_index, gpi)

        refname = data.columns.values[reference_index]
        reference = data[refname]
        for column in data:
            if column == refname:
                continue
            params = parameters[f"{column}_{refname}"]
            matcher = CDFMatching()
            nbins = len(params) // 2
            matcher.x_perc_ = params[:nbins]
            matcher.y_perc_ = params[nbins:]
            data[column] = pd.Series(matcher.predict(data[column]),
                                     index=data.index)

        return data

    def calc_parameters(self, data, reference_index):
        """
        Calculate the percentiles used for CDF matching.

        Parameters
        ----------
        data: pandas.DataFrame
            temporally matched dataset

        Returns
        -------
        matchers: dictionary
            keys -> Names of columns in the input data frame
            values -> nbins x 3 numpy.ndarrays with columns x_perc, y_perc,
                      percentiles
        """

        parameters = {}
        refname = data.columns[reference_index]
        for column in data.columns:
            if column == refname:
                continue
            matcher = CDFMatching(percentiles=self.percentiles,
                                  linear_edge_scaling=self.linear_edge_scaling)
            matcher.fit(data[column], data[refname])
            nbins = matcher.nbins
            params = np.zeros((2, nbins), matcher.x_perc_.dtype)
            params[0, :] = matcher.x_perc_
            params[1, :] = matcher.y_perc_
            parameters[f"{column}_{refname}"] = params
        return parameters

    def get_parameters(self, data, reference_index, gpi):
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
            params = self.calc_parameters(data, reference_index)
            self.store_parameters(gpi, params)
        return params

    def load_parameters(self, gpi):
        data = self.io.read(gpi)
        if data is not None:
            unwanted_keys = ["lat", "lon", "alt", "time", "location_id"]
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
        dim_info = {"dims": {}}
        for key in parameters:
            dim_info["dims"][key] = ("obs", "src_ref", "percentiles")
            dtypes.append(
                (key, parameters[key].dtype, (parameters[key].shape[-1],))
            )
            data.append(parameters[key])

        data = np.core.records.fromarrays(
            data, dtype=np.dtype(dtypes, metadata=dim_info)
        )
        self.io.write(gpi, data)
