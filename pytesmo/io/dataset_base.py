# Copyright (c) 2014,Vienna University of Technology, Department of Geodesy
# and Geoinformation. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of
#      Geodesy and Geoinformation nor the names of its contributors may be
#      used to endorse or promote products derived from this software without
#      specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Mar 19, 2014

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''
import numpy as np
from datetime import datetime, timedelta
import abc


class DatasetTSBase(object):
    """
    Dateset base class that implements basic functions and also abstract
    methods that have to be implemented by child classes.

    Parameters
    ----------
    path : string
        Path to dataset.
    grid : pytesmo.grid.grids.BasicGrid of CellGrid instance
        Grid on which the time series data is stored.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, path, grid):
        self.path = path
        self.grid = grid

    def _read_lonlat(self, lon, lat, **kwargs):
        """
        Reading time series for given longitude and latitude coordinate.

        Parameters
        ----------
        lon : float
            Longitude coordinate.
        lat : float
            Latitude coordinate.

        Returns
        -------
        data : pandas.DataFrame
            pandas.DateFrame with DateTimeIndex.
        """
        gp, _ = self.grid.find_nearest_gpi(lon, lat)
        return self.read_gp(gp, **kwargs)

    def read_ts(self, *args, **kwargs):
        """
        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lat,lon coordinates and then reading it
        """
        if len(args) == 1:
            data = self.read_gp(args[0], **kwargs)
        if len(args) == 2:
            data = self._read_lonlat(args[0], args[1], **kwargs)

        return data

    def iter_ts(self, ll_bbox=None):
        """
        Yield all time series for a grid or for grid points in a given
        lon/lat bound box (ll_bbox).

        Parameters
        ----------
        ll_bbox : tuple of floats (latmin, latmax, lonmin, lonmax)
            Set to lon/lat bounding box to yield only points in that area.

        Returns
        -------
        data : pandas.DataFrame
            pandas.DateFrame with DateTimeIndex
        """
        if ll_bbox is None:
            for gp, _, _ in self.grid.grid_points():
                yield self.read_gp(gp)
        else:
            latmin, latmax, lonmin, lonmax = ll_bbox
            gp_ll_bbox = self.grid.get_bbox_grid_points(latmin, latmax,
                                                        lonmin, lonmax)
            for gp in gp_ll_bbox:
                yield self.read_gp(gp)

    @abc.abstractmethod
    def read_gp(self, gpi, **kwargs):
        """
        Reads time series for a given grid point index(gpi)

        Parameters
        ----------
        gpi : int
            grid point index

        Returns
        -------
        data : pandas.DataFrame
            pandas.DateFrame with DateTimeIndex
        """
        return


class DatasetImgBase(object):
    """
    Dateset base class that implements basic functions and also abstract
    methods that have to be implemented by child classes.

    Parameters
    ----------
    path : string
        Path to dataset.
    grid : pytesmo.grid.grids.BasicGrid of CellGrid instance, optional
        Grid on which all the images of the dataset are stored. This is not
        relevant for datasets that are stored e.g. in orbit geometry
    img_offsets : numpy array of timedeltas, optional
        Gives the time offsets in a day that the images of one day have.
        e.g.: a model data with 6 hourly data with images at 0, 6, 12 and
        18 o'clock would have img_offsets=np.array([timedelta(hours=0),
                                                    timedelta(hours=6),
                                                    timedelta(hours=12),
                                                    timedelta(hours=18)])
        Default is set to np.array([timedelta(hours=0)]) for daily images
        with timestamp 0
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, path, grid=None,
                 img_offsets=np.array([timedelta(hours=0)])):
        self.grid = grid
        self.path = path
        self.img_offsets = img_offsets

    def _get_possible_timestamps(self, timestamp):
        """
        Get the timestamps as datetime array that are possible for the
        given day, if the timestamps are

        Parameters
        ----------
        timestamp : datetime.datetime
            Specific day.

        Returns
        -------
        timestamps : numpy.ndarray
            Datetime array of possible timestamps.
        """
        day = datetime(timestamp.year, timestamp.month, timestamp.day)
        return day + self.img_offsets

    @abc.abstractmethod
    def _read_spec_img(self, timestamp, **kwargs):
        """
        Read specific image for given datetime timestamp.

        Parameters
        ----------
        timestamp : datetime.datetime
            exact observation timestamp of the image that should be read

        Returns
        -------
        data : dict
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset
        metadata : dict
            dictionary of numpy arrays that hold the metadata
        timestamp : datetime.datetime
            exact timestamp of the image
        lon : numpy.array or None
            array of longitudes, if None self.grid will be assumed
        lat : numpy.array or None
            array of latitudes, if None self.grid will be assumed
        time_var : string or None
            variable name of observation times in the data dict, if None all
            observations have the same timestamp
        """
        return

    def read_img(self, timestamp, **kwargs):
        """
        Return an image if a specific datetime is given.

        Parameters
        ----------
        timestamp : datetime.datetime
            Time stamp.

        Returns
        -------
        data : dict
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset
        metadata : dict
            dictionary of numpy arrays that hold the metadata
        timestamp : datetime.datetime
            exact timestamp of the image
        lon : numpy.array or None
            array of longitudes, if None self.grid will be assumed
        lat : numpy.array or None
            array of latitudes, if None self.grid will be assumed
        time_var : string or None
            variable name of observation times in the data dict, if None all
            observations have the same timestamp
        """
        if type(timestamp) == datetime:
            if timestamp in self._get_possible_timestamps(timestamp):
                return self._read_spec_img(timestamp, **kwargs)
            else:
                raise ValueError("given timestamp is not a valid image "
                                 "timestamp for this dataset")
        else:
            raise TypeError("given timestamp is not a datetime")

    def daily_images(self, day, **kwargs):
        """
        Yield all images for a day.

        Parameters
        ----------
        day : datetime.date or datetime.datetime
            Specific day.

        Returns
        -------
        data : dict
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset
        metadata : dict
            dictionary of numpy arrays that hold the metadata
        timestamp : datetime.datetime
            exact timestamp of the image
        lon : numpy.array or None
            array of longitudes, if None self.grid will be assumed
        lat : numpy.array or None
            array of latitudes, if None self.grid will be assumed
        time_var : string or None
            variable name of observation times in the data dict, if None all
            observations have the same timestamp
        """
        for image_datetime in self._get_possible_timestamps(day):
            yield_img = self._read_spec_img(image_datetime, **kwargs)
            yield yield_img
