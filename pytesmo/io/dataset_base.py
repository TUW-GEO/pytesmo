# Copyright (c) 2014,Vienna University of Technology, Department of Geodesy
# and Geoinformation. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be
#     used to endorse or promote products derived from this software without
#     specific prior written permission.

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

"""
Created on Mar 19, 2014

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
"""

import numpy as np
from datetime import datetime
import abc
import os
import glob


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

    def get_nearest_gp_info(self, lon, lat):
        """
        get info for nearest grid point

        Parameters
        ----------
        lon : float
            Longitude coordinate.
        lat : float
            Latitude coordinate.

        Returns
        -------
        gpi : int
            Grid point index of nearest grid point.
        gp_lon : float
            Lontitude coordinate of nearest grid point.
        gp_lat : float
            Latitude coordinate of nearest grid point.
        gp_dist : float
            Geodetic distance to nearest grid point.
        """
        gpi, gp_dist = self.grid.find_nearest_gpi(lon, lat)
        gp_lon, gp_lat = self.grid.gpi2lonlat(gpi)

        return gpi, gp_lon, gp_lat, gp_dist

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
            for gp, _, _, _ in self.grid.grid_points():
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


class DatasetStaticBase(object):

    """
    Dataset base class for arrays that do have a grid associated with them but
    are not image time series.

    Parameters
    ----------
    filename : string
        path and filename of file to load
    grid : pytesmo.BasicGrid or similar grid definition class
        defines the grid on which the dataset is stored
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, filename, grid):
        self.filename = filename
        self.grid = grid
        self.data = None

    @abc.abstractmethod
    def read_data(self):
        """
        Reads the data and returns it as a dictionary of numpy arrays.

        Returns
        -------
        data : dict
            dictionary of numpy arrays
        """
        return

    def read_pos(self, *args, **kwargs):
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

    def _read_lonlat(self, lon, lat, **kwargs):
        """
        Reading data for given longitude and latitude coordinate.

        Parameters
        ----------
        lon : float
            Longitude coordinate.
        lat : float
            Latitude coordinate.

        Returns
        -------
        data : dict of values
            data record.
        """
        gp, _ = self.grid.find_nearest_gpi(lon, lat)
        return self.read_gp(gp, **kwargs)

    def read_gp(self, gpi, **kwargs):
        """
        Reads data record for a given grid point index(gpi)

        Parameters
        ----------
        gpi : int
            grid point index

        Returns
        -------
        data : dict of values
            data record.
        """
        if self.data is None:
            self.data = self.read_data()

        gp_data = {}
        for key in self.data:
            # make sure the data is a 1D array when using
            # the gpi as indices
            gp_data[key] = np.ravel(self.data[key])[gpi]

        return gp_data


class DatasetImgBase(object):

    """
    Dateset base class that implements basic functions and also abstract
    methods that have to be implemented by child classes.

    Parameters
    ----------
    path : string
        Path to dataset.
    filename_templ : string
        template of how datetimes fit into the filename.
        e.g. "ASCAT_%Y%m%d_image.nc" will be translated into the filename
        ASCAT_20070101_image.nc for the date 2007-01-01.
    sub_path : string or list optional
        if given it is used to generate a sub path from the given timestamp.
        This is useful if files are sorted by year or month.
        If a list is one subfolder per item is assumed. This can be used
        if the files for May 2007 are e.g. in folders 2007/05/ then the
        list ['%Y', '%m'] works.
    grid : pytesmo.grid.grids.BasicGrid of CellGrid instance, optional
        Grid on which all the images of the dataset are stored. This is not
        relevant for datasets that are stored e.g. in orbit geometry
    exact_templ : boolean, optional
        if True then the filename_templ matches the filename exactly.
        If False then the filename_templ will be used in glob to find
        the file.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, path, filename_templ="",
                 sub_path=None, grid=None,
                 exact_templ=True):
        self.grid = grid
        self.fname_templ = filename_templ
        self.path = path
        if type(sub_path) == str:
            sub_path = [sub_path]
        self.sub_path = sub_path
        self.exact_templ = exact_templ

    @abc.abstractmethod
    def _read_spec_file(self, filename, timestamp=None, **kwargs):
        """
        Read specific image for given filename

        Parameters
        ----------
        filename : string
            filename
        timestamp : datetime, optional
           can be given here if it is already
           known since it has to be returned.

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
        time : numpy.array or None
            observation times of the data as numpy array of julian dates,
            if None all observations have the same timestamp
        """
        return

    def _search_files(self, timestamp, custom_templ=None,
                      str_param=None):
        """
        searches for filenames for the given timestamp.
        This function is used by _build_filename which then
        checks if a unique filename was found

        Parameters
        ----------
        timestamp: datetime
            datetime for given filename
        custom_tmpl : string, optional
            if given not the fname_templ is used but the custom templ
            This is convienint for some datasets where no all filenames
            follow the same convention and where the read_img function
            can choose between templates based on some condition.
        str_param : dict, optional
            if given then this dict will be applied to the template using
            the fname_template.format(**str_param) notation before the resulting
            string is put into datetime.strftime.

            example from python documentation
            >>> coord = {'latitude': '37.24N', 'longitude': '-115.81W'}
            >>> 'Coordinates: {latitude}, {longitude}'.format(**coord)
            'Coordinates: 37.24N, -115.81W'
        """
        if custom_templ is not None:
            fname_templ = custom_templ
        else:
            fname_templ = self.fname_templ

        if str_param is not None:
            fname_templ = fname_templ.format(**str_param)
        if self.sub_path is None:
            search_file = os.path.join(
                self.path, timestamp.strftime(fname_templ))

        else:
            sub_path = ""
            for s in self.sub_path:
                sub_path = os.path.join(sub_path, timestamp.strftime(s))
            search_file = os.path.join(self.path,
                                       sub_path,
                                       timestamp.strftime(fname_templ))
        if self.exact_templ:
            return [search_file]
        else:
            filename = glob.glob(search_file)

        if not filename:
            raise IOError("File not found {:}".format(search_file))

        return filename

    def _build_filename(self, timestamp, custom_templ=None,
                        str_param=None):
        """
        This function uses _search_files to find the correct
        filename and checks if the search was unambiguous

        Parameters
        ----------
        timestamp: datetime
            datetime for given filename
        custom_tmpl : string, optional
            if given not the fname_templ is used but the custom templ
            This is convienint for some datasets where no all filenames
            follow the same convention and where the read_img function
            can choose between templates based on some condition.
        str_param : dict, optional
            if given then this dict will be applied to the template using
            the fname_template.format(**str_param) notation before the resulting
            string is put into datetime.strftime.

            example from python documentation

            >>> coord = {'latitude': '37.24N', 'longitude': '-115.81W'}
            >>> 'Coordinates: {latitude}, {longitude}'.format(**coord)
            'Coordinates: 37.24N, -115.81W'
        """
        filename = self._search_files(timestamp, custom_templ=custom_templ,
                                      str_param=str_param)

        if len(filename) > 1:
            raise IOError(
                "File search is ambiguous {:}".format(filename))

        return filename[0]

    def _assemble_img(self, timestamp, **kwargs):
        """
        Function between read_img and _build_filename that can
        be used to read a different file for each parameter in a image
        dataset. In the standard impementation it is assumed
        that all necessary information of a image is stored in the
        one file whose filename is built by the _build_filname function.

        Parameters
        ----------
        timestamp : datatime
            timestamp of the image to assemble

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
        return self._read_spec_file(self._build_filename(timestamp),
                                    timestamp=timestamp, **kwargs)

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
        return self._assemble_img(timestamp, **kwargs)

    def tstamps_for_daterange(self, start_date, end_date):
        """
        Return all valid timestamps in a given date range.
        This method must be implemented if iteration over
        images should be possible.

        Parameters
        ----------
        start_date : datetime.date or datetime.datetime
            start date
        end_date : datetime.date or datetime.datetime
            end date

        Returns
        -------
        dates : list
            list of datetimes
        """

        raise NotImplementedError(
            "Please implement to enable iteration over date ranges.")

    def iter_images(self, start_date, end_date, **kwargs):
        """
        Yield all images for a given date range.

        Parameters
        ----------
        start_date : datetime.date or datetime.datetime
            start date
        end_date : datetime.date or datetime.datetime
            end date

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
        timestamps = self.tstamps_for_daterange(start_date, end_date)

        if timestamps:
            for timestamp in timestamps:
                yield_img = self.read_img(
                    timestamp, **kwargs)
                yield yield_img
        else:
            raise IOError("no files found for given date range")

    def daily_images(self, day, **kwargs):
        """
        Yield all images for a day.

        Parameters
        ----------
        day : datetime.date

        Returns
        -------
        data : dict
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset
        metadata : dict
            dictionary of numpy arrays that hold metadata
        timestamp : datetime.datetime
            exact timestamp of the image
        lon : numpy.array or None
            array of longitudes, if None self.grid will be assumed
        lat : numpy.array or None
            array of latitudes, if None self.grid will be assumed
        jd : string or None
            name of the field in the data array representing the observation
            dates
        """
        for img in self.iter_images(day, day, **kwargs):
            yield img
