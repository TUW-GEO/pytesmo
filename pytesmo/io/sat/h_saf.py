# Copyright (c) 2014,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

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
Created on May 21, 2014

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

import os
import glob
from datetime import datetime, timedelta
import numpy as np
import warnings

import pytesmo.io.dataset_base as dataset_base

import pytesmo.io.bufr.bufr as bufr_reader
try:
    import pygrib
except ImportError:
    warnings.warn('pygrib can not be imported H14 images can not be read.')

import pytesmo.timedate.julian as julian


class H08img(dataset_base.DatasetImgBase):
    """
    Reads HSAF H08 images. The images have to be uncompressed in the following folder structure
    path -
         month_path_str (default 'h08_%Y%m_buf')

    For example if path is set to /home/user/hsaf08 and month_path_str is left to the default 'h08_%Y%m_buf'
    then the images for March 2012 have to be in
    the folder /home/user/hsaf08/h08_201203_buf/

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h08_%Y%m_buf'
    day_search_str: string, optional
        to provide an iterator over all images of a day the method _get_possible_timestamps
        looks for all available images on a day on the harddisk. This string is used in
        datetime.datetime.strftime and in glob.glob to search for all files on a day.
        Default : 'h08_%Y%m%d_*.buf'
    file_search_str: string, optional
        this string is used in datetime.datetime.strftime and glob.glob to find a 3 minute bufr file
        by the exact date.
        Default: 'h08_%Y%m%d_%H%M%S*.buf'
    """

    def __init__(self, path, month_path_str='h08_%Y%m_buf',
                 day_search_str='h08_%Y%m%d_*.buf',
                 file_search_str='h08_%Y%m%d_%H%M%S*.buf'):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        super(H08img, self).__init__(path, grid=None)

    def _get_possible_timestamps(self, timestamp):
        """
        Get the timestamps as datetime array that are possible for the
        given day, if the timestamps are

        For this product it is not fixed but has to be looked up from
        the hard disk since bufr files are not regular spaced and only
        europe is in this product. For a global product a 3 minute
        spacing could be used as a fist approximation

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

        filelist = glob.glob(os.path.join(self.path, day.strftime(self.month_path_str),
                                          day.strftime(self.day_search_str)))

        if len(filelist) == 0:
            raise ValueError("No data for this day")
        img_offsets = []
        for _file in filelist:
            filename = os.path.split(_file)[1]
            offset = timedelta(hours=int(filename[13:15]),
                               minutes=int(filename[15:17]),
                               seconds=int(filename[17:19]))
            img_offsets.append(offset)

        return day + np.array(img_offsets)

    def _read_spec_img(self, timestamp, lat_lon_bbox=None):
        """
        Read specific image for given datetime timestamp.

        Parameters
        ----------
        timestamp : datetime.datetime
            exact observation timestamp of the image that should be read
        lat_lon_bbox : list, optional
            list of lat,lon cooridnates of bounding box
            [lat_min, lat_max, lon_min, lon_max]

        Returns
        -------
        data : dict or None
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset, if no data was found None is returned
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
        filename = glob.glob(os.path.join(self.path, timestamp.strftime(self.month_path_str),
                                          timestamp.strftime(self.file_search_str)))
        if len(filename) != 1:
            raise ValueError("There must be exactly one file for given timestamp %s" % timestamp.isoformat())

        with bufr_reader.BUFRReader(filename[0]) as bufr:
            lons = []
            ssm = []
            ssm_noise = []
            ssm_corr_flag = []
            ssm_proc_flag = []
            data_in_bbox = True
            for i, message in enumerate(bufr.messages()):
                if i == 0:
                    # first message is just lat, lon extent
                    # check if any data in bbox
                    if lat_lon_bbox is not None:
                        lon_min, lon_max = message[0, 2], message[0, 3]
                        lat_min, lat_max = message[0, 4], message[0, 5]
                        if (lat_lon_bbox[0] > lat_max or lat_lon_bbox[1] < lat_min or
                            lat_lon_bbox[2] > lon_max or lat_lon_bbox[3] < lon_min):
                            data_in_bbox = False
                            break
                    # print 'columns', math.ceil((message[:, 3] - message[:, 2]) / 0.00416667)
                    # print 'rows', math.ceil((message[:, 5] - message[:, 4]) / 0.00416667)
                elif data_in_bbox:
                    # first 5 elements are there only once, after that, 4 elements are repeated
                    # till the end of the array these 4 are ssm, ssm_noise, ssm_corr_flag and
                    # ssm_proc_flag
                    # each message contains the values for 120 lons between lat_min and lat_max
                    # the grid spacing is 0.00416667 degrees
                    lons.append(message[:, 0])
                    lat_min = message[0, 1]
                    lat_max = message[0, 2]
                    ssm.append(message[:, 4::4])
                    ssm_noise.append(message[:, 5::4])
                    ssm_corr_flag.append(message[:, 6::4])
                    ssm_proc_flag.append(message[:, 7::4])

        if data_in_bbox:
            ssm = np.rot90(np.vstack(ssm)).astype(np.float32)
            ssm_noise = np.rot90(np.vstack(ssm_noise)).astype(np.float32)
            ssm_corr_flag = np.rot90(np.vstack(ssm_corr_flag)).astype(np.float32)
            ssm_proc_flag = np.rot90(np.vstack(ssm_proc_flag)).astype(np.float32)
            lats_dim = np.linspace(lat_max, lat_min, ssm.shape[0])
            lons_dim = np.concatenate(lons)

            data = {'ssm': ssm,
                    'ssm_noise': ssm_noise,
                    'proc_flag': ssm_proc_flag,
                    'corr_flag': ssm_corr_flag
                    }

            # if there are is a gap in the image it is not a 2D array in lon, lat space
            # but has a jump in latitude or longitude
            # detect a jump in lon or lat spacing
            lon_jump_ind = np.where(np.diff(lons_dim) > 0.00418)[0]
            if lon_jump_ind.size > 1:
                print "More than one jump in longitude"
            if lon_jump_ind.size == 1:
                diff_lon_jump = np.abs(lons_dim[lon_jump_ind] - lons_dim[lon_jump_ind + 1])
                missing_elements = np.round(diff_lon_jump / 0.00416666)
                missing_lons = np.linspace(lons_dim[lon_jump_ind],
                                           lons_dim[lon_jump_ind + 1], missing_elements,
                                           endpoint=False)

                # fill up longitude dimension to full grid
                lons_dim = np.concatenate([lons_dim[:lon_jump_ind], missing_lons, lons_dim[lon_jump_ind + 1:]])
                # fill data with NaN values
                empty = np.empty((lats_dim.shape[0], missing_elements))
                empty.fill(1e38)
                for key in data:
                    data[key] = np.concatenate([data[key][:, :lon_jump_ind], empty, data[key][:, lon_jump_ind + 1:]], axis=1)

            lat_jump_ind = np.where(np.diff(lats_dim) > 0.00418)[0]
            if lat_jump_ind.size > 1:
                print "More than one jump in latitude"
            if lat_jump_ind.size == 1:
                diff_lat_jump = np.abs(lats_dim[lat_jump_ind] - lats_dim[lat_jump_ind + 1])
                missing_elements = np.round(diff_lat_jump / 0.00416666)
                missing_lats = np.linspace(lats_dim[lat_jump_ind],
                                           lats_dim[lat_jump_ind + 1], missing_elements,
                                           endpoint=False)

                # fill up longitude dimension to full grid
                lats_dim = np.concatenate([lats_dim[:lat_jump_ind], missing_lats, lats_dim[lat_jump_ind + 1:]])
                # fill data with NaN values
                empty = np.empty((missing_elements, lons_dim.shape[0]))
                empty.fill(1e38)
                for key in data:
                    data[key] = np.concatenate([data[key][:lat_jump_ind, :], empty, data[key][lat_jump_ind + 1:, :]], axis=0)

            lons, lats = np.meshgrid(lons_dim, lats_dim)
            # only return data in bbox
            if lat_lon_bbox is not None:
                data_ind = np.where((lats >= lat_lon_bbox[0]) &
                                    (lats <= lat_lon_bbox[1]) &
                                    (lons >= lat_lon_bbox[2]) &
                                    (lons <= lat_lon_bbox[3]))
                # indexing returns 1d array
                # get shape of lats_dim and lons_dim to be able to reshape
                # the 1d arrays to the correct 2d shapes
                lats_dim_shape = np.where((lats_dim >= lat_lon_bbox[0]) &
                                        (lats_dim <= lat_lon_bbox[1]))[0].shape[0]
                lons_dim_shape = np.where((lons_dim >= lat_lon_bbox[2]) &
                                        (lons_dim <= lat_lon_bbox[3]))[0].shape[0]

                lons = lons[data_ind].reshape(lats_dim_shape, lons_dim_shape)
                lats = lats[data_ind].reshape(lats_dim_shape, lons_dim_shape)
                for key in data:
                    data[key] = data[key][data_ind].reshape(lats_dim_shape, lons_dim_shape)

            return data, {}, timestamp, lons, lats, None

        else:
            return None, {}, timestamp, None, None, None

    def read_img(self, timestamp, lat_lon_bbox=None):
        """
        Return an image if a specific datetime is given.

        Parameters
        ----------
        timestamp : datetime.datetime
            Time stamp.
        lat_lon_bbox : list, optional
            list of lat,lon cooridnates of bounding box
            [lat_min, lat_max, lon_min, lon_max]

        Returns
        -------
        data : dict or None
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset, if no data was found None is returned
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
                return self._read_spec_img(timestamp, lat_lon_bbox=lat_lon_bbox)
            else:
                raise ValueError("given timestamp is not a valid image "
                                 "timestamp for this dataset")
        else:
            raise TypeError("given timestamp is not a datetime")

    def daily_images(self, day, lat_lon_bbox=None):
        """
        Yield all images for a day.

        Parameters
        ----------
        day : datetime.date or datetime.datetime
            Specific day.

        lat_lon_bbox : list, optional
            list of lat,lon cooridnates of bounding box
            [lat_min, lat_max, lon_min, lon_max]

        Returns
        -------
        data : dict or None
            dictionary of numpy arrays that hold the image data for each
            variable of the dataset. If not data was found None is returned
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
            yield_img = self._read_spec_img(image_datetime, lat_lon_bbox=lat_lon_bbox)
            yield yield_img


class H07img(dataset_base.DatasetImgBase):
    """
    Class for reading HSAF H07 SM OBS 1 images in bufr format.
    The images have the same structure as the ASCAT 3 minute pdu files
    and these 2 readers could be merged in the future
    The images have to be uncompressed in the following folder structure
    path -
         month_path_str (default 'h07_%Y%m_buf')

    For example if path is set to /home/user/hsaf07 and month_path_str is left to the default 'h07_%Y%m_buf'
    then the images for March 2012 have to be in
    the folder /home/user/hsaf07/h07_201203_buf/

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h07_%Y%m_buf'
    day_search_str: string, optional
        to provide an iterator over all images of a day the method _get_possible_timestamps
        looks for all available images on a day on the harddisk. This string is used in
        datetime.datetime.strftime and in glob.glob to search for all files on a day.
        Default : 'h07_%Y%m%d_*.buf'
    file_search_str: string, optional
        this string is used in datetime.datetime.strftime and glob.glob to find a 3 minute bufr file
        by the exact date.
        Default: 'h07_%Y%m%d_%H%M%S*.buf'
    """

    def __init__(self, path, month_path_str='h07_%Y%m_buf',
                 day_search_str='h07_%Y%m%d_*.buf',
                 file_search_str='h07_%Y%m%d_%H%M%S*.buf'):
        self.path = path
        self.month_path_str = month_path_str
        self.day_search_str = day_search_str
        self.file_search_str = file_search_str
        super(H07img, self).__init__(path, grid=None)

    def _get_possible_timestamps(self, timestamp):
        """
        Get the timestamps as datetime array that are possible for the
        given day, if the timestamps are

        For this product it is not fixed but has to be looked up from
        the hard disk since bufr files are not regular spaced and only
        europe is in this product. For a global product a 3 minute
        spacing could be used as a fist approximation

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

        filelist = glob.glob(os.path.join(self.path, day.strftime(self.month_path_str),
                                          day.strftime(self.day_search_str)))

        if len(filelist) == 0:
            raise ValueError("No data for this day")
        img_offsets = []
        for _file in filelist:
            filename = os.path.split(_file)[1]
            offset = timedelta(hours=int(filename[13:15]),
                               minutes=int(filename[15:17]),
                               seconds=int(filename[17:19]))
            img_offsets.append(offset)

        return day + np.array(img_offsets)

    def _read_spec_img(self, timestamp):
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
        filename = glob.glob(os.path.join(self.path, timestamp.strftime(self.month_path_str),
                                          timestamp.strftime(self.file_search_str)))
        if len(filename) != 1:
            raise ValueError("There must be exactly one file for given timestamp %s" % timestamp.isoformat())

        latitude = []
        longitude = []
        ssm = []
        dates = []
        orbit_number = []
        direction_of_motion = []
        ssm_sens = []
        frozen_lsf = []
        snow_cover = []
        topo_complex = []
        ssm_noise = []
        ssm_mean = []
        beam_ident = []
        azimuth = []
        incidence = []
        sig0 = []
        sigma40 = []
        sigma40_noise = []

        with bufr_reader.BUFRReader(filename[0]) as bufr:
            for message in bufr.messages():

                latitude.append(message[:, 12])
                longitude.append(message[:, 13])
                ssm.append(message[:, 64])
                orbit_number.append(message[:, 15])
                direction_of_motion.append(message[:, 5])
                ssm_sens.append(message[:, 70])
                frozen_lsf.append(message[:, 79])
                snow_cover.append(message[:, 78])
                topo_complex.append(message[:, 81])
                ssm_noise.append(message[:, 65])
                ssm_mean.append(message[:, 73])
                sigma40.append(message[:, 66])
                sigma40_noise.append(message[:, 67])

                beam_ident.append([message[:, 20],
                                   message[:, 34],
                                    message[:, 48]])
                incidence.append([message[:, 21],
                                  message[:, 35],
                                  message[:, 49]])
                azimuth.append([message[:, 22],
                                message[:, 36],
                                message[:, 50]])
                sig0.append([message[:, 23],
                             message[:, 37],
                             message[:, 51]])

                years = message[:, 6].astype(int)
                months = message[:, 7].astype(int)
                days = message[:, 8].astype(int)
                hours = message[:, 9].astype(int)
                minutes = message[:, 10].astype(int)
                seconds = message[:, 11].astype(int)

                dates.append(julian.julday(months, days, years,
                                            hours, minutes, seconds))

        ssm = np.concatenate(ssm)
        latitude = np.concatenate(latitude)
        longitude = np.concatenate(longitude)
        orbit_number = np.concatenate(orbit_number)
        direction_of_motion = np.concatenate(direction_of_motion)
        ssm_sens = np.concatenate(ssm_sens)
        frozen_lsf = np.concatenate(frozen_lsf)
        snow_cover = np.concatenate(snow_cover)
        topo_complex = np.concatenate(topo_complex)
        ssm_noise = np.concatenate(ssm_noise)
        ssm_mean = np.concatenate(ssm_mean)
        dates = np.concatenate(dates)
        sigma40 = np.concatenate(sigma40)
        sigma40_noise = np.concatenate(sigma40_noise)

        data = {'ssm': ssm,
                'ssm_noise': ssm_noise,
                'snow_cover': snow_cover,
                'frozen_prob': frozen_lsf,
                'topo_complex': topo_complex,
                'jd': dates
                }

        return data, {}, timestamp, longitude, latitude, 'jd'


class H14img(dataset_base.DatasetImgBase):
    """
    Class for reading HSAF H14 SM DAS 2 products in grib format
    The images have to be uncompressed in the following folder structure
    path -
         month_path_str (default 'h14_%Y%m_grib')

    For example if path is set to /home/user/hsaf14 and month_path_str is left to the default 'h14_%Y%m_grib'
    then the images for March 2012 have to be in
    the folder /home/user/hsaf14/h14_201203_grib/

    Parameters
    ----------
    path: string
        path where the data is stored
    month_path_str: string, optional
        if the files are stored in folders by month as is the standard on the HSAF FTP Server
        then please specify the string that should be used in datetime.datetime.strftime
        Default: 'h14_%Y%m_grib'
    file_str: string, optional
        this string is used in datetime.datetime.strftime to get the filename of a H14 daily grib file
        Default: 'H14_%Y%m%d00.grib'
    expand_grid : boolean, optional
        if set the images will be expanded to a 2D image during reading
        if false the images will be returned as 1D arrays on the
        reduced gaussian grid
        Default: True
    """

    def __init__(self, path, month_path_str='h14_%Y%m_grib',
                 file_str='H14_%Y%m%d00.grib',
                 expand_grid=True):
        self.path = path
        self.month_path_str = month_path_str
        self.file_search_str = file_str
        super(H14img, self).__init__(path, grid=None)
        self.expand_grid = expand_grid

    def _read_spec_img(self, timestamp):
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
        filename = os.path.join(self.path, timestamp.strftime(self.month_path_str),
                                timestamp.strftime(self.file_search_str))

        param_names = {'40': 'SM_layer1_0-7cm',
                       '41': 'SM_layer2_7-28cm',
                       '42': 'SM_layer3_28-100cm',
                       '43': 'SM_layer4_100-289cm'}
        data = {}

        with pygrib.open(filename) as grb:
            for i, message in enumerate(grb):
                message.expand_grid(self.expand_grid)
                if i == 1:
                    lats, lons = message.latlons()
                data[param_names[message['parameterName']]] = message.values

        return data, {}, timestamp, lons, lats, None
