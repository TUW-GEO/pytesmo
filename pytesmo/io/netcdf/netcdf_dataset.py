"""
Reads and writes netcdf time series according to the Climate Forecast Metadata
Conventions

Created on Dec 09, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
"""

import os
import numpy as np
import netCDF4
import datetime
import pandas as pd

import pytesmo.io.dataset_base as dsbase
import pytesmo.grid.grids as grids


class DatasetError(Exception):
    pass


class Dataset(object):

    """
    NetCDF file wrapper class that makes some things easier

    Parameters
    ----------
    filename : string
        filename of netCDF file. If already exiting then it will be opened
        as read only unless the append keyword is set. if the overwrite
        keyword is set then the file will be overwritten
    name : string, optional
        will be written as a global attribute if the file is a new file
    file_format : string, optional
        file format
    mode : string, optional
        access mode. default 'r'
        'r' means read-only; no data can be modified.
        'w' means write; a new file is created, an existing file with the
            same name is deleted.
        'a' and 'r+' mean append (in analogy with serial files); an existing
            file is opened for reading and writing.
        Appending s to modes w, r+ or a will enable unbuffered shared access
        to NETCDF3_CLASSIC or NETCDF3_64BIT formatted files. Unbuffered
        access may be useful even if you don't need shared access, since it
        may be faster for programs that don't access data sequentially.
        This option is ignored for NETCDF4 and NETCDF4_CLASSIC
        formatted files.
    zlib : boolean, optional
        Default True
        if set netCDF compression will be used
    complevel : int, optional
        Default 4
        compression level used from 1(low compression) to 9(high compression)
    """

    def __init__(self, filename, name=None, file_format="NETCDF4",
                 mode='r', zlib=True, complevel=4):

        self.dataset_name = name
        self.filename = filename
        self.file = None
        self.file_format = file_format
        self.buf_len = 0
        self.global_attr = {}
        self.global_attr['id'] = os.path.split(self.filename)[1]
        s = "%Y-%m-%d %H:%M:%S"
        self.global_attr['date_created'] = datetime.datetime.now().strftime(s)
        if self.dataset_name is not None:
            self.global_attr['dataset_name'] = self.dataset_name
        self.zlib = zlib
        self.complevel = complevel
        self.mode = mode

        if self.mode == "a" and not os.path.exists(self.filename):
            self.mode = "w"
        if self.mode == 'w':
            path = os.path.dirname(self.filename)
            if not os.path.exists(path):
                os.makedirs(path)

        self.dataset = netCDF4.Dataset(self.filename, self.mode,
                                       format=self.file_format)

    def _set_global_attr(self):
        """
        Write global attributes to NetCDF file.
        """
        self.dataset.setncatts(self.global_attr)
        self.global_attr = {}

    def create_dim(self, name, n):
        """
        Create dimension for NetCDF file.
        if it does not yet exist

        Parameters
        ----------
        name : str
            Name of the NetCDF dimension.
        n : int
            Size of the dimension.
        """
        if name not in self.dataset.dimensions.keys():
            self.dataset.createDimension(name, size=n)

    def write_var(self, name, data=None, dim=None, attr={}, dtype=None,
                  zlib=None, complevel=None, chunksizes=None, **kwargs):
        """
        Create or overwrite values in a NetCDF variable. The data will be
        written to disk once flush or close is called

        Parameters
        ----------
        name : str
            Name of the NetCDF variable.
        data : np.ndarray, optional
            Array containing the data.
            if not given then the variable will be left empty
        dim : tuple, optional
            A tuple containing the dimension names.
        attr : dict, optional
            A dictionary containing the variable attributes.
        dtype: data type, string or numpy.dtype, optional
            if not given data.dtype will be used
        zlib: boolean, optional
            explicit compression for this variable
            if not given then global attribute is used
        complevel: int, optional
            explicit compression level for this variable
            if not given then global attribute is used
        chunksizes : tuple, optional
            chunksizes can be used to manually specify the
            HDF5 chunksizes for each dimension of the variable.
        """

        fill_value = None
        if '_FillValue' in attr:
            fill_value = attr.pop('_FillValue')

        if dtype is None:
            dtype = data.dtype

        if zlib is None:
            zlib = self.zlib
        if complevel is None:
            complevel = self.complevel

        if name in self.dataset.variables.keys():
            var = self.dataset.variables[name]
        else:
            var = self.dataset.createVariable(name, dtype,
                                              dim, fill_value=fill_value,
                                              zlib=zlib, complevel=complevel,
                                              chunksizes=chunksizes, **kwargs)
        if data is not None:
            var[:] = data

        for attr_name, attr_value in attr.iteritems():
            self.dataset.variables[name].setncattr(attr_name, attr_value)

    def append_var(self, name, data):
        """
        append data along unlimited dimension(s) of variable

        Parameters
        ----------
        name : string
            name of variable to append to
        data : numpy.array
            numpy array of correct dimension

        Raises
        ------
        IOError
            if appending to variable without unlimited dimension
        """
        if name in self.dataset.variables.keys():
            var = self.dataset.variables[name]
            dim_unlimited = []
            key = []
            for index, dim in enumerate(var.dimensions):
                unlimited = self.dataset.dimensions[dim].isunlimited()
                dim_unlimited.append(unlimited)
                if not unlimited:
                    # if the dimension is not unlimited set the slice to :
                    key.append(slice(None, None, None))
                else:
                    # if unlimited set slice of this dimension to
                    # append meaning
                    # [var.shape[index]:]
                    key.append(slice(var.shape[index], None, None))

            dim_unlimited = np.array(dim_unlimited)
            nr_unlimited = np.where(dim_unlimited)[0].size
            key = tuple(key)
            # if there are unlimited dimensions we can do an append
            if nr_unlimited > 0:
                var[key] = data
            else:
                raise IOError(''.join(('Cannot append to variable that ',
                                       'has no unlimited dimension')))

    def read_var(self, name):
        """
        reads variable from netCDF file

        Parameters
        ----------
        name : string
            name of the variable
        """

        if self.mode in ['r', 'r+']:
            if name in self.dataset.variables.keys():
                return self.dataset.variables[name][:]

    def add_global_attr(self, name, value):
        self.global_attr[name] = value

    def flush(self):
        if self.dataset is not None:
            if self.mode in ['w', 'r+']:
                self._set_global_attr()
                self.dataset.sync()

    def close(self):
        if self.dataset is not None:
            self.flush()
            self.dataset.close()
            self.dataset = None

    def __enter__(self):
        return self

    def __exit__(self, value_type, value, traceback):
        self.close()


class OrthoMultiTs(Dataset):

    """
    Implementation of the Orthogonal multidimensional array representation
    of time series according to the NetCDF CF-conventions 1.6.

    Parameters
    ----------
    filename : string
        filename of netCDF file. If already exiting then it will be opened
        as read only unless the append keyword is set. if the overwrite
        keyword is set then the file will be overwritten
    n_loc : int, optional
        number of locations that this netCDF file contains time series for
        only required for new file
    loc_dim_name : string, optional
        name of the location dimension
    obs_dim_name : string, optional
        name of the observations dimension
    loc_ids_name : string, optional
        name of variable that has the location id's stored
    loc_descr_name : string, optional
        name of variable that has additional location information
        stored
    time_units : string, optional
        units the time axis is given in.
        Default: "days since 1900-01-01 00:00:00"
    time_var : string, optional
        name of time variable
        Default: time
    lat_var : string, optional
        name of latitude variable
        Default: lat
    lon_var : string, optional
        name of longitude variable
        Default: lon
    alt_var : string, optional
        name of altitude variable
        Default: alt
    unlim_chunksize : int, optional
        chunksize to use along unlimited dimensions, other chunksizes
        will be calculated by the netCDF library
    read_bulk : boolean, optional
        if set to True the data of all locations is read into memory,
        and subsequent calls to read_ts read from the cache and not from disk
        this makes reading complete files faster#
    read_dates : boolean, optional
        if false dates will not be read automatically but only on specific
        request useable for bulk reading because currently the netCDF
        num2date routine is very slow for big datasets
    """

    def __init__(self, filename, n_loc=None, loc_dim_name='locations',
                 obs_dim_name='time', loc_ids_name='location_id',
                 loc_descr_name='location_description',
                 time_units="days since 1900-01-01 00:00:00",
                 time_var='time', lat_var='lat', lon_var='lon', alt_var='alt',
                 unlim_chunksize=None, read_bulk=False, read_dates=True,
                 **kwargs):
        super(OrthoMultiTs, self).__init__(filename, **kwargs)

        self.n_loc = n_loc
        self.obs_dim_name = obs_dim_name
        self.loc_ids_name = loc_ids_name
        self.loc_dim_name = loc_dim_name
        self.loc_descr_name = loc_descr_name
        self.time_var = time_var
        self.lat_var = lat_var
        self.lon_var = lon_var
        self.alt_var = alt_var
        self.time_units = time_units
        self.unlim_chunksize = unlim_chunksize
        if unlim_chunksize is not None:
            self.unlim_chunksize = [unlim_chunksize]

        # variable to track write operations
        self.write_operations = 0
        self.write_offset = None

        # variable which lists the variables that should not be
        # considered time series even if they have the correct dimension
        self.not_timeseries = [self.time_var]

        # initialize dimensions and index_variable
        if self.mode == 'w':
            self._init_dimensions_and_lookup()
            self._init_location_variables()
            self._init_location_id_and_time()

            self.global_attr['featureType'] = 'timeSeries'

        # index, to be read upon first reading operation
        self.index = None
        # date variables, for OrthogonalMulitTs it can be stored
        # since it is the same for all variables in a file
        self.constant_dates = True
        self.dates = None
        self.read_dates_auto = read_dates
        self.read_bulk = read_bulk
        # if read bulk is activated the arrays will
        # be read into the local variables dict
        # if it is not activated the data will be read
        # from the netCDF variables
        if not self.read_bulk:
            self.variables = self.dataset.variables
        else:
            self.variables = {}

    def _init_dimensions_and_lookup(self):
        """
        Initializes the dimensions and variables for the lookup
        between locations and entries in the time series
        """
        if self.n_loc is None:
            raise ValueError('Number of locations '
                             'have to be set for new file')

        self.create_dim(self.loc_dim_name, self.n_loc)
        self.create_dim(self.obs_dim_name, None)

    def _init_location_id_and_time(self):
        """
        initialize the dimensions and variables that are the basis of
        the format
        """
        # make variable that contains the location id
        self.write_var(self.loc_ids_name, data=None, dim=self.loc_dim_name,
                       dtype=np.int)
        self.write_var(self.loc_descr_name, data=None, dim=self.loc_dim_name,
                       dtype='str')
        # initialize time variable
        self.write_var(self.time_var, data=None, dim=self.obs_dim_name,
                       attr={'standard_name': 'time',
                             'long_name': 'time of measurement',
                             'units': self.time_units},
                       dtype=np.double,
                       chunksizes=self.unlim_chunksize)

    def _init_location_variables(self):
        # write station information, longitude, latitude and altitude
        self.write_var(self.lon_var, data=None, dim=self.loc_dim_name,
                       attr={'standard_name': 'longitude',
                             'long_name': 'location longitude',
                             'units': 'degrees_east',
                             'valid_range': (-180.0, 180.0)},
                       dtype=np.float)
        self.write_var(self.lat_var, data=None, dim=self.loc_dim_name,
                       attr={'standard_name': 'latitude',
                             'long_name': 'location latitude',
                             'units': 'degrees_north',
                             'valid_range': (-90.0, 90.0)},
                       dtype=np.float)
        self.write_var(self.alt_var, data=None, dim=self.loc_dim_name,
                       attr={'standard_name': 'height',
                             'long_name': 'vertical distance above the '
                             'surface',
                             'units': 'm',
                             'positive': 'up',
                             'axis': 'Z'},
                       dtype=np.float)

    def _read_index(self):
        self.index = self.dataset.variables[self.loc_ids_name][:]

    def _find_free_index_pos(self):
        """
        if the index is not yet filled completely this function
        gets the id of the first free position

        This function depends on the masked array being used if no
        data is yet in the file

        Returns
        -------
        idx : int
            first free index position

        Raises
        ------
        DatasetError
            if no free index is found
        """
        if self.index is None:
            self._read_index()

        masked = np.where(self.index.mask)[0]
        # all indexes already filled
        if len(masked) == 0:
            raise DatasetError('No free index available')
        else:
            idx = np.min(masked)

        return idx

    def _get_loc_index(self, loc_id):
        """
        gets index of location id in index variable
        """
        if self.index is None:
            self._read_index()
        loc_ix = np.where(loc_id == self.index)[0]
        if loc_ix.size != 1:
            raise IOError('Index problem %d elements found' % loc_ix.size)
        return loc_ix[0]

    def _add_location(self, loc_id, lon, lat, alt=None, loc_descr=None):
        """
        add a new location to the dataset

        Paramters
        ---------
        loc_id : int or numpy.array
            location id
        lon : float or numpy.array
            longitudes of location
        lat : float or numpy.array
            longitudes of location
        alt : float or numpy.array
            altitude of location
        loc_descr : string or numpy.array
            location description
        """
        if type(loc_id) != np.ndarray:
            loc_id = np.array([loc_id])
        if type(lon) != np.ndarray:
            lon = np.array([lon])
        # netCDF library can not handle arrays of length 1 that contain onla a
        # None value
        if lon.size == 1 and lon[0] is None:
            lon = None
        if type(lat) != np.ndarray:
            lat = np.array([lat])
        # netCDF library can not handle arrays of length 1 that contain onla a
        # None value
        if lat.size == 1 and lat[0] is None:
            lat = None
        if alt is not None:
            if type(alt) != np.ndarray:
                alt = np.array([alt])
            # netCDF library can not handle arrays of length 1 that contain
            # onla a None value
            if alt.size == 1 and alt[0] is None:
                alt = None

        # remove location id's that are already in the file
        locations = np.ma.compressed(
            self.dataset.variables[self.loc_ids_name][:])
        loc_count = len(locations)
        if loc_count > 0:
            loc_ids_new = np.invert(np.in1d(loc_id, locations))
            if len(np.nonzero(loc_ids_new)[0]) == 0:
                # no new locations to add
                return None
        else:
            loc_ids_new = slice(None, None, None)
        idx = self._find_free_index_pos()
        index = np.arange(len(loc_id[loc_ids_new])) + idx
        self.dataset.variables[self.loc_ids_name][index] = loc_id[loc_ids_new]
        if lon is not None:
            self.dataset.variables[self.lon_var][index] = lon[loc_ids_new]
        if lat is not None:
            self.dataset.variables[self.lat_var][index] = lat[loc_ids_new]
        if alt is not None:
            self.dataset.variables[self.alt_var][index] = alt[loc_ids_new]
        if loc_descr is not None:
            if type(loc_descr) != np.ndarray:
                loc_descr = np.array(loc_descr)
            if len(index) == 1:
                index = int(index[0])
                loc_ids_new = 0
                self.dataset.variables[self.loc_descr_name][
                    index] = str(loc_descr)
            else:
                self.dataset.variables[self.loc_descr_name][
                    index] = loc_descr[loc_ids_new].astype(object)

        # update index variable after adding location
        self.index[index] = np.ma.asarray(loc_id)

        return idx

    def _get_all_ts_variables(self):
        """
        gets all variable names that have the self.obs_dim_name as only
        dimension indicating that they are time series observations. This
        does not include the self.time_var variable

        Returns
        -------
        variables : list
            list of variable names
        """
        ts_var = []

        for variable_name in self.dataset.variables:
            if variable_name not in self.not_timeseries:
                if self.obs_dim_name in \
                        self.dataset.variables[variable_name].dimensions:
                    ts_var.append(variable_name)

        return ts_var

    def _get_index_of_ts(self, loc_id):
        """
        get the indes of a time series
        """
        try:
            loc_ix = self._get_loc_index(loc_id)
        except IOError:
            msg = "".join(("Time series for Location #", loc_id.__str__(),
                           " not found."))
            raise IOError(msg)

        # [loc_ix,:]
        _slice = (loc_ix, slice(None, None, None))

        return _slice

    def _get_loc_ix_from_obs_ix(self, obs_ix):
        """
        Get location index from observation index. In case of OrthoMultiTs
        all measurements are taken at the same time and therefore all
        location id's are affected.

        Parameters
        ----------
        obs_ix : int
            Observation index.

        Returns
        -------
        loc_ix : int
            Location index.
        """
        return self.read_var(self.loc_ids_name)

    def read_time(self, loc_id):
        """
        read the time stamps for the given location id
        in this case the location id is irrelevant since they
        all have the same timestamps
        """
        return self.dataset.variables[self.time_var][:]

    def read_dates(self, loc_id):
        self.dates = netCDF4.num2date(self.read_time(loc_id),
                                      units=self.dataset.variables[
                                          self.time_var].units,
                                      calendar='standard')
        return self.dates

    def _read_var_ts(self, loc_id, var):
        """
        read a time series of a variable at a given location id

        Parameters
        ----------
        loc_id : int
            id of location, can be a grid point id or some other id
        var : string
            name of variable to read
        """
        index = self._get_index_of_ts(loc_id)

        if self.read_bulk:
            if var not in self.variables.keys():
                self.variables[var] = self.dataset.variables[var][:]

        return self.variables[var][index]

    def read_ts(self, variables, loc_id, dates_direct=False):
        """
        reads time series of variables

        Parameters
        ----------
        variables : list or string
        loc_id : int
            location_id
        dates_direct : boolean, optional
            if True the dates are read directly from the netCDF file
            without conversion to datetime
        """
        if type(variables) != list:
            variables = [variables]

        ts = {}
        for variable in variables:
            data = self._read_var_ts(loc_id, variable)
            ts[variable] = data

        if not dates_direct:
            # only read dates if they should be read automatically
            if self.read_dates_auto:
                # only read dates if they have not been read
                # or if they are different for each location id which is
                # the case if self.constant_dates is set to False
                if self.dates is None:
                    self.read_dates(loc_id)
                if not self.constant_dates:
                    self.read_dates(loc_id)
            ts['time'] = self.dates
        else:
            if self.read_dates_auto:
                # only read dates if they have not been read
                # or if they are different for each location id which is
                # the case if self.constant_dates is set to False
                ts['time'] = self.read_time(loc_id)

        return ts

    def read_all_ts(self, loc_id, dates_direct=False):
        """
        read a time series of all time series variables at a given location id

        Parameters
        ----------
        loc_id : int
            id of location, can be a grid point id or some other id
        dates_direct : boolean, optional
            if True the dates are read directly from the netCDF file
            without conversion to datetime

        Returns
        -------
        time_series : dict
            keys of var and time with numpy.arrays as values
        """
        ts = self.read_ts(self._get_all_ts_variables(), loc_id)
        return ts

        for variable in self._get_all_ts_variables():
            data = self._read_var_ts(loc_id, variable)
            ts[variable] = data

        if not dates_direct:
            # only read dates if they should be read automatically
            if self.read_dates_auto:
                # only read dates if they have not been read
                # or if they are different for each location id which is i
                # the case if self.constant_dates is set to False
                if self.dates is None:
                    self.read_dates(loc_id)
                if not self.constant_dates:
                    self.read_dates(loc_id)
            ts['time'] = self.dates
        else:
            if self.read_dates_auto:
                # only read dates if they have not been read
                # or if they are different for each location id which is
                # the case if self.constant_dates is set to False
                ts['time'] = self.read_time(loc_id)

        return ts

    def extend_time(self, dates, direct=False):
        """
        Extend the time dimension and variable by the given dates

        Parameters
        ----------
        dates : numpy.array of datetime objects or floats
        direct : boolean
            if true the dates are already converted into floating
            point number of correct magnitude
        """
        if direct:
            self.append_var(self.time_var, dates)
        else:
            self.append_var(self.time_var,
                            netCDF4.date2num(dates,
                                             units=self.dataset.variables[
                                                 self.time_var].units,
                                             calendar='standard'))

    def write_ts(self, loc_id, data, dates, extend_time='first',
                 loc_descr=None, lon=None, lat=None, alt=None,
                 fill_values=None, attributes=None, dates_direct=False):
        """
        write time series data, if not yet existing also add location to file
        for this data format it is assumed that in each write/append cycle
        the same amount of data is added.

        Parameters
        ----------
        loc_id : int
            location id
        data : dict
            dictionary with variable names as keys and numpy.arrays as values
        dates: numpy.array
            array of datetime objects
        extend_time : string or boolean, optional
            one of 'first', True or False
            'first' : only extend the time variable on the first write
                      operation after opening the file
            True : extend the time variable when writing
            False : only write variable and ignore dates
                    the assumption is that the time variable already has the
                    correct length and content
        attributes : dict, optional
            dictionary of attributes that should be added to the netCDF
            variables. can also be a dict of dicts for each variable name
            as in the data dict.
        dates_direct : boolean
            if true the dates are already converted into floating
            point number of correct magnitude
        """
        try:
            idx = self._get_loc_index(loc_id)
        except IOError:
            idx = self._add_location(loc_id, lon, lat, alt, loc_descr)

        # find out if attributes is a dict to be used for all variables or if
        # there is a dictionary of attributes for each variable
        unique_attr = False
        if attributes is not None:
            if sorted(data.keys()) == sorted(attributes.keys()):
                unique_attr = True

        for key in data:
            if data[key].size != dates.size:
                raise DatasetError("".join(("timestamps and dataset %s ",
                                            "must have the same size" % key)))

        # add to time variable only on the first write operation
        if ((self.write_operations == 0 and extend_time == 'first') or
                (type(extend_time) == bool and extend_time)):
            self.length_before_extend = self.dataset.variables[
                self.time_var].size
            self.extend_time(dates, direct=dates_direct)

        for key in data:

            internal_attributes = {'name': key,
                                   'coordinates': 'lat lon alt'}

            if type(fill_values) == dict:
                internal_attributes['_FillValue'] = fill_values[key]

            if attributes is not None:
                if unique_attr:
                    variable_attributes = attributes[key]
                else:
                    variable_attributes = attributes

                internal_attributes.update(variable_attributes)

            if self.unlim_chunksize is None:
                chunksizes = None
            else:
                chunksizes = [self.n_loc, self.unlim_chunksize[0]]
            self.write_var(key, data=None, dim=(self.loc_dim_name,
                                                self.obs_dim_name),
                           attr=internal_attributes,
                           dtype=data[key].dtype, chunksizes=chunksizes)

            if self.write_offset is None:
                # find start of elements that are not yet filled with values
                _slice_new = slice(self.length_before_extend, None, None)
                masked = \
                    np.where(
                        self.dataset.variables[key][idx, _slice_new].mask)[0]
                # all indexes already filled
                if len(masked) == 0:
                    raise DatasetError('No free data slots available')
                else:
                    self.write_offset = np.min(
                        masked) + self.length_before_extend

            _slice = slice(self.write_offset, None, None)
            # has to be reshaped to 2 dimensions because it is written
            # into 2d variable otherwise netCDF library gets confused,
            # might be a bug in netCDF?
            self.dataset.variables[key][idx, _slice] = \
                data[key].reshape(1, data[key].size)

        self.write_operations += 1

    def write_ts_all_loc(self, loc_ids, data, dates, loc_descrs=None,
                         lons=None, lats=None, alts=None, fill_values=None,
                         attributes=None, dates_direct=False):
        """
        write time series data in bulk, for this the user has to provide
        a 2D array with dimensions (self.nloc, dates) that is filled with
        the time series of all grid points in the file

        Parameters
        ----------
        loc_ids : numpy.array
            location ids along the first axis of the data array
        data : dict
            dictionary with variable names as keys and 2D numpy.arrays as
            values
        dates: numpy.array
            array of datetime objects with same size as second dimension of
            data arrays
        attributes : dict, optional
            dictionary of attributes that should be added to the netCDF
            variables. can also be a dict of dicts for each variable name as
            in the data dict.
        dates_direct : boolean
            if true the dates are already converted into floating
            point number of correct magnitude
        """

        if self.n_loc != loc_ids.size:
            raise ValueError(''.join(('loc_ids is not the same number of ',
                                      'locations in the file')))
        for key in data:
            if data[key].shape[1] != dates.size:
                raise DatasetError("".join(("timestamps and dataset ",
                                            "second dimension %s must have ",
                                            " the same size" % key)))
            if data[key].shape[0] != self.n_loc:
                raise DatasetError("".join(("datasets first dimension ",
                                            "%s must have the same size as ",
                                            "number of locations ",
                                            "in the file" % key)))

        # make sure zip works even if one of the parameters is not given
        if lons is None:
            lons = np.repeat(None, self.n_loc)
        if lats is None:
            lats = np.repeat(None, self.n_loc)
        if alts is None:
            alts = np.repeat(None, self.n_loc)

        # find out if attributes is a dict to be used for all variables or if
        # there is a dictionary of attributes for each variable
        unique_attr = False
        if attributes is not None:
            if sorted(data.keys()) == sorted(attributes.keys()):
                unique_attr = True

        self._add_location(loc_ids, lons, lats, alts, loc_descrs)

        for key in data:

            internal_attributes = {'name': key,
                                   'coordinates': 'lat lon alt'}

            if type(fill_values) == dict:
                internal_attributes['_FillValue'] = fill_values[key]

            if attributes is not None:
                if unique_attr:
                    variable_attributes = attributes[key]
                else:
                    variable_attributes = attributes

                internal_attributes.update(variable_attributes)

            if self.unlim_chunksize is None:
                chunksizes = None
            else:
                chunksizes = [self.n_loc, self.unlim_chunksize[0]]
            self.write_var(key, data=None, dim=(self.loc_dim_name,
                                                self.obs_dim_name),
                           attr=internal_attributes,
                           dtype=data[key].dtype, chunksizes=chunksizes)

            if self.write_offset is None:
                # current shape tells us how many elements are already
                # in the file
                self.write_offset = self.dataset.variables[key].shape[1]

            _slice = slice(self.write_offset, self.write_offset + dates.size,
                           None)
            self.dataset.variables[key][:, _slice] = data[key]

        # fill up time variable
        if dates_direct:
            self.dataset.variables[self.time_var][self.write_offset:] = dates
        else:
            self.dataset.variables[self.time_var][self.write_offset:] = \
                netCDF4.date2num(dates,
                                 units=self.dataset.variables[
                                     self.time_var].units,
                                 calendar='standard')


class ContiguousRaggedTs(OrthoMultiTs):

    """
    Class that represents a Contiguous ragged array representation of
    time series according to NetCDF CF-conventions 1.6.

    Parameters
    ----------
    filename : string
        filename of netCDF file. If already exiting then it will be opened
        as read only unless the append keyword is set. if the overwrite
        keyword is set then the file will be overwritten
    n_loc : int, optional
        number of locations that this netCDF file contains time series for
        only required for new file
    n_obs : int, optional
        how many observations will be saved into this netCDF file in total
        only required for new file
    obs_loc_lut : string, optional
        variable name in the netCDF file that contains the lookup between
        observations and locations
    loc_dim_name : string, optional
        name of the location dimension
    obs_dim_name : string, optional
        name of the observations dimension
    loc_ids_name : string, optional
        name of variable that has the location id's stored
    loc_descr_name : string, optional
        name of variable that has additional location information
        stored
    time_units : string, optional
        units the time axis is given in.
        Default: "days since 1900-01-01 00:00:00"
    time_var : string, optional
        name of time variable
        Default: time
    lat_var : string, optional
        name of latitude variable
        Default: lat
    lon_var : string, optional
        name of longitude variable
        Default: lon
    alt_var : string, optional
        name of altitude variable
        Default: alt
    """

    def __init__(self, filename, n_loc=None, n_obs=None,
                 obs_loc_lut='row_size', obs_dim_name='obs', **kwargs):

        self.n_obs = n_obs
        self.obs_loc_lut = obs_loc_lut

        super(ContiguousRaggedTs, self).__init__(filename, n_loc=n_loc,
                                                 obs_dim_name=obs_dim_name,
                                                 ** kwargs)
        self.constant_dates = False

    def _init_dimensions_and_lookup(self):
        """
        Initializes the dimensions and variables for the lookup
        between locations and entries in the time series
        """
        if self.n_loc is None:
            raise ValueError('Number of locations '
                             'have to be set for new file')
        if self.n_obs is None:
            raise ValueError('Number of observations have '
                             'to be set for new file')

        self.create_dim(self.loc_dim_name, self.n_loc)
        self.create_dim(self.obs_dim_name, self.n_obs)

        attr = {'long_name': 'number of observations at this location',
                'sample_dimension': self.obs_dim_name}
        self.write_var(self.obs_loc_lut, data=None, dim=self.loc_dim_name,
                       dtype=np.int, attr=attr,
                       chunksizes=self.unlim_chunksize)

    def _get_index_of_ts(self, loc_id):
        """
        Parameters
        ----------
        loc_id: int

        Raises
        ------
        IOError
            if location id could not be found
        """
        try:
            loc_ix = self._get_loc_index(loc_id)
        except IOError:
            msg = "".join(("Time series for Location #", loc_id.__str__(),
                           " not found."))
            raise IOError(msg)

        start = np.sum(self.variables[self.obs_loc_lut][:loc_ix])
        end = np.sum(self.variables[self.obs_loc_lut][:loc_ix + 1])

        return slice(start, end)

    def _get_loc_ix_from_obs_ix(self, obs_ix):
        """
        Get location index from observation index.

        Parameters
        ----------
        obs_ix : int
            Observation index.

        Returns
        -------
        loc_ix : int
            Location index.
        """
        bins = np.hstack((0, np.cumsum(self.variables[self.obs_loc_lut])))
        loc_ix = np.digitize(obs_ix, bins) - 1

        return loc_ix

    def read_time(self, loc_id):
        """
        read the time stamps for the given location id
        in this case it works like a normal time series variable
        """
        return self._read_var_ts(loc_id, self.time_var)

    def write_ts(self, loc_id, data, dates, loc_descr=None, lon=None,
                 lat=None, alt=None, fill_values=None, attributes=None,
                 dates_direct=False):
        """
        write time series data, if not yet existing also add location to file

        Parameters
        ----------
        loc_id : int
            location id
        data : dict
            dictionary with variable names as keys and numpy.arrays as values
        dates: numpy.array
            array of datetime objects
        attributes : dict, optional
            dictionary of attributes that should be added to the netCDF
            variables. can also be a dict of dicts for each variable name
            as in the data dict.
        dates_direct : boolean
            if true the dates are already converted into floating
            point number of correct magnitude
        """
        try:
            idx = self._get_loc_index(loc_id)
        except IOError:
            idx = self._add_location(loc_id, lon, lat, alt, loc_descr)

        # find out if attributes is a dict to be used for all variables or if
        # there is a dictionary of attributes for each variable
        unique_attr = False
        if attributes is not None:
            if sorted(data.keys()) == sorted(attributes.keys()):
                unique_attr = True

        for key in data:
            if data[key].size != dates.size:
                raise DatasetError("".join(("timestamps and dataset %s ",
                                            "must have the same size" % key)))

        # add number of new elements to index_var
        self.dataset.variables[self.obs_loc_lut][idx] = dates.size

        index = self._get_index_of_ts(loc_id)
        for key in data:

            internal_attributes = {'name': key,
                                   'coordinates': 'time lat lon alt'}

            if type(fill_values) == dict:
                internal_attributes['_FillValue'] = fill_values[key]

            if attributes is not None:
                if unique_attr:
                    variable_attributes = attributes[key]
                else:
                    variable_attributes = attributes

                internal_attributes.update(variable_attributes)

            self.write_var(key, data=None, dim=self.obs_dim_name,
                           attr=internal_attributes,
                           dtype=data[key].dtype,
                           chunksizes=self.unlim_chunksize)
            self.dataset.variables[key][index] = data[key]

        if dates_direct:
            self.dataset.variables[self.time_var][index] = dates
        else:
            self.dataset.variables[self.time_var][index] = \
                netCDF4.date2num(dates,
                                 units=self.dataset.variables[
                                     self.time_var].units,
                                 calendar='standard')


class IndexedRaggedTs(ContiguousRaggedTs):

    """
    Class that represents a Indexed ragged array representation of time series
    according to NetCDF CF-conventions 1.6.
    """

    def __init__(self, filename, n_loc=None, obs_loc_lut='locationIndex',
                 **kwargs):
        # set n_obs to None for unlimited dimension
        super(IndexedRaggedTs, self).__init__(filename, n_loc=n_loc,
                                              n_obs=None,
                                              obs_loc_lut=obs_loc_lut,
                                              **kwargs)
        self.not_timeseries.append(self.obs_loc_lut)
        self.constant_dates = False

    def _init_dimensions_and_lookup(self):
        """
        Initializes the dimensions and variables for the lookup
        between locations and entries in the time series
        """
        if self.n_loc is None:
            raise ValueError('Number of locations '
                             'have to be set for new file')

        self.create_dim(self.loc_dim_name, self.n_loc)
        self.create_dim(self.obs_dim_name, self.n_obs)
        attr = {'long_name': 'which location this observation is for',
                'instance_dimension': self.loc_dim_name}
        self.write_var(self.obs_loc_lut, data=None, dim=self.obs_dim_name,
                       dtype=np.int, attr=attr,
                       chunksizes=self.unlim_chunksize)

    def _get_index_of_ts(self, loc_id):
        """
        Parameters
        ----------
        loc_id: int
            Location index.

        Raises
        ------
        IOError
            if location id could not be found
        """
        try:
            loc_ix = self._get_loc_index(loc_id)
        except IOError:
            msg = "".join(("Time series for Location #", loc_id.__str__(),
                           " not found."))
            raise IOError(msg)

        index = np.where(self.variables[self.obs_loc_lut] == loc_ix)[0]

        if len(index) == 0:
            msg = "".join(("Time series for Location #", loc_id.__str__(),
                           " not found."))
            raise IOError(msg)

        return index

    def _get_loc_ix_from_obs_ix(self, obs_ix):
        """
        Get location index from observation index.

        Parameters
        ----------
        obs_ix : int
            Observation index.

        Returns
        -------
        loc_ix : int
            Location index.
        """
        return self.variables[self.obs_loc_lut][obs_ix]

    def write_ts(self, loc_id, data, dates, loc_descr=None, lon=None,
                 lat=None, alt=None, fill_values=None, attributes=None,
                 dates_direct=False):
        """
        write time series data, if not yet existing also add location to file

        Parameters
        ----------
        loc_id : int
            location id
        data : dict
            dictionary with variable names as keys and numpy.arrays as values
        dates: numpy.array
            array of datetime objects
        attributes : dict, optional
            dictionary of attributes that should be added to the netCDF
            variables. can also be a dict of dicts for each variable name as
            in the data dict.
        dates_direct : boolean
            if true the dates are already converted into floating
            point number of correct magnitude
        """
        try:
            idx = self._get_loc_index(loc_id)
        except IOError:
            idx = self._add_location(loc_id, lon, lat, alt, loc_descr)

        # find out if attributes is a dict to be used for all variables or if
        # there is a dictionary of attributes for each variable
        unique_attr = False
        if attributes is not None:
            if sorted(data.keys()) == sorted(attributes.keys()):
                unique_attr = True

        for key in data:
            if data[key].size != dates.size:
                raise DatasetError("".join(("timestamps and dataset %s ",
                                            "must have the same size" % key)))

        # add number of new elements to index_var
        indices = np.empty_like(dates, dtype=np.int)
        indices.fill(idx)
        self.append_var(self.obs_loc_lut, indices)

        index = self._get_index_of_ts(loc_id)

        # if this is the case the data is an append and only the last
        # dates.size elements are needed
        if index.size > dates.size:
            index = index[index.size - dates.size:]

        for key in data:

            internal_attributes = {'name': key,
                                   'coordinates': 'time lat lon alt'}

            if type(fill_values) == dict:
                internal_attributes['_FillValue'] = fill_values[key]

            if attributes is not None:
                if unique_attr:
                    variable_attributes = attributes[key]
                else:
                    variable_attributes = attributes

                internal_attributes.update(variable_attributes)

            # does nothing if variable exists already
            self.write_var(key, data=None, dim=self.obs_dim_name,
                           attr=internal_attributes,
                           dtype=data[key].dtype,
                           chunksizes=self.unlim_chunksize)

            self.dataset.variables[key][index] = data[key]

        if dates_direct:
            self.dataset.variables[self.time_var][index] = dates
        else:
            self.dataset.variables[self.time_var][index] = \
                netCDF4.date2num(dates,
                                 units=self.dataset.variables[
                                     self.time_var].units,
                                 calendar='standard')


class NetCDFGriddedTS(dsbase.DatasetTSBase):

    """
    Base class for reading time series data on a cell grid
    written with one of the netCDF writers
    in general.io.netcdf

    Parameters
    ----------
    grid : grid object
        that implements find_nearest_gpi() and gpi2cell()
    read_bulk : boolean, optional
        if true read_bulk will be activated
    data_path : string, optional
        path to the data directory
    parameters : list, optional
        if given only parameters from this list will be read
    ioclass : class
        class to use to read the data
    offsets : dict, optional
        offset to apply to a variable, in addition to the offset
        specified in the netCDF file
    scale_factors : dict, optional
        scale factors to apply to a variable

    """

    def __init__(self, grid=None, read_bulk=False,
                 data_path=None, parameters=None,
                 ioclass=None, cell_filename_template='%04d.nc',
                 offsets=None, scale_factors=None):

        self.parameters = parameters
        self.ioclass = ioclass
        self.netcdf_obj = None
        self.cell_file_templ = cell_filename_template
        self.read_bulk = read_bulk
        self.previous_cell = None
        self.offsets = offsets
        self.scale_factors = scale_factors
        if self.ioclass == OrthoMultiTs:
            self.read_dates = False
        else:
            self.read_dates = True
        self.dates = None
        super(NetCDFGriddedTS, self).__init__(data_path, grid)

    def read_gp(self, gpi, period=None):
        """
        Method reads data for given gpi

        Parameters
        ----------
        gpi : int
            grid point index on dgg grid
        period : list
            2 element array containing datetimes [start,end]

        Returns
        -------
        ts : pandas.DataFrame
            time series
        """
        cell = self.grid.gpi2cell(gpi)
        filename = os.path.join(self.path, self.cell_file_templ % cell)
        if self.read_bulk:
            if self.previous_cell != cell:
                # print "Switching cell to %04d reading gpi %d" % (cell, gpi)
                if self.netcdf_obj is not None:
                    self.netcdf_obj.close()
                    self.netcdf_obj = None
                self.netcdf_obj = self.ioclass(filename,
                                               read_bulk=self.read_bulk,
                                               read_dates=self.read_dates)
                self.previous_cell = cell
        else:
            if self.netcdf_obj is not None:
                self.netcdf_obj.close()
                self.netcdf_obj = None
            self.netcdf_obj = self.ioclass(filename, read_bulk=self.read_bulk,
                                           read_dates=self.read_dates)

        if self.parameters is None:
            data = self.netcdf_obj.read_all_ts(gpi)
        else:
            data = self.netcdf_obj.read_ts(self.parameters, gpi)

        if self.dates is None or self.read_dates:
            self.dates = self.netcdf_obj.read_dates(gpi)
        time = self.dates

        # remove time column from dataframe, only index should contain time
        try:
            data.pop('time')
        except KeyError:
            # if the time value is not found then do nothing
            pass

        ts = pd.DataFrame(data, index=time)

        if period is not None:
            ts = ts[period[0]:period[1]]

        if self.scale_factors is not None:
            for scale_column in self.scale_factors:
                if scale_column in ts.columns:
                    ts[scale_column] *= self.scale_factors[scale_column]

        if self.offsets is not None:
            for offset_column in self.offsets:
                if offset_column in ts.columns:
                    ts[offset_column] += self.offsets[offset_column]

        if not self.read_bulk:
            self.netcdf_obj.close()
            self.netcdf_obj = None
        return ts


class netCDF2DImageStack(Dataset):

    """
    Class for writing stacks of 2D images into netCDF.
    """

    def __init__(self, filename, grid=None, times=None,
                 mode='r', name=''):
        self.grid = grid
        if len(grid.shape) != 2:
            raise ValueError("grid must be 2D grid for a imagestack")
        self.filename = filename
        self.times = times
        self.variables = []
        self.time_var = 'time'
        self.time_units = "days since 1900-01-01"
        self.time_chunksize = 1
        self.lon_chunksize = 1
        self.lat_chunksize = len(self.grid.latdim)
        super(netCDF2DImageStack, self).__init__(filename, name=name,
                                                 mode=mode)

        if self.mode == 'w':
            self._init_dimensions()
            self._init_time()
            self._init_location_variables()
        elif self.mode in ['a', 'r']:
            self._load_grid()
            self._load_variables()

    def _init_dimensions(self):
        self.create_dim('lon', len(self.grid.londim))
        self.create_dim('lat', len(self.grid.latdim))
        self.create_dim('time', len(self.times))

    def _load_grid(self):
        lons = self.dataset.variables['lon'][:]
        lats = self.dataset.variables['lat'][:]
        self.grid = grids.gridfromdims(lons, lats)

    def _load_variables(self):
        for var in self.dataset.variables:
            if self.dataset.variables[var].dimensions == ('time', 'lat', 'lon'):
                self.variables.append(var)

    def _init_time(self):
        """
        initialize the dimensions and variables that are the basis of
        the format
        """
        # initialize time variable
        time_data = netCDF4.date2num(self.times, self.time_units)
        self.write_var(self.time_var, data=time_data, dim='time',
                       attr={'standard_name': 'time',
                             'long_name': 'time of measurement',
                             'units': self.time_units},
                       dtype=np.double,
                       chunksizes=[self.time_chunksize])

    def _init_location_variables(self):
        # write station information, longitude, latitude and altitude
        self.write_var('lon', data=self.grid.londim, dim='lon',
                       attr={'standard_name': 'longitude',
                             'long_name': 'location longitude',
                             'units': 'degrees_east',
                             'valid_range': (-180.0, 180.0)},
                       dtype=np.float)
        self.write_var('lat', data=self.grid.latdim, dim='lat',
                       attr={'standard_name': 'latitude',
                             'long_name': 'location latitude',
                             'units': 'degrees_north',
                             'valid_range': (-90.0, 90.0)},
                       dtype=np.float)

    def init_variable(self, var):
        self.write_var(var, data=None, dim=('time', 'lat', 'lon'),
                       dtype=np.float,
                       attr={'_FillValue': -9999.})

    def write_ts(self, gpi, data):
        """
        write a time series into the imagestack
        at the given gpi

        Parameters
        ----------
        self: type
            description
        gpi: int or numpy.array
            grid point indices to write to
        data: dictionary
            dictionary of int or numpy.array for each variable
            that should be written
            shape must be (len(gpi), len(times))
        """
        gpi = np.atleast_1d(gpi)

        for i, gp in enumerate(gpi):
            row, column = self.grid.gpi2rowcol(gp)
            for var in data:
                if var not in self.variables:
                    self.variables.append(var)
                    self.init_variable(var)
                self.dataset.variables[var][
                    :, row, column] = np.atleast_2d(data[var])[i, :]

    def __setitem__(self, gpi, data):
        """
        write a time series into the imagestack
        at the given gpi

        Parameters
        ----------
        self: type
            description
        gpi: int or numpy.array
            grid point indices to write to
        data: dictionary
            dictionary of int or numpy.array for each variable
            that should be written
            shape must be (len(gpi), len(times))
        """
        gpi = np.atleast_1d(gpi)

        for i, gp in enumerate(gpi):
            row, column = self.grid.gpi2rowcol(gp)
            for var in data:
                if var not in self.variables:
                    self.variables.append(var)
                    self.init_variable(var)
                self.dataset.variables[var][
                    :, row, column] = np.atleast_2d(data[var])[i, :]

    def __getitem__(self, key):

        gpi = np.atleast_1d(key)
        data = {}
        for i, gp in enumerate(gpi):
            row, column = self.grid.gpi2rowcol(gp)
            for var in self.variables:
                data[var] = self.dataset.variables[var][
                    :, row, column]

        return pd.DataFrame(data, index=self.times)
