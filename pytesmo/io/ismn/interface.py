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
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Aug 5, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''

import pytesmo.io.ismn.metadata_collector as metadata_collector
import pytesmo.io.ismn.readers as readers
import pygeogrids.grids as grids

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


try:
    from mpl_toolkits.basemap import Basemap
    basemap_installed = True
    from matplotlib.patches import Rectangle
except ImportError:
    basemap_installed = False


class ISMNError(Exception):
    pass


class ISMN_station(object):

    """
    Knows everything about the station, like which variables are measured there in which depths
    and in which files the data is stored. This is not completely true for the CEOP format
    since depth_from and depth_to are not easily knowable without parsing the whole file.
    For CEOP format depth_from and depth_to will only contain the phrase 'multiple' instead
    of the actual depth

        Parameters
    ----------
    metadata : numpy.array
        part of the structured array from metadata_collector.collect_from_folder()
        which contains only fields for one station

    Attributes
    ----------
    network : string
        network the time series belongs to
    station : string
        station name the time series belongs to
    latitude : float
        latitude of station
    longitude : float
        longitude of station
    elevation : float
        elevation of station
    variables : numpy.array
        variables measured at this station
        one of
                * 'soil moisture',
                * 'soil temperature',
                * 'soil suction',
                * 'precipitation',
                * 'air temperature',
                * 'field capacity',
                * 'permanent wilting point',
                * 'plant available water',
                * 'potential plant available water',
                * 'saturation',
                * 'silt fraction',
                * 'snow depth',
                * 'sand fraction',
                * 'clay fraction',
                * 'organic carbon',
                * 'snow water equivalent',
                * 'surface temperature',
                * 'surface temperature quality flag original'
    depth_from : numpy.array
        shallower depth of layer the variable with same index was measured at
    depth_to : numpy.array
        deeper depth of layer the variable with same index was measured at
    sensors : numpy.array
        sensor names of variables
    filenames : numpy.array
        filenames in which the data is stored

    Methods
    -------
    get_variables()
        returns the variables measured at this station
    get_depths(variable)
        get the depths in which a variable was measured at this station
    get_sensors(variable,depth_from,depth_to)
        get the sensors for the given variable, depth combination
    read_variable(variable,depth_from=None,depth_to=None,sensor=None)
        read the data for the given parameter combination
    """

    def __init__(self, metadata):
        """
        Goes through the passed metadata array and fills the attributes accordingly.
        """
        self.network = None
        self.station = None
        self.latitude = None
        self.longitude = None
        self.elevation = None
        self.variables = []
        self.depth_from = []
        self.depth_to = []
        self.sensors = []
        self.filenames = []

        for dataset in metadata:
            if self.network is None:
                self.network = dataset['network']
            elif self.network != dataset['network']:
                raise ISMNError(
                    "Networks in Station metadata are not the same")
            if self.station is None:
                self.station = dataset['station']
            elif self.station != dataset['station']:
                raise ISMNError(
                    "Station names in Station metadata are not the same")
            self.latitude = dataset['latitude']
            self.longitude = dataset['longitude']
            self.elevation = dataset['elevation']
            self.variables.append(dataset['variable'])
            self.depth_from.append(dataset['depth_from'])
            self.depth_to.append(dataset['depth_to'])
            self.sensors.append(dataset['sensor'])
            self.filenames.append(dataset['filename'])

        self.variables = np.array(self.variables)
        self.depth_from = np.array(self.depth_from)
        self.depth_to = np.array(self.depth_to)
        self.sensors = np.array(self.sensors)
        self.filenames = np.array(self.filenames)

    def get_variables(self):
        """
        get a list of variables measured at this station

        Returns
        -------
        variables : numpy.array
            array of variables measured at this station
        """
        return np.unique(self.variables)

    def get_depths(self, variable):
        """
        get depths at which the given variable was measured at this station

        Parameters
        ----------
        variable : string
            variable string best one of those returned by get_variables() or
            one of
                * 'soil moisture',
                * 'soil temperature',
                * 'soil suction',
                * 'precipitation',
                * 'air temperature',
                * 'field capacity',
                * 'permanent wilting point',
                * 'plant available water',
                * 'potential plant available water',
                * 'saturation',
                * 'silt fraction',
                * 'snow depth',
                * 'sand fraction',
                * 'clay fraction',
                * 'organic carbon',
                * 'snow water equivalent',
                * 'surface temperature',
                * 'surface temperature quality flag original'

        Returns
        -------
        depth_from : numpy.array
        depth_to : numpy.array
        """
        if variable in self.variables:
            index = np.where(variable == self.variables)
            return self.depth_from[index], self.depth_to[index]
        else:
            return None, None

    def get_sensors(self, variable, depth_from, depth_to):
        """
        get the sensors at which the variable was measured at the
        given depth

        Parameters
        ----------
        variable : string
            variable abbreviation
            one of
                * 'soil moisture',
                * 'soil temperature',
                * 'soil suction',
                * 'precipitation',
                * 'air temperature',
                * 'field capacity',
                * 'permanent wilting point',
                * 'plant available water',
                * 'potential plant available water',
                * 'saturation',
                * 'silt fraction',
                * 'snow depth',
                * 'sand fraction',
                * 'clay fraction',
                * 'organic carbon',
                * 'snow water equivalent',
                * 'surface temperature',
                * 'surface temperature quality flag original'
        depth_from : float
            shallower depth of layer the variable was measured at
        depth_to : float
            deeper depth of layer the variable was measured at

        Returns
        -------
        sensors : numpy.array
            array of sensors found for the given combination of variable and depths

        Raises
        ------
        ISMNError
            if no sensor was found for the given combination of variable and depths
        """
        ind_sensors = np.where((variable == self.variables) &
                               (depth_from == self.depth_from) &
                               (depth_to == self.depth_to))[0]

        if ind_sensors.size == 0:
            raise ISMNError("variable-depth_from-depth_to combination does not exist."
                            "Please check which depths do exist with get_depths_for_variable")
        else:
            return self.sensors[ind_sensors]

    def read_variable(self, variable, depth_from=None, depth_to=None, sensor=None):
        """
        actually reads the given variable from the file. Parameters are
        required until any ambiguity is resolved. If there is only one depth for
        the given variable then only variable is required. If there are multiple
        depths at least depth_from is required. If there are multiple depth_to
        possibilities for one variable-depth_from combination also depth_to has to
        be specified. If 2 sensors are measuring the same variable in the same
        depth then also the sensor has to be specified.

        Parameters
        ----------
        variable: string
            variable to read
            one of
                * 'soil moisture',
                * 'soil temperature',
                * 'soil suction',
                * 'precipitation',
                * 'air temperature',
                * 'field capacity',
                * 'permanent wilting point',
                * 'plant available water',
                * 'potential plant available water',
                * 'saturation',
                * 'silt fraction',
                * 'snow depth',
                * 'sand fraction',
                * 'clay fraction',
                * 'organic carbon',
                * 'snow water equivalent',
                * 'surface temperature',
                * 'surface temperature quality flag original'
        depth_from : float, optional
            shallower depth of layer the variable was measured at
        depth_to : float, optional
            deeper depth of layer the variable was measured at
        sensor : string, optional
            name of the sensor

        Returns
        -------
        data : readers.ISMNTimeSeries
            ISMNTimeSeries object containing the relevant metadata for the time series
            as well as a .data pointing to a pandas.DataFrame

        Raises
        ------
        ISMNError:
            if not all ambiguity was resolved by the given input parameters or
            if no data was found for the given input parameters

        """
        if depth_from is None:
            depth_f, depth_t = self.get_depths(variable)
            if depth_f.size > 1:
                raise ISMNError("there are multiple depths for this variable"
                                "Please specify the one you want to read")
            elif depth_f.size == 1:
                depth_from = depth_f[0]
            elif depth_f.size == 0:
                raise ISMNError("there are no depths for this variable"
                                "Something went wrong")
        if depth_to is None:
            depth_f, depth_t = self.get_depths(variable)
            if depth_t.size > 1:
                raise ISMNError("there are multiple depths with the same depth_from value"
                                "Please specify the depth_to value you want to read")
            elif depth_t.size == 1:
                depth_to = depth_t[0]
            elif depth_t.size == 0:
                raise ISMNError("there are no depths for this variable"
                                "Something went wrong")

        if sensor is None:
            sensors = self.get_sensors(variable, depth_from, depth_to)
            if sensors.size > 1:
                raise ISMNError("there are multiple sensors for this combination of "
                                "variable, depth_to, depth_from. Please specify which one "
                                "you want to read")
            elif sensors.size == 1:
                sensor = sensors[0]
            elif sensors.size == 0:
                raise ISMNError("there are no sensors for this variable, depth_from, depth_to "
                                "combination. Please make sure you specified valid depths")

        index_filename = np.where((variable == self.variables) &
                                  (depth_from == self.depth_from) &
                                  (depth_to == self.depth_to) &
                                  (sensor == self.sensors))[0]

        if index_filename.size != 1:
            raise ISMNError("There is no data for this combination of variable, depth_from, "
                            "depth_to and sensor. Please check.")
        else:
            return readers.read_data(self.filenames[index_filename[0]])

    def data_for_variable(self, variable, min_depth=None, max_depth=None):
        """
        function to go through all the depth_from, depth_to, sensor combinations
        for the given variable and yields ISMNTimeSeries if a match is found.
        if min_depth and/or max_depth where given it only returns a
        ISMNTimeSeries if depth_from >= min_depth and/or depth_to <= max_depth

        Parameters
        ----------
        variable: string
            variable to read
            one of
                * 'soil moisture',
                * 'soil temperature',
                * 'soil suction',
                * 'precipitation',
                * 'air temperature',
                * 'field capacity',
                * 'permanent wilting point',
                * 'plant available water',
                * 'potential plant available water',
                * 'saturation',
                * 'silt fraction',
                * 'snow depth',
                * 'sand fraction',
                * 'clay fraction',
                * 'organic carbon',
                * 'snow water equivalent',
                * 'surface temperature',
                * 'surface temperature quality flag original'
        min_depth : float, optional
            depth_from of variable has to be >= min_depth in order to be
            included.
        max_depth : float, optional
            depth_to of variable has to be <= max_depth in order to be
            included.

        Returns
        -------
        time_series : iterator(pytesmo.io.ismn.readers.ISMNTimeSeries)
            ISMNTimeSeries object containing data and metadata
        """

        if min_depth is None:
            min_depth = np.min(self.depth_from)
        if max_depth is None:
            max_depth = np.max(self.depth_to)

        for var, d1, d2, filename in zip(self.variables, self.depth_from, self.depth_to, self.filenames):
            if var != variable:
                continue

            if ((d1 >= min_depth) &
                    (d2 <= max_depth)):

                yield readers.read_data(filename)

    def get_min_max_obs_timestamp(self, variable="soil moisture", min_depth=None, max_depth=None):
        """
        goes throug the filenames associated with a station
        and reads the date of the first and last observation to get
        and approximate time coverage of the station.
        This is just an overview. If holes have to be detected the
        complete file must be read.

        Parameters
        ----------
        self: type
            description
        variable: string, optional
            one of
                * 'soil moisture',
                * 'soil temperature',
                * 'soil suction',
                * 'precipitation',
                * 'air temperature',
                * 'field capacity',
                * 'permanent wilting point',
                * 'plant available water',
                * 'potential plant available water',
                * 'saturation',
                * 'silt fraction',
                * 'snow depth',
                * 'sand fraction',
                * 'clay fraction',
                * 'organic carbon',
                * 'snow water equivalent',
                * 'surface temperature',
                * 'surface temperature quality flag original'
        min_depth : float, optional
            depth_from of variable has to be >= min_depth in order to be
            included.
        max_depth : float, optional
            depth_to of variable has to be <= max_depth in order to be
            included.
        Returns
        -------
        start_date: datetime
        end_date: datetime
        """
        start_date = None
        end_date = None

        if min_depth is None:
            min_depth = np.min(self.depth_from)
        if max_depth is None:
            max_depth = np.max(self.depth_to)

        for var, d1, d2, filename in zip(self.variables, self.depth_from, self.depth_to, self.filenames):

            if var == variable and ((d1 >= min_depth) & (d2 <= max_depth)):
                sdate, edate = readers.get_min_max_timestamp(filename)
                if start_date is None or start_date > sdate:
                    start_date = sdate
                if end_date is None or end_date < edate:
                    end_date = edate

        return start_date, end_date


class ISMN_Interface(object):

    """
    class provides interface to ISMN data downloaded from the ISMN website

    upon initialization it collects metadata from all files in
    path_to_data and saves metadata information
    in numpy file in folder path_to_data/python_metadata/
    First initialization can take a minute or so if all ISMN
    data is present in path_to_data

    Parameters
    ----------
    path_to_data : string
        filepath to unzipped ISMN data containing the Network folders
    network : string or list, optional
        provide name of network to only load the given network

    Raises
    ------
    ISMNError
        if given network was not found in path_to_data

    Attributes
    ----------
    metadata : numpy.array
        metadata array for all stations contained in the path given during initialization
    grid : pygeogrids.grid.BasicGrid
        Grid object used for finding nearest insitu station for given lon lat

    Methods
    -------
    find_nearest_station(lon,lat)
        find nearest station for given coordinates
    """

    def __init__(self, path_to_data, network=None):
        """
        collects metadata from all files in path_to_data and saves metadata information
        in numpy file in folder path_to_data/python_metadata/
        First initialization can take a minute or so if all ISMN
        data is present in path_to_data
        """

        if not os.path.exists(os.path.join(path_to_data, 'python_metadata', 'metadata.npy')):
            self.metadata = metadata_collector.collect_from_folder(
                path_to_data)
            os.mkdir(os.path.join(path_to_data, 'python_metadata'))
            np.save(
                os.path.join(path_to_data, 'python_metadata', 'metadata.npy'), self.metadata)
            #np.savetxt(os.path.join(path_to_data,'python_metadata','metadata.npy'), self.metadata,delimiter=',')
        else:
            self.metadata = np.load(
                os.path.join(path_to_data, 'python_metadata', 'metadata.npy'))

        if network is not None:
            if type(network) is not list:
                network = [network]
            # initialize binary mask the size of metadata
            mask = np.zeros(self.metadata.shape[0], dtype=np.bool)
            for net in network:
                if net in self.metadata['network']:
                    mask = mask | (self.metadata['network'] == net)
                else:
                    raise ISMNError("Network {} not found".format(net))
            self.metadata = self.metadata[mask]

        # initialize grid object for all stations
        self.grid = grids.BasicGrid(self.metadata['longitude'],
                                    self.metadata['latitude'],
                                    setup_kdTree=False)

    def list_networks(self):
        """
        returns numpy.array of networks available through the interface

        Returns
        -------
        networks : numpy.array
            unique network names available
        """
        return np.unique(self.metadata['network'])

    def list_stations(self, network=None):
        """
        returns numpy.array of station names available through the interface

        Parameters
        ----------
        network : string, optional
            if network name is given only stations belonging to the network
            are returned

        Returns
        -------
        networks : numpy.array
            unique network names available
        """

        if network is None:
            return np.unique(self.metadata['station'])
        elif network in self.list_networks():
            return np.unique(self.metadata[self.metadata['network'] == network]['station'])

    def get_station(self, stationname, network=None):
        """
        get ISMN_station object by station name

        Parameters
        ----------
        stationname : string
            name of station
        network : string, optional
            network name, has to be used if stations belonging
            to different networks have the same name

        Returns
        -------
        ISMN_station : ISMN_station object

        Raises
        ------
        ISMNError
            if stationname was not found
        """

        if network is not None:
            all_index = np.where((self.metadata['station'] == stationname) &
                                 (self.metadata['network'] == network))[0]
        else:
            all_index = np.where(self.metadata['station'] == stationname)[0]

        if all_index.size == 0:
            raise ISMNError("stationname was not found")

        metadatasub = self.metadata[all_index]

        if np.unique(metadatasub['network']).size > 1:
            raise ISMNError("stationname occurs in multiple networks")

        return ISMN_station(self.metadata[all_index])

    def stations_that_measure(self, variable):
        """
        Goes through all stations and returns those that measure the specified
        variable

        Parameters
        ----------
        variable : string
            variable name
            one of
                * 'soil moisture',
                * 'soil temperature',
                * 'soil suction',
                * 'precipitation',
                * 'air temperature',
                * 'field capacity',
                * 'permanent wilting point',
                * 'plant available water',
                * 'potential plant available water',
                * 'saturation',
                * 'silt fraction',
                * 'snow depth',
                * 'sand fraction',
                * 'clay fraction',
                * 'organic carbon',
                * 'snow water equivalent',
                * 'surface temperature',
                * 'surface temperature quality flag original'

        Returns
        -------
        ISMN_station : ISMN_station object
        """

        for network in self.list_networks():

            for stationname in self.list_stations(network=network):

                station = self.get_station(stationname, network=network)

                if variable in station.variables:
                    yield station

    def get_dataset_ids(self, variable, min_depth=0, max_depth=0.1):
        """
        returnes list of dataset_id's that can be used to read a
        dataset directly through the read_ts function
        """
        if max_depth < min_depth:
            raise ValueError("max_depth can not be less than min_depth")

        ids = np.where((self.metadata['variable'] == variable) &
                       (self.metadata['depth_to'] <= max_depth) &
                       (self.metadata['depth_from'] >= min_depth))[0]

        return ids

    def read_ts(self, idx):
        """
        read a time series directly by the id

        Parameters
        ----------
        idx : int
            id into self.metadata, best one of those returned
            from get_dataset_ids()

        Returns
        -------
        timeseries : pandas.DataFrame
            of the read data
        """
        ts = readers.read_data(self.metadata['filename'][idx])
        return ts.data

    def find_nearest_station(self, lon, lat, return_distance=False):
        """
        finds the nearest station available in downloaded data

        Parameters
        ----------
        lon : float
            Longitude of point
        lat : float
            Latitude of point
        return_distance : boolean, optional
            if True also distance is returned

        Returns
        -------
        station : ISMN_station
            ISMN_station object
        distance : float, optional
            distance to station in meters, measured in cartesian coordinates and not on
            a great circle. Should be OK for small distances
        """

        index, d = self.grid.find_nearest_gpi(lon, lat)

        all_index = np.where(
            self.metadata['station'] == self.metadata['station'][index])

        if return_distance:
            return ISMN_station(self.metadata[all_index]), d
        else:
            return ISMN_station(self.metadata[all_index])

    def plot_station_locations(self, axes=None):
        """
        plots available stations on a world map in robinson projection
        only available if basemap is installed

        Parameters
        ----------
        axes: matplotlib.Axes, optional
            If given then plot will be on this axes.

        Returns
        -------
        fig: matplotlib.Figure
            created figure instance. If axes was given this will be None.
        axes: matplitlib.Axes
            used axes instance.

        Raises
        ------
        ISMNError
            if basemap is not installed
        """
        if basemap_installed:

            if axes is None:
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 0.9, 1])
            else:
                fig = None
                ax = axes
            colormap = plt.get_cmap('Set1')

            ismn_map = Basemap(projection='robin', lon_0=0)

            uniq_networks = self.list_networks()

            colorsteps = np.arange(0, 1, 1 / float(uniq_networks.size))
            rect = []

            for j, network in enumerate(uniq_networks):

                stations_idx = np.where(self.metadata['network'] == network)[0]
                unique_stations, us_idx = np.unique(
                    self.metadata['station'][stations_idx], return_index=True)

                netcolor = colormap(colorsteps[j])
                rect.append(Rectangle((0, 0), 1, 1, fc=netcolor))

                for i, station in enumerate(unique_stations):
                    lat, lon = self.metadata['latitude'][stations_idx[us_idx[i]]], self.metadata[
                        'longitude'][stations_idx[us_idx[i]]]
                    x, y = ismn_map(lon, lat)

                    im = ismn_map.scatter(
                        x, y, c=netcolor, s=10, marker='s', edgecolors='none', ax=ax)

            ismn_map.drawcoastlines(linewidth=0.25)
            ismn_map.drawcountries(linewidth=0.25)
            ismn_map.drawstates(linewidth=0.25)
            plt.legend(
                rect, uniq_networks.tolist(), loc='lower center', ncol=uniq_networks.size / 4)

            return fig, ax
        else:
            raise ISMNError('Basemap is not installed.')

    def get_min_max_obs_timestamps(self, variable="soil moisture", min_depth=None, max_depth=None):
        """
        get minimum and maximum timestamps per station

        Parameters
        ----------
        self: type
            description
        variable: string, optional
            one of
                * 'soil moisture',
                * 'soil temperature',
                * 'soil suction',
                * 'precipitation',
                * 'air temperature',
                * 'field capacity',
                * 'permanent wilting point',
                * 'plant available water',
                * 'potential plant available water',
                * 'saturation',
                * 'silt fraction',
                * 'snow depth',
                * 'sand fraction',
                * 'clay fraction',
                * 'organic carbon',
                * 'snow water equivalent',
                * 'surface temperature',
                * 'surface temperature quality flag original'
        min_depth : float, optional
            depth_from of variable has to be >= min_depth in order to be
            included.
        max_depth : float, optional
            depth_to of variable has to be <= max_depth in order to be
            included.

        Returns
        -------
        data : pd.DataFrame
            dataframe with multiindex Network Station and
            columns start_date and end_date
        """
        networks = []
        stations = []
        start_dates = []
        end_dates = []
        for network in self.list_networks():

            for stationname in self.list_stations(network=network):
                # append station and network names to lists for
                # construction of pandas mulitindex
                networks.append(network)
                stations.append(stationname)

                station = self.get_station(stationname, network=network)
                startd, endd = station.get_min_max_obs_timestamp(variable=variable, min_depth=min_depth,
                                                                 max_depth=max_depth)
                start_dates.append(startd)
                end_dates.append(endd)

        data = pd.DataFrame({"start date": start_dates,
                             "end date": end_dates}, index=[np.array(networks), np.array(stations)])
        return data
