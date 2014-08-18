# Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the <organization> nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Jul 31, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

import os
import pandas as pd
from datetime import datetime
import numpy as np


variable_lookup = {'sm': 'soil moisture',
                   'ts': 'soil temperature',
                   'su': 'soil suction',
                   'p': 'precipitation',
                   'ta': 'air temperature',
                   'fc': 'field capacity',
                   'wp': 'permanent wilting point',
                   'paw': 'plant available water',
                   'ppaw': 'potential plant available water',
                   'sat': 'saturation',
                   'si_h': 'silt fraction',
                   'sd': 'snow depth',
                   'sa_h': 'sand fraction',
                   'cl_h': 'clay fraction',
                   'oc_h': 'organic carbon',
                   'sweq': 'snow water equivalent',
                   'tsf': 'surface temperature',
                   'tsfq': 'surface temperature quality flag original'
                  }


class ReaderException(Exception):
    pass


class ISMNTSError(Exception):
    pass


class ISMNTimeSeries(object):
    """
    class that contains a time series of ISMN data read from one text file

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
    variable : list
        variable measured
    depth_from : list
        shallower depth of layer the variable was measured at
    depth_to : list
        deeper depth of layer the variable was measured at
    sensor : string
        sensor name
    data : pandas.DataFrame
        data of the time series
    """
    def __init__(self, data):

        for key in data:
            setattr(self, key, data[key])

    def __repr__(self):

        return '%s %s %.2f m - %.2f m %s measured with %s ' % (self.network, self.station,
                                                   self.depth_from[0], self.depth_to[0],
                                                   self.variable[0], self.sensor)

    def plot(self, *args, **kwargs):
        """
        wrapper for pandas.DataFrame.plot which adds title to plot
        and drops NaN values for plotting
        Returns
        -------
        ax : axes
            matplotlib axes of the plot

        Raises
        ------
        ISMNTSError
            if data attribute is not a pandas.DataFrame
        """
        if type(self.data) is pd.DataFrame:
            tempdata = self.data.dropna()
            tempdata = tempdata[tempdata.columns[0]]
            ax = tempdata.plot(*args, figsize=(15, 5), **kwargs)
            ax.set_title(self.__repr__())
            return ax
        else:
            raise ISMNTSError("data attribute is not a pandas.DataFrame")


def get_info_from_file(filename):
    """
    reads first line of file and splits filename
    this can be used to construct necessary metadata information
    for all ISMN formats

    Parameters
    ----------
    filename : string
        filename including path

    Returns
    -------
    header_elements : list
        first line of file split into list
    filename_elements : list
        filename without path split by _
    """
    with open(filename, 'U') as f:
        header = f.readline()
    header_elements = header.split()

    path, filen = os.path.split(filename)
    filename_elements = filen.split('_')

    return header_elements, filename_elements


def get_metadata_header_values(filename):
    """
    get metadata from ISMN textfiles in the format called
    Variables stored in separate files (CEOP formatted)

    Parameters
    ----------
    filename : string
        path and name of file

    Returns
    -------
    metadata : dict
        dictionary of metadata information
    """

    header_elements, filename_elements = get_info_from_file(filename)

    if len(filename_elements) > 9:
        sensor = '_'.join(filename_elements[6:len(filename_elements) - 2])
    else:
        sensor = filename_elements[6]

    if filename_elements[3] in variable_lookup:
        variable = [variable_lookup[filename_elements[3]]]
    else:
        variable = [filename_elements[3]]

    metadata = {'network': header_elements[1],
                'station': header_elements[2],
                'latitude': float(header_elements[3]),
                'longitude': float(header_elements[4]),
                'elevation': float(header_elements[5]),
                'depth_from': [float(header_elements[6])],
                'depth_to': [float(header_elements[7])],
                'variable': variable,
                'sensor': sensor}

    return metadata


def read_format_header_values(filename):
    """
    Reads ISMN textfiles in the format called
    Variables stored in separate files (Header + values)

    Parameters
    ----------
    filename : string
        path and name of file

    Returns
    -------
    time_series : ISMNTimeSeries
        ISMNTimeSeries object initialized with metadata and data from file
    """

    metadata = get_metadata_header_values(filename)

    data = pd.read_csv(filename, skiprows=1, delim_whitespace=True,
                       names=['date', 'time', metadata['variable'][0],
                              metadata['variable'][0] + '_flag',
                              metadata['variable'][0] + '_orig_flag'])

    date_index = data.apply(lambda x: datetime.strptime('%s%s' % (x['date'], x['time']), '%Y/%m/%d%H:%M'), axis=1)

    del data['date']
    del data['time']

    data.index = date_index
    data.index.names = ['date']

    metadata['data'] = data

    return ISMNTimeSeries(metadata)


def get_metadata_ceop_sep(filename):
    """
    get metadata from ISMN textfiles in the format called
    Variables stored in separate files (CEOP formatted)

    Parameters
    ----------
    filename : string
        path and name of file

    Returns
    -------
    metadata : dict
        dictionary of metadata information
    """

    header_elements, filename_elements = get_info_from_file(filename)

    if len(filename_elements) > 9:
        sensor = '_'.join(filename_elements[6:len(filename_elements) - 2])
    else:
        sensor = filename_elements[6]

    if filename_elements[3] in variable_lookup:
        variable = [variable_lookup[filename_elements[3]]]
    else:
        variable = [filename_elements[3]]

    metadata = {'network': filename_elements[1],
                'station': filename_elements[2],
                'variable': variable,
                'depth_from': [float(filename_elements[4])],
                'depth_to': [float(filename_elements[5])],
                'sensor': sensor,
                'latitude': float(header_elements[7]),
                'longitude': float(header_elements[8]),
                'elevation': float(header_elements[9])
                }

    return metadata


def read_format_ceop_sep(filename):
    """
    Reads ISMN textfiles in the format called
    Variables stored in separate files (CEOP formatted)

    Parameters
    ----------
    filename : string
        path and name of file

    Returns
    -------
    time_series : ISMNTimeSeries
        ISMNTimeSeries object initialized with metadata and data from file
    """

    metadata = get_metadata_ceop_sep(filename)

    data = pd.read_csv(filename, delim_whitespace=True, usecols=[0, 1, 12, 13, 14],
                       names=['date', 'time', metadata['variable'][0], metadata['variable'][0] + '_flag', metadata['variable'][0] + '_orig_flag'])

    date_index = data.apply(lambda x: datetime.strptime('%s%s' % (x['date'], x['time']), '%Y/%m/%d%H:%M'), axis=1)

    del data['date']
    del data['time']

    data.index = date_index
    data.index.names = ['date']

    metadata['data'] = data

    return ISMNTimeSeries(metadata)


def get_metadata_ceop(filename):
    """
    get metadata from ISMN textfiles in the format called
    CEOP Reference Data Format

    Parameters
    ----------
    filename : string
        path and name of file

    Returns
    -------
    metadata : dict
        dictionary of metadata information
    """

    header_elements, filename_elements = get_info_from_file(filename)

    metadata = {'network': filename_elements[1],
                'station': header_elements[6],
                'variable': ['ts', 'sm'],
                'sensor': 'n.s',
                'depth_from': ['multiple'],
                'depth_to': ['multiple'],
                'latitude': float(header_elements[7]),
                'longitude': float(header_elements[8]),
                'elevation': float(header_elements[9])
                }

    return metadata


def read_format_ceop(filename):
    """
    Reads ISMN textfiles in the format called
    CEOP Reference Data Format

    Parameters
    ----------
    filename : string
        path and name of file

    Returns
    -------
    time_series : ISMNTimeSeries
        ISMNTimeSeries object initialized with metadata and data from file
    """
    metadata = get_metadata_ceop(filename)
    data = pd.read_csv(filename, delim_whitespace=True, usecols=[0, 1, 11, 12, 13, 14, 15],
                       names=['date', 'time', 'depth_from',
                              metadata['variable'][0], metadata['variable'][0] + '_flag',
                              metadata['variable'][1], metadata['variable'][1] + '_flag'],
                       na_values=['-999.99'])

    date_index = data.apply(lambda x: datetime.strptime('%s%s' % (x['date'], x['time']), '%Y/%m/%d%H:%M'), axis=1)
    depth_index = data['depth_from']

    del data['date']
    del data['time']
    del data['depth_from']

    data.index = pd.MultiIndex.from_arrays([depth_index, depth_index, date_index])
    data.index.names = ['depth_from', 'depth_to', 'date']

    data = data.sortlevel(0)

    metadata['depth_from'] = np.unique(data.index.get_level_values(0).values).tolist()
    metadata['depth_to'] = np.unique(data.index.get_level_values(1).values).tolist()
    metadata['data'] = data

    return ISMNTimeSeries(metadata)


def get_format(filename):
    """
    get's the file format from the length of
    the header and filename information

    Parameters
    ----------
    filename : string

    Returns
    -------
    methodname : string
        name of method used to read the detected format

    Raises
    ------
    ReaderException
        if filename or header parts do not fit one of the formats
    """
    header_elements, filename_elements = get_info_from_file(filename)
    if len(filename_elements) == 5 and len(header_elements) == 16:
        return 'ceop'
    if len(header_elements) == 15 and len(filename_elements) >= 9:
        return 'ceop_sep'
    if len(header_elements) < 14 and len(filename_elements) >= 9:
        return 'header_values'
    raise ReaderException("This does not seem to be a valid ISMN filetype %s" % filename)


def read_data(filename):
    """
    reads ISMN data in any format

    Parameters
    ----------
    filename: string

    Returns
    -------
    timeseries: IMSNTimeSeries
    """
    dicton = globals()
    func = dicton['read_format_' + get_format(filename)]
    return func(filename)


def get_metadata(filename):
    """
    reads ISMN metadata from any format

    Parameters
    ----------
    filename: string

    Returns
    -------
    metadata: dict
    """
    dicton = globals()
    func = dicton['get_metadata_' + get_format(filename)]
    return func(filename)
