# Copyright (c) 2015,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology,
#      Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

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
tests for the ismn interface
Created on Thu Feb 26 12:36:30 2015

@author: christoph.paulik@geo.tuwien.ac.at
'''

import matplotlib
matplotlib.use('Agg')
from pytesmo.io.ismn import interface
import os
import datetime
import pytest
import numpy.testing as nptest


def test_min_max_obstime_getting():
    """
    test of the getting minimum and maxiumum observation time
    of a station
    """

    path_header_values = os.path.join(os.path.dirname(__file__),
                                      '..', 'test-data', 'ismn', 'format_header_values', 'SMOSMANIA')
    hv_interface = interface.ISMN_Interface(path_header_values)

    station = hv_interface.get_station('Narbonne')
    startd, endd = station.get_min_max_obs_timestamp()
    assert startd == datetime.datetime(2007, 1, 1, 1)
    assert endd == datetime.datetime(2007, 1, 31, 23)

    path_ceop_sep = os.path.join(os.path.dirname(__file__),
                                 '..', 'test-data', 'ismn', 'format_ceop_sep', 'SMOSMANIA')
    ceop_sep_interface = interface.ISMN_Interface(path_ceop_sep)

    station = ceop_sep_interface.get_station('Narbonne')
    startd, endd = station.get_min_max_obs_timestamp()
    assert startd == datetime.datetime(2007, 1, 1, 1)
    assert endd == datetime.datetime(2007, 1, 31, 23)


def test_min_max_obstime_networks():
    """
    test of the getting minimum and maxiumum observation time
    of several networks
    """

    path_header_values = os.path.join(os.path.dirname(__file__),
                                      '..', 'test-data', 'ismn', 'multinetwork', 'header_values')
    hv_interface = interface.ISMN_Interface(path_header_values)
    data = hv_interface.get_min_max_obs_timestamps(min_depth=0, max_depth=0.1)
    assert data.loc['MAQU']['end date'][
        0] == datetime.datetime(2010, 7, 31, 23)
    assert data.loc['MAQU']['end date'][
        1] == datetime.datetime(2010, 7, 31, 23)
    assert data.loc['MAQU']['start date'][
        1] == datetime.datetime(2008, 7, 1, 0)
    assert data.loc['SCAN']['start date'][
        1] == datetime.datetime(2007, 1, 1, 0)
    assert data.loc['SOILSCAPE']['start date'][
        1] == datetime.datetime(2012, 12, 14, 19)


def test_interface_network_init():
    """
    test limitation of interface to certain networks
    """

    path_header_values = os.path.join(os.path.dirname(__file__),
                                      '..', 'test-data', 'ismn', 'multinetwork', 'header_values')
    hv_interface = interface.ISMN_Interface(
        path_header_values, network=['SCAN'])
    assert hv_interface.list_networks().size == 1
    assert hv_interface.list_networks()[0] == 'SCAN'
    hv_interface = interface.ISMN_Interface(
        path_header_values, network=['SCAN', 'MAQU'])
    assert hv_interface.list_networks().size == 2


def test_find_nearest_station():
    """
    Test nearest neighbor search
    """
    path_header_values = os.path.join(os.path.dirname(__file__),
                                      '..', 'test-data', 'ismn', 'multinetwork', 'header_values')
    hv_interface = interface.ISMN_Interface(
        path_header_values, network=['SCAN'])
    station, distance = hv_interface.find_nearest_station(-90, 35, True)
    assert station.station == "AAMU-jtg"
    assert station.network == "SCAN"
    nptest.assert_almost_equal(distance, 316228.53147802927)


@pytest.mark.mpl_image_compare(tolerance=7)
def test_interface_plotting():
    """
    test plotting of networks
    """
    path_header_values = os.path.join(os.path.dirname(__file__),
                                      '..', 'test-data', 'ismn', 'multinetwork', 'header_values')
    hv_interface = interface.ISMN_Interface(
        path_header_values, network=['SCAN'])
    fig, axes = hv_interface.plot_station_locations()
    return fig
