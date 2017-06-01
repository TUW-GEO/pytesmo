# Copyright (c) 2014,Vienna University of Technology, Department of Geodesy and Geoinformation
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
Created on Mar 25, 2014

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''


from pyresample import geometry, kd_tree
import numpy as np


def resample_to_grid_only_valid_return(input_data, src_lon, src_lat, target_lon, target_lat,
                                       methods='nn', weight_funcs=None,
                                       min_neighbours=1, search_rad=18000, neighbours=8,
                                       fill_values=None):
    """
    resamples data from dictionary of numpy arrays using pyresample
    to given grid.
    Searches for the neighbours and then resamples the data
    to the grid given in togrid if at least
    min_neighbours neighbours are found

    Parameters
    ----------
    input_data : dict of numpy.arrays
    src_lon : numpy.array
        longitudes of the input data
    src_lat : numpy.array
        src_latitudes of the input data
    target_lon : numpy.array
        longitudes of the output data
    target_src_lat : numpy.array
        src_latitudes of the output data
    methods : string or dict, optional
        method of spatial averaging. this is given to pyresample
        and can be
        'nn' : nearest neighbour
        'custom' : custom weight function has to be supplied in weight_funcs
        see pyresample documentation for more details
        can also be a dictionary with a method for each array in input data dict
    weight_funcs : function or dict of functions, optional
        if method is 'custom' a function like func(distance) has to be given
        can also be a dictionary with a function for each array in input data dict
    min_neighbours: int, optional
        if given then only points with at least this number of neighbours will be
        resampled
        Default : 1
    search_rad : float, optional
        search radius in meters of neighbour search
        Default : 18000
    neighbours : int, optional
        maximum number of neighbours to look for for each input grid point
        Default : 8
    fill_values : number or dict, optional
        if given the output array will be filled with this value if no valid
        resampled value could be computed, if not a masked array will be returned
        can also be a dict with a fill value for each variable
    Returns
    -------
    data : dict of numpy.arrays
        resampled data on part of the target grid over which data was found
    mask: numpy.ndarray
        boolean mask into target grid that specifies where data was resampled

    Raises
    ------
    ValueError :
        if empty dataset is resampled
    """
    output_data = {}

    if target_lon.ndim == 2:
        target_lat = target_lat.ravel()
        target_lon = target_lon.ravel()

    input_swath = geometry.SwathDefinition(src_lon, src_lat)
    output_swath = geometry.SwathDefinition(target_lon, target_lat)

    (valid_input_index,
     valid_output_index,
     index_array,
     distance_array) = kd_tree.get_neighbour_info(input_swath,
                                                  output_swath,
                                                  search_rad,
                                                  neighbours=neighbours)

    # throw away points with less than min_neighbours neighbours
    # find points with valid neighbours
    # get number of found neighbours for each grid point/row
    if neighbours > 1:
        nr_neighbours = np.isfinite(distance_array).sum(1)
        neigh_condition = nr_neighbours >= min_neighbours
        mask = np.invert(neigh_condition)
        enough_neighbours = np.nonzero(neigh_condition)[0]
    if neighbours == 1:
        nr_neighbours = np.isfinite(distance_array)
        neigh_condition = nr_neighbours >= min_neighbours
        mask = np.invert(neigh_condition)
        enough_neighbours = np.nonzero(neigh_condition)[0]
        distance_array = np.reshape(
            distance_array, (distance_array.shape[0], 1))
        index_array = np.reshape(index_array, (index_array.shape[0], 1))

    if enough_neighbours.size == 0:
        raise ValueError(
            "No points with at least %d neighbours found" % min_neighbours)

    # remove neighbourhood info of input grid points that have no neighbours to not have to
    # resample to whole output grid for small input grid file
    distance_array = distance_array[enough_neighbours, :]
    index_array = index_array[enough_neighbours, :]
    valid_output_index = valid_output_index[enough_neighbours]

    for param in input_data:

        data = input_data[param]

        if type(methods) == dict:
            method = methods[param]
        else:
            method = methods

        if method is not 'nn':
            if type(weight_funcs) == dict:
                weight_func = weight_funcs[param]
            else:
                weight_func = weight_funcs
        else:
            weight_func = None

        neigh_slice = slice(None, None, None)
        # check if method is nn, if so only use first row of index_array and
        # distance_array
        if method == 'nn':
            neigh_slice = (slice(None, None, None), 0)

        if type(fill_values) == dict:
            fill_value = fill_values[param]
        else:
            fill_value = fill_values

        output_array = kd_tree.get_sample_from_neighbour_info(
            method,
            enough_neighbours.shape,
            data,
            valid_input_index,
            valid_output_index,
            index_array[neigh_slice],
            distance_array[neigh_slice],
            weight_funcs=weight_func,
            fill_value=fill_value)

        output_data[param] = output_array

    return output_data, mask


def resample_to_grid(input_data, src_lon, src_lat, target_lon, target_lat,
                     methods='nn', weight_funcs=None,
                     min_neighbours=1, search_rad=18000, neighbours=8,
                     fill_values=None):
    """
    resamples data from dictionary of numpy arrays using pyresample
    to given grid.
    Searches for the neighbours and then resamples the data
    to the grid given in togrid if at least
    min_neighbours neighbours are found

    Parameters
    ----------
    input_data : dict of numpy.arrays
    src_lon : numpy.array
        longitudes of the input data
    src_lat : numpy.array
        src_latitudes of the input data
    target_lon : numpy.array
        longitudes of the output data
    target_src_lat : numpy.array
        src_latitudes of the output data
    methods : string or dict, optional
        method of spatial averaging. this is given to pyresample
        and can be
        'nn' : nearest neighbour
        'custom' : custom weight function has to be supplied in weight_funcs
        see pyresample documentation for more details
        can also be a dictionary with a method for each array in input data dict
    weight_funcs : function or dict of functions, optional
        if method is 'custom' a function like func(distance) has to be given
        can also be a dictionary with a function for each array in input data dict
    min_neighbours: int, optional
        if given then only points with at least this number of neighbours will be
        resampled
        Default : 1
    search_rad : float, optional
        search radius in meters of neighbour search
        Default : 18000
    neighbours : int, optional
        maximum number of neighbours to look for for each input grid point
        Default : 8
    fill_values : number or dict, optional
        if given the output array will be filled with this value if no valid
        resampled value could be computed, if not a masked array will be returned
        can also be a dict with a fill value for each variable
    Returns
    -------
    data : dict of numpy.arrays
        resampled data on given grid
    Raises
    ------
    ValueError :
        if empty dataset is resampled
    """

    output_data = {}
    output_shape = target_lat.shape
    if target_lon.ndim == 2:
        target_lat = target_lat.ravel()
        target_lon = target_lon.ravel()

    resampled_data, mask = resample_to_grid_only_valid_return(input_data,
                                                              src_lon, src_lat,
                                                              target_lon, target_lat,
                                                              methods=methods,
                                                              weight_funcs=weight_funcs,
                                                              min_neighbours=min_neighbours,
                                                              search_rad=search_rad,
                                                              neighbours=neighbours)
    for param in input_data:
        data = resampled_data[param]
        orig_data = input_data[param]

        if type(fill_values) == dict:
            fill_value = fill_values[param]
        else:
            fill_value = fill_values

        # construct arrays in output grid form
        if fill_value is not None:
            output_array = np.zeros(
                target_lat.shape, dtype=orig_data.dtype) + fill_value
        else:
            output_array = np.zeros(target_lat.shape, dtype=orig_data.dtype)
            output_array = np.ma.array(output_array, mask=mask)
        output_array[~mask] = data

        output_data[param] = output_array.reshape(output_shape)

    return output_data


def hamming_window(radius, distances):
    """
    Hamming window filter.

    Parameters
    ----------
    radius : float32
        Radius of the window.
    distances : numpy.ndarray
        Array with distances.

    Returns
    -------
    weights : numpy.ndarray
        Distance weights.
    """
    alpha = 0.54
    weights = alpha + (1 - alpha) * np.cos(np.pi / radius * distances)

    return weights
