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
Created on Jul 30, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

import numpy as np

try:
    import pykdtree.kdtree as pykd
    pykdtree_installed = True
except ImportError:
    pykdtree_installed = False

try:
    import scipy.spatial as sc_spat
    scipy_installed = True
except ImportError:
    scipy_installed = False


class findGeoNN(object):

    """
    class that takes lat,lon coordinates, transformes them to cartesian (X,Y,Z)
    coordinates and provides a interface to scipy.spatial.kdTree
    as well as pykdtree if installed

    Parameters
    ----------
    lon : numpy.array or list
        longitudes of the points in the grid
    lat : numpy.array or list
        latitudes of the points in the grid
    R : float, optional
        Radius of the earth to use for calculating 3D coordinates
    grid : boolean, optional
        if True then lon and lat are assumed to be the coordinates of a grid
        and will be used in numpy.meshgrid to get coordinates for all
        grid points
    kd_tree_name : string, optional
        name of kdTree implementation to use, either
        'pykdtree' to use pykdtree or
        'scipy' to use scipy.spatial.kdTree
        Fallback is always scipy if any other string is given
        or if pykdtree is not installed. standard is pykdtree since it is faster

    Attributes
    ----------
    R : float
        earth radius to use in computation of x,y,z coordinates
    coords : numpy.array
        3D array of cartesian x,y,z coordinates
    kd_tree_name: string
        name of kdTree implementation to use, either
        'pykdtree' to use pykdtree or
        'scipy' to use scipy.spatial.kdTree
        Fallback is always scipy if any other string is given
        or if pykdtree is not installed
    kdtree: object
        kdTree object that is built only once and saved in this attribute

    Methods
    -------
    find_nearest_index(lon,lat)
        finds the nearest neighbor of the given lon,lat coordinates in the lon,lat
        arrays given during initialization and returns the index of the nearest neighbour
        in those arrays.

    """

    def __init__(self, lon, lat, R=6370997.0, grid=False, kd_tree_name='pykdtree'):
        """
        init method, prepares lon and lat arrays for _transform_lonlats if
        necessary

        """
        if grid:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            lat_init = lat_grid.flatten()
            lon_init = lon_grid.flatten()
            self.lat_size = len(lat)
            self.lon_size = len(lon)
        else:
            if lat.shape != lon.shape:
                raise Exception(
                    "lat and lon np.arrays have to have equal shapes")
            lat_init = lat
            lon_init = lon
        # Earth radius
        self.R = R
        self.kd_tree_name = kd_tree_name
        self.coords = self._transform_lonlats(lon_init, lat_init)
        self.kdtree = None
        self.grid = grid

    def _transform_lonlats(self, lon, lat):
        """
        calculates cartesian 3D coordinates from given lon,lat

        Parameters
        ----------
        lon : numpy.array, list or float
            longitudes of the points in the grid
        lat : numpy.array, list or float
            latitudes of the points in the grid

        Returns
        -------
        coords : np.array
            3D cartesian coordinates
        """
        lon = np.array(lon)
        lat = np.array(lat)
        coords = np.zeros((lon.size, 3))
        # calculated in float64, otherwise numerical inconsistencies happened
        # on different systems
        lons_rad = np.radians(lon, dtype=np.float64)
        lats_rad = np.radians(lat, dtype=np.float64)
        coords[:, 0] = self.R * np.cos(lats_rad) * np.cos(lons_rad)
        coords[:, 1] = self.R * np.cos(lats_rad) * np.sin(lons_rad)
        coords[:, 2] = self.R * np.sin(lats_rad)

        return coords

    def _build_kdtree(self):
        """
        Build the kdtree and saves it in the self.kdtree attribute
        """
        if self.kd_tree_name == 'pykdtree' and pykdtree_installed:
            self.kdtree = pykd.KDTree(self.coords)
        elif scipy_installed:
            self.kdtree = sc_spat.KDTree(self.coords)
        else:
            raise Exception("No supported kdtree implementation installed.\
                             Please install pykdtree or scipy.")

    def find_nearest_index(self, lon, lat, max_dist=np.Inf):
        """
        finds nearest index, builds kdTree if it does not yet exist

        Parameters
        ----------
        lon : float, list or numpy.array
            longitude of point
        lat : float, list or numpy.array
            latitude of point
        max_dist : float, optional
            maximum distance to consider for search

        Returns
        -------
        d : float, numpy.array
            distances of query coordinates to the nearest grid point,
            distance is given in cartesian coordinates and is not the
            great circle distance at the moment. This should be OK for
            most applications that look for the nearest neighbor which
            should not be hundreds of kilometers away.
        ind : int, numpy.array
            indices of nearest neighbor
        index_lon : numpy.array, optional
            if self.grid is True then return index into lon array of grid definition
        index_lat : numpy.array, optional
            if self.grid is True then return index into lat array of grid definition
        """
        if self.kdtree is None:
            self._build_kdtree()

        query_coords = self._transform_lonlats(lon, lat)

        d, ind = self.kdtree.query(query_coords, distance_upper_bound=max_dist)

        if not self.grid:
            return d, ind
        else:
            # calculate index position in grid definition arrays assuming row-major
            # flattening of arrays after numpy.meshgrid
            index_lat = ind / self.lon_size
            index_lon = ind % self.lon_size
            return d, index_lon.astype(np.int32), index_lat.astype(np.int32)
