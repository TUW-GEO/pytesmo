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
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Aug 26, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''

import pytesmo.grid.nearest_neighbor as NN

import numpy as np
from itertools import izip

class GridDefinitionError(Exception):
    pass

class GridIterationError(Exception):
    pass

class BasicGrid(object):
    """
    Grid that just has lat,lon coordinates and can find the 
    nearest neighbour. It can also yield the gpi, lat, lon
    information in order.
    
    Parameters
    ----------
    lon : numpy.array
        longitudes of the points in the grid
    lat : numpy.array
        latitudes of the points in the grid
    gpis : numpy.array, optional
        if the gpi numbers are in a different order than the 
        lon and lat arrays an array containing the gpi numbers
        can be given
    subset : numpy.array, optional
        if the active part of the array is only a subset of
        all the points then the subset array which is a index
        into lon and lat can be given here
    setup_kdTree : boolean, optional
        if set (default) then the kdTree for nearest neighbour
        search will be built on initialization
    shape : tuple, optional
        if given the grid can be reshaped into the given shape
        this indicates that it is a regular grid and fills the 
        attributes self.londim and self.latdim which 
        define the grid only be the meridian coordinates(self.londim) and
        the coordinates of the circles of latitude(self.latdim).
        The shape has to be given as (latdim, londim) 
    
    Attributes
    ----------
    arrlon : numpy.array
        array of all longitudes of the grid
    arrlat : numpy.array
        array of all latitudes of the grid
    n_gpi : int
        number of gpis in the grid
    gpidirect : boolean
        if true the gpi number is equal to the index
        of arrlon and arrlat
    gpis : numpy.array
        gpi number for elements in arrlon and arrlat
        gpi[i] is located at arrlon[i],arrlat[i]
    subset : numpy.array
        if given then this contains the indices of a subset of
        the grid. This can be used if only a part of a grid is 
        interesting for a application. e.g. land points, or only
        a specific country
    allpoints : boolean
        if False only a subset of the grid is active
    activearrlon : numpy.array
        array of longitudes that are active, is defined by
        arrlon[subset] if a subset is given otherwise equal to 
        arrlon
    activearrlat : numpy.array
        array of latitudes that are active, is defined by
        arrlat[subset] if a subset is given otherwise equal to 
        arrlat
    activegpis : numpy.array
        array of gpis that are active, is defined by
        gpis[subset] if a subset is given otherwise equal to 
        gpis        
    issplit : boolean
        if True then the array was split in n parts with 
        the self.split function
    kdTree : object
        grid.nearest_neighbor.findGeoNN object for
        nearest neighbor search
    shape : tuple, optional
        if given during initialization then this is
        the shape the grid can be reshaped to
        this only makes sense for regular lat,lon grids
    latdim : numpy.array, optional
        if shape is given this attribute has contains
        all latitudes that make up the regular lat,lon grid
    londim : numpy.array, optional
        if shape is given this attribute has contains
        all longitudes that make up the regular lat,lon grid    
    """
    
    def __init__(self, lon, lat, gpis=None, subset=None, setup_kdTree=True,
                 shape=None):
        """
        init method, prepares lon and lat arrays for _transform_lonlats if 
        necessary
        
        """
        if lat.shape != lon.shape: 
            raise GridDefinitionError("lat and lon np.arrays have to have equal shapes")

        self.n_gpi = len(lon)

        self.arrlon = lon
        self.arrlat = lat
        
        
        if shape is not None:
            if len(self.arrlat) % shape[0] != 0:
                raise GridDefinitionError("Given shape does not have the correct first dimension." 
                                          " Length of lat array is not divisible by shape[0] without rest")
        
            if len(self.arrlon) % shape[1] != 0:
                raise GridDefinitionError("Given shape does not have the correct second dimension." 
                                          " Length of lon array is not divisible by shape[1] without rest")
        
            
            self.shape = shape
            self.latdim = np.reshape(self.arrlat, self.shape)[:, 0]
            self.londim = np.reshape(self.arrlon, self.shape)[0, :]
            
        
        if gpis is None:
            self.gpis = np.arange(self.n_gpi)
            self.gpidirect = True
        else:
            if lat.shape != gpis.shape: 
                raise GridDefinitionError("lat, lon gpi np.arrays have to have equal shapes")
            self.gpis = gpis  
            self.gpidirect = False
        
        self.subset = subset
        
        if subset is not None:
            self.activearrlon = self.arrlon[subset]
            self.activearrlat = self.arrlat[subset]
            self.activegpis = self.gpis[subset]   
            self.allpoints = False
        else:
            self.activearrlon = self.arrlon
            self.activearrlat = self.arrlat  
            self.activegpis = self.gpis
            self.allpoints = True
        
        self.issplit = False
        
        self.kdTree = None
        if setup_kdTree:  
            self.kdTree = NN.findGeoNN(self.activearrlon, self.activearrlat)
    
    def split(self, n):  
        """function splits the grid into n parts
        this changes not function but grid_points() which takes
        the argument n and will only iterate through this part of 
        the grid
        
        Parameters
        ----------
        n : int
            number of parts the grid should be split into
        """
        self.issplit = True
        self.subarrlats = np.array_split(self.activearrlat, n)
        self.subarrlons = np.array_split(self.activearrlon, n)
        self.subgpis = np.array_split(self.activegpis, n)
    
    def unite(self):
        """
        unites a split array, so that it can be iterated over as a whole
        again
        """
        self.issplit = False
    
    def grid_points(self, *args):
        """
        Yields all grid points in order
        
        Parameters
        ----------
        n : int, optional
            if the grid is split in n parts using the split function
            then this iterator will only iterate of the nth part of the
            grid
        
        Returns
        -------
        gpi : long
            grid point index
        lon : float
            longitude of gpi
        lat : float
            longitude of gpi
        """
         
        if not self.issplit and len(args) == 0:
            return self._normal_grid_points()
        elif self.issplit and len(args) == 1:
            return self._split_grid_points(args[0])
        
        raise GridIterationError("this function only takes an argument if the grid is split, "
                                 "and takes no argument if the grid is not split")
    
    def get_grid_points(self, *args):
        """
        Returns all active grid points
        
        Parameters
        ----------
        n : int, optional
            if the grid is split in n parts using the split function
            then this function will only return the nth part of the
            grid
            
        Returns
        -------
        gpis : numpy.array
        arrlon : numpy.array
        arrlat :numpy.array
        """
        
        if not self.issplit and len(args) == 0:
            return (self.activegpis,
                    self.activearrlon,
                    self.activearrlat)
            
        elif self.issplit and len(args) == 1:
            n = args[0]
            return (self.subgpis[n],
                    self.subarrlons[n],
                    self.subarrlats[n])
        
        
         
    def _normal_grid_points(self):
        """
        Yields all grid points in order
        
        Returns
        -------
        gpi : long
            grid point index
        lon : float
            longitude of gpi
        lat : float
            longitude of gpi
        """
        
        for i, (lon, lat) in enumerate(izip(self.activearrlon, self.activearrlat)):
            yield self.activegpis[i], lon, lat
            
    def _split_grid_points(self, n):
        """
        Yields all grid points or split grid in order
        
        Parameters
        ----------
        n : int
            number of subgrid to yield
        
        Returns
        -------
        gpi : long
            grid point index
        lon : float
            longitude of gpi
        lat : float
            longitude of gpi
        """
        
        for i, (lon, lat) in enumerate(izip(self.subarrlons[n], self.subarrlats[n])):
            yield self.subgpis[n][i], lon, lat        
            
    def find_nearest_gpi(self, lon, lat, max_dist=np.Inf):
        """
        finds nearest gpi, builds kdTree if it does not yet exist
        
        Parameters
        ----------
        lon : float
            longitude of point
        lat : float
            latitude of point  
            
        Returns
        -------
        gpi : long
            grid point index
        distance : float
            distance of gpi to given lon, lat
            At the moment not on a great circle but in spherical cartesian coordinates
        """  
        
        if self.kdTree is None:
            self.kdTree = NN.findGeoNN(self.activearrlon, self.activearrlat)
        
        d, ind = self.kdTree.find_nearest_index(lon, lat, max_dist=max_dist)    
        
        if self.gpidirect and self.allpoints:
            return ind[0], d
        
        return self.activegpis[ind[0]], d    
    
    def gpi2lonlat(self, gpi):
        """
        Longitude and Latitude for given GPI.

        Parameters
        ----------
        gpi : int32
            Grid Point Index.

        Returns
        -------
        lon : float
            Longitude (deg) of GPI.
        lat : float
        """
        if self.gpidirect:
            return self.arrlon[gpi], self.arrlat[gpi]
        else:
            index = np.where(self.activegpis == gpi)[0][0] 
            return self.activearrlon[index], self.activearrlat[index]

    def calc_lut(self, other, max_dist=np.Inf, into_subset=False):
        """
        takes other BasicGrid or CellGrid objects and computes 
        a lookup table between them.
        the lut will have the size of self.n_gpis and will
        for every grid point have the nearest index into other.arrlon etc.
        
        Parameters
        ----------
        other : grid object
            to which to calculate the lut to
        max_dist : float, optional
            maximum allowed distance in meters
        into_subset : boolean, optional
            if set the returned lut will have the index into the subset
            if the other grid is a subset of a grid.
            Example:
            if e.g. ind_l is used for the warp_grid some datasets will
            be given as arrays with len(ind_l) elements. These
            datasets can not be indexed with gpi numbers but have to 
            be indexed with indices into the subset
        """
    
        if self.kdTree.kdtree is None:
            self.kdTree._build_kdtree()
        
        if other.kdTree.kdtree is None:
            other.kdTree._build_kdtree()    
    
        if self.kdTree.kdtree is not None and other.kdTree.kdtree is not None:
            dist, index = other.kdTree.find_nearest_index(self.activearrlon, self.activearrlat , max_dist=max_dist)
            
            valid_index = np.where(dist != np.inf)[0]
            dist = dist[valid_index]
            index = index[valid_index]
            if not other.gpidirect or not other.allpoints:
                if not into_subset:
                    index = other.activegpis[index]
            
            active_lut = np.empty_like(self.activearrlat, dtype=np.int64)
            active_lut.fill(-1)
            active_lut[valid_index] = index
            
            if not self.allpoints:
                gpi_lut = np.empty_like(self.gpis)
                gpi_lut.fill(-1)
                gpi_lut[self.subset] = active_lut
            else:
                gpi_lut = active_lut    
            
            return gpi_lut
          
        
class CellGrid(BasicGrid):        
    """
    Grid that has lat,lon coordinates as well as cell informatin. 
    It can find nearest neighbour. It can also yield the gpi, lat, lon, cell
    information in cell order. This is important if the data on the grid
    is saved in cell files on disk as we can go through all grid points
    with optimized IO performance
    
    Parameters
    ----------
    lon : numpy.array
        longitudes of the points in the grid
    lat : numpy.array
        latitudes of the points in the grid
    cells : numpy.array 
        of same shape as lon and lat, containing the cell number
        of each gpi
    gpis : numpy.array, optional
        if the gpi numbers are in a different order than the 
        lon and lat arrays an array containing the gpi numbers
        can be given    
    subset : numpy.array, optional
        if the active part of the array is only a subset of
        all the points then the subset array which is a index
        into lon, lat and cells can be given here   
        
    Attributes     
    ----------
    arrcell : numpy.array
        array of cell number with same shape as arrlon,arrlat
    activearrcell : numpy.array
        array of longitudes that are active, 
        is defined by arrlon[subset] if a subset is given otherwise equal to arrlon
    """
    
    def __init__(self, lon, lat, cells, gpis=None, subset=None):
        super(CellGrid, self).__init__(lon, lat, gpis=gpis, subset=subset, setup_kdTree=False)
        
        if self.arrlon.shape != cells.shape:
            raise GridDefinitionError("lat, lon and cells np.arrays have to have equal shapes")
        self.arrcell = cells
        
        if subset is not None:
            self.activearrcell = self.arrcell[subset]
        else:
            self.activearrcell = self.arrcell    
        
        self.kdTree = NN.findGeoNN(self.activearrlon, self.activearrlat)
        
    
    def gpi2cell(self, gpi):
        """
        Cell for given GPI.

        Parameters
        ----------
        gpi : int32
            Grid Point Index.

        Returns
        -------
        cell : int
            Cell number of GPI.
        """
        if self.gpidirect:
            return self.arrcell[gpi]
        else:
            index = np.where(self.activegpis == gpi)[0][0]
            return self.activearrcell[index]
    
    def get_cells(self):
        """
        function to get all cell numbers of the grid
        
        Returns
        -------
        cells : numpy.array
            unique cell numbers
        """
        return np.unique(self.activearrcell)
    
    def get_grid_points(self, *args):
        """
        Returns all active grid points
        
        Parameters
        ----------
        n : int, optional
            if the grid is split in n parts using the split function
            then this function will only return the nth part of the
            grid
            
        Returns
        -------
        gpis : numpy.array
        arrlon : numpy.array
        arrlat :numpy.array
        cells : numpy.array
        """
        
        if not self.issplit and len(args) == 0:
            return (self.activegpis,
                    self.activearrlon,
                    self.activearrlat,
                    self.activearrcell)
            
        elif self.issplit and len(args) == 1:
            n = args[0]
            return (self.subgpis[n],
                    self.subarrlons[n],
                    self.subarrlats[n],
                    self.subcells[n])
    
    
    
    def grid_points_for_cell(self, cell):
        """
        get all grid points for a given cell number
        
        Parameters
        ----------
        cell : int
            cell number
        
        Returns
        -------
        gpis : numpy.array
            gpis belonging to cell
        """
        cell_index = np.where(cell == self.activearrcell)
        
        return (self.activegpis[cell_index],
                self.activearrlon[cell_index],
                self.activearrlat[cell_index])
    
    def split(self, n):
        """function splits the grid into n parts
        this changes not function but grid_points() which takes
        the argument n and will only iterate through this part of 
        the grid
        
        Parameters
        ----------
        n : int
            number of parts the grid should be split into
        """
        self.issplit = True
        # sort by cell number to split correctly
        sorted_index = np.argsort(self.activearrcell)
        self.subarrlats = np.array_split(self.activearrlat[sorted_index], n)
        self.subarrlons = np.array_split(self.activearrlon[sorted_index], n)
        self.subgpis = np.array_split(self.activegpis[sorted_index], n)
        self.subcells = np.array_split(self.activearrcell[sorted_index], n)
    
    def _normal_grid_points(self):
        """
        Yields all grid points in cell order
        
        Returns
        -------
        gpi : long
            grid point index
        lon : float
            longitude of gpi
        lat : float
            longitude of gpi
        cell : int
            cell number
        """
        
        uniq_cells = np.unique(self.activearrcell)
        
        for cell in uniq_cells:
            cell_gpis = np.where(cell == self.activearrcell)[0]
            for gpi in cell_gpis:
                yield self.activegpis[gpi], self.activearrlon[gpi], self.activearrlat[gpi], cell
    

    def _split_grid_points(self, n):
        """
        Yields all grid points in cell order
        
        Parameters
        ----------
        n : int
                number of subgrid to yield
        
        Returns
        -------
        gpi : long
            grid point index
        lon : float
            longitude of gpi
        lat : float
            longitude of gpi
        cell : int
            cell number
        """
        
        uniq_cells = np.unique(self.subcells[n])
        
        for cell in uniq_cells:
            cell_gpis = np.where(cell == self.subcells[n])[0]
            for gpi in cell_gpis:
                yield self.subgpis[n][gpi], self.subarrlons[n][gpi], self.subarrlats[n][gpi], cell



        
    
    
    
