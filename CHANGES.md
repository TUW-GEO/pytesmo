# v0.3.6, 2015-12-10

* make sure that climatologies are always 366 elements
* add new options to climatology calculation for filling NaN values
* add option to climatology calculation for wraparound before the smoothing

# v0.3.5, 2015-11-04

* fix bug in anomaly calculation that occurred when the climatology series had
a name already
* add option in anomaly calculation to respect leap years during matching
* improve testing of scaling functions
* add linear CDF scaling based on stored percentiles
* add utility function for MATLAB like percentile calculation
* add utility function for making sure elements in an array are unique by
  using iterative interpolation

# v0.3.4, 2015-10-23

* fix #63 by moving data preparation before period checks
* fix bug in exponential and boxcar filter. Problem was that nan values were not
  ignored correctly

# v0.3.3, 2015-08-26

* add option to temporal resampling to exclude window boundaries
* fix #48 by reintroducting netcdf imports
* fix #60 by importing correctly from pygeogrids
* fix #56 by allowing read_bulk keyword for ASCAT_SSM
* fix #58 by using cKDTree keyword if available
* lookup table indexing fixed, see #59

# v0.3.2, 2015-07-09
* hotfix for temporal resampling problem when time series where of unequal lenghts

# v0.3.1, 2015-07-09
* added validation framework and example on how to use it
* fix bug (issue #51) in temporal matching
* added test data as git submodule

# v0.3.0, 2015-05-26
* added calculation of pearson R confidence intervals based on fisher z transform
* ISMN reader can now get the data coverage for stations and networks
* ISMN interface can now be restricted to a list of networks
* added python3 support
* moved grid functionality to pygeogrids package, pytesmo grids are deprecated
  and will be removed in future releases
* include triple collocation example and improve documentation see issue #24

# v0.2.5, 2014-12-15
* fixed ASCAT verion detection for latest H25 dataset WARP55R22
* added example for Soil Water Index calculation

# v0.2.4, 2014-12-09
* moved to pyscaffold structure
* added tests for modules
* added grid generation routines
* fix for issue #15
* updated classes to work with new base classes, does not change API
* added travis CI support
* changed theme of documentation, and enabled read the docs

# v0.2.3, 2014-10-03
* added grouping module

# v0.2.2, 2014-10-03
* fixed bug that lead to old grids without shape information not loading

# v0.2.1, 2014-8-14
* added functionality to save grid as 2 dimensional array in grid.netcdf if
  grid is regular and shape information is given

# v0.2.0, 2014-06-12
* added readers, tests and examples for H-SAF image products H07, H08 and H14
* added resample method that makes using pyresample a easier for the dictionary structure that
  pytesmo uses for image data
* added colormap reader for custom colormaps

# v0.1.3, 2014-05-26
* fixed bug in grid.nearest_neighbour that caused different results on
  different systems. Radians are now always calculated at 64bit accuracy
* ISMN routines now read the new ISMN download format
* df_metrics.bias now also returns a namedtuple

# v0.1.2, 2014-04-16
* Reader for different versions of netCDF H25 HSAF product
* added functionality to save grid definitions to netCDF files
* Fixed Bug that masked all data if snow probabilities did not exist
* Added tests

# v0.1.1, 2013-11-18
* Added readers for netCDF H25 HSAF product
* Added readers for netCDF ERS soil moisture product
* Added general grid classes
* Performance improvements for anomaly and climatology calculation through usage of cython
* Introduced df_metrics module for convienent calculation of metrics for data saved in pandas.DataFrames
