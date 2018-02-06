v0.6.9, 2018-02-06
==================

- Add extendent collocation metric
- Fix initial value for exponential filter
- Fix #123


v0.6.8, 2017-08-29
==================

-  Adapt validation framework examples to new ASCAT package version.
-  Adapt ERS reader to new ASCAT package version.
-  Make validation framework work with datasets that contain NaN
   columns.
-  Make validation framework work with pygeobase.object\_base.TS objects
   and subclasses.
-  Add scaler classes to the validation framework making it possible to
   use e.g. stored CDF parameters during validation.
-  ensure\_iterable does no longer take a single string as an iterable.
   We want to keep the string as one object.

v0.6.7, 2017-07-25
==================

-  Add respect leap years option for climatology calculation.

v0.6.6, 2017-07-14
==================

-  Compatible with Python 3.6

v0.6.5, 2017-07-10
==================

-  Add additonal functions for working with dekads.

v0.6.4, 2017-06-02
==================

-  Refactor resampling routine to be more modular and better usable
   outside of pytesmo.

v0.6.3, 2017-04-28
==================

-  temporal matching performance improvement of approx. 50%
-  Add functions for handling dekadal dates. See
   ``pytesmo.timedate.dekad``.

v0.6.2, 2017-01-13
==================

-  Fix metadata for new version of pypi.

v0.6.1, 2017-01-13
==================

-  Add return\_clim keyword to anomaly calculation. Useful for getting
   both anomaly and climatology in one pandas.DataFrame. Also used in
   time series anomaly plot.
-  Fix bug in julian2date which led to negative microseconds in some
   edge cases.

v0.6.0, 2016-07-29
==================

-  Moved the ASCAT readers to the ascat package. The functionality is
   the same, just replace ``import pytesmo.io.sat.ascat`` by
   ``import ascat`` and everything should work the same as before.
-  The H07 reader now returns also ssm mean as a value between 0 and
   100. Before it was between 0 and 1 and inconsistent with the other
   ssm values.
-  Fix small bug in julian date calculation and add tests for it.
-  Add hamming window to resample module

v0.5.2, 2016-04-26
==================

-  Fix bugs when the validation framework encountered empty datasets for
   various reasons.
-  Add dataset adapters for masking and anomaly calculation.
-  Improve performance of moving average calculation and ISMN readers.

v0.5.1, 2016-04-21
==================

-  Fix bug in jobs argument passing to Validation class.
-  Add support to use a pre initialized DataManager instance in the
   Validation class.
-  Add support for per dataset reading method names in the DataManager.
   This relaxes the assumption that every dataset has a ``read_ts``
   method.

v0.5.0, 2016-04-20
==================

-  Fix bug in temporal resampling if input was a pandas.Series
-  Major refactoring of validation framwork. Please see updated
   documentation and example for detailed changes. The most important
   breaking changes are:
-  'type' is no longer used in the dataset dictionary.
-  the temporal matcher does no longer need to be specified since a
   reasonable default was developed that should handle most cases
-  metrics calculators are now given as dictionaries of functions.
   Please see the docs for an explanation and an example.
-  cell\_based\_jobs keyword was removed in favor of a more general
   definition of jobs.

New features are the possibility to use unrelated masking datasets and
the possibility to temporally match any number of datasets and give them
in sets of k datasets to multiple metric calculators.

-  Changes in the scaling module, escpecially CDF matching. The new CDF
   scaling module is more modular and does not make any assumptions
   about how unique the percentiles for the CDF matching have to be. CDF
   matching now returns NaN values if non unique percentiles are in the
   data. There are new functions that rescale based on pre-calculated
   percentiles so these can be used if the user wants to make sure that
   the percentiles are unique before matching.

v0.4.0, 2016-03-24
==================

-  Fix bug in validation framework due to error prone string formatting
   in warnings.
-  Remove grid functionality. Use
   `pygeogrids <https://github.com/TUW-GEO/pygeogrids>`__ from now on.
-  Fix bug in moving average calculation when input had size 1.
-  Add recursive calculation of Pearson correlation coefficent.
-  Change H-SAF reading interface to use pygeobase consistently. This
   changes the interface slightly as the ``read_img`` method is now
   called just ``read``
-  H07 reader now returns more variables.
-  Resampling interface now respects dtype of input data.
-  Improvements in ISMN plotting interface make it possible to use the
   plot not only show it.

v0.3.6, 2015-12-10
==================

-  make sure that climatologies are always 366 elements
-  add new options to climatology calculation for filling NaN values
-  add option to climatology calculation for wraparound before the
   smoothing

v0.3.5, 2015-11-04
==================

-  fix bug in anomaly calculation that occurred when the climatology
   series had a name already
-  add option in anomaly calculation to respect leap years during
   matching
-  improve testing of scaling functions
-  add linear CDF scaling based on stored percentiles
-  add utility function for MATLAB like percentile calculation
-  add utility function for making sure elements in an array are unique
   by using iterative interpolation

v0.3.4, 2015-10-23
==================

-  fix #63 by moving data preparation before period checks
-  fix bug in exponential and boxcar filter. Problem was that nan values
   were not ignored correctly

v0.3.3, 2015-08-26
==================

-  add option to temporal resampling to exclude window boundaries
-  fix #48 by reintroducting netcdf imports
-  fix #60 by importing correctly from pygeogrids
-  fix #56 by allowing read\_bulk keyword for ASCAT\_SSM
-  fix #58 by using cKDTree keyword if available
-  lookup table indexing fixed, see #59

v0.3.2, 2015-07-09
==================

-  hotfix for temporal resampling problem when time series where of
   unequal lenghts

v0.3.1, 2015-07-09
==================

-  added validation framework and example on how to use it
-  fix bug (issue #51) in temporal matching
-  added test data as git submodule

v0.3.0, 2015-05-26
==================

-  added calculation of pearson R confidence intervals based on fisher z
   transform
-  ISMN reader can now get the data coverage for stations and networks
-  ISMN interface can now be restricted to a list of networks
-  added python3 support
-  moved grid functionality to pygeogrids package, pytesmo grids are
   deprecated and will be removed in future releases
-  include triple collocation example and improve documentation see
   issue #24

v0.2.5, 2014-12-15
==================

-  fixed ASCAT verion detection for latest H25 dataset WARP55R22
-  added example for Soil Water Index calculation

v0.2.4, 2014-12-09
==================

-  moved to pyscaffold structure
-  added tests for modules
-  added grid generation routines
-  fix for issue #15
-  updated classes to work with new base classes, does not change API
-  added travis CI support
-  changed theme of documentation, and enabled read the docs

v0.2.3, 2014-10-03
==================

-  added grouping module

v0.2.2, 2014-10-03
==================

-  fixed bug that lead to old grids without shape information not
   loading

v0.2.1, 2014-8-14
=================

-  added functionality to save grid as 2 dimensional array in
   grid.netcdf if grid is regular and shape information is given

v0.2.0, 2014-06-12
==================

-  added readers, tests and examples for H-SAF image products H07, H08
   and H14
-  added resample method that makes using pyresample a easier for the
   dictionary structure that pytesmo uses for image data
-  added colormap reader for custom colormaps

v0.1.3, 2014-05-26
==================

-  fixed bug in grid.nearest\_neighbour that caused different results on
   different systems. Radians are now always calculated at 64bit
   accuracy
-  ISMN routines now read the new ISMN download format
-  df\_metrics.bias now also returns a namedtuple

v0.1.2, 2014-04-16
==================

-  Reader for different versions of netCDF H25 HSAF product
-  added functionality to save grid definitions to netCDF files
-  Fixed Bug that masked all data if snow probabilities did not exist
-  Added tests

v0.1.1, 2013-11-18
==================

-  Added readers for netCDF H25 HSAF product
-  Added readers for netCDF ERS soil moisture product
-  Added general grid classes
-  Performance improvements for anomaly and climatology calculation
   through usage of cython
-  Introduced df\_metrics module for convienent calculation of metrics
   for data saved in pandas.DataFrames
