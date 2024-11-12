=========
Changelog
=========

Unreleased changes in master
============================

Version 0.16.1, 2024-11-13
==========================
- THIS VERSION IS STILL COMPATIBLE WITH ``NUMPY<2.0``, from v0.17 on pytesmo will require ``numpy>=2.0``
- Old ``pytesmo.timeseries.plotting`` module was removed
- Fixed and filtered many warnings that were printed by tests until now
- Fixed an issue with the intra-annual metrics adapter when an empty time series is passed (PR `#307 <https://github.com/TUW-GEO/pytesmo/pull/307>`_)
- Metapackage updated (pyscaffold 4.5) (PR `#307 <https://github.com/TUW-GEO/pytesmo/pull/307>`_)
- C modules were outdated and could not be compiled, therefore rebuilt (PR `#307 <https://github.com/TUW-GEO/pytesmo/pull/307>`_)

Version 0.16.0, 2023-09-12
==========================
- Adapters for metric calculatores were introduced. The here implemented adapters compute metrics based on temporal subsets of the time series, which can be used for rolling window metrics, seasonal metrics or multiple arbitrary sub-periods. (PR `#266 <https://github.com/TUW-GEO/pytesmo/pull/266>`_)
- Fixed a bug where the status code of a successful Triple Collocation run was still set to -1 ("unknown error"). (PR `#301 <https://github.com/TUW-GEO/pytesmo/pull/301>`_)

Version 0.15.2, 2023-06-14
==========================
- Add kwarg to `time_series.anomaly.calc_climatology` to compute the standard deviation of the climatology (e.g. for anomaly z-scores) (PR `#290 <https://github.com/TUW-GEO/pytesmo/pull/290>`_)

Version 0.15.1, 2023-03-29
==========================
- Adding scikit-learn as installation dependency. (PR `#298 <https://github.com/TUW-GEO/pytesmo/pull/298>`_)

Version 0.15.0, 2023-03-28
==========================
- fixing setting of status code for bootstrapped triple collocation  (PR `#285 <https://github.com/TUW-GEO/pytesmo/pull/285>`_)
- Exclude netCDF4 at version 1.6.2 due to bug when writing strings with zlib compression (PR `#288 <https://github.com/TUW-GEO/pytesmo/pull/288>`_)
- added test for data_manager_failed (PR `#287 <https://github.com/TUW-GEO/pytesmo/pull/287>`_)
- `read_ts` is not longer used in pytesmo (PR `#289 <https://github.com/TUW-GEO/pytesmo/pull/289>`_)
- support for `max_dist` keyword in validation framework to limit comparisons to close data points (PR `#291 <https://github.com/TUW-GEO/pytesmo/pull/291>`_)
- Reimplementation of CDF matching (PR `#259 <https://github.com/TUW-GEO/pytesmo/pull/259>`_). This removed all other implementations, i.e. ``lin_cdf_match`` and ``cdf_beta_match``. These now are either removed (``lin_cdf_match``), or point to the new implementation (``cdf_beta_match``)

Version 0.14.4, 2023-01-27
==========================
- option to ignore NaN values in filter fields (PR `#282 <https://github.com/TUW-GEO/pytesmo/pull/282>`_)

Version 0.14.3, 2022-12-26
==========================
- fixing rounding error issues in metric calculations (PR `#281 <https://github.com/TUW-GEO/pytesmo/pull/281>`_)

Version 0.14.2, 2022-12-14
==========================
- small bug fixes/doc updates (PRs `#273 <https://github.com/TUW-GEO/pytesmo/pull/273>`_, `#275 <https://github.com/TUW-GEO/pytesmo/pull/275>`_, `#276 <https://github.com/TUW-GEO/pytesmo/pull/276>`_,  `#278 <https://github.com/TUW-GEO/pytesmo/pull/278>`_)
- more consistent error handling (PR `#277 <https://github.com/TUW-GEO/pytesmo/pull/277>`_

Version 0.14.1, 2022-10-25
==========================
- small bug fixes and documentation updates (PRs `#272 <https://github.com/TUW-GEO/pytesmo/pull/272>`_, `#271 <https://github.com/TUW-GEO/pytesmo/pull/271>`_, `#270 <https://github.com/TUW-GEO/pytesmo/pull/270>`_)

Version 0.14.0, 2022-05-13
==========================
- fixed bugs in resample_mean (PR `#262 <https://github.com/TUW-GEO/pytesmo/pull/262>`_)
- Integration tests with new ISMN Interface (PR `#263 <https://github.com/TUW-GEO/pytesmo/pull/263>`_)
- Add option to Validation Framework to ignore errors in ``Validation.calc()`` (PR `#263 <https://github.com/TUW-GEO/pytesmo/pull/263>`_)
- ipynb files from docs/examples are now also used as (optional) tests (PR `#263 <https://github.com/TUW-GEO/pytesmo/pull/263>`_)
- ``yapf`` for code formatting (see developers guide) (Fix #248, PR `#263 <https://github.com/TUW-GEO/pytesmo/pull/263>`_)
- validation framework option to force dataset combinations that include reference dataset updated (PR `#265 <https://github.com/TUW-GEO/pytesmo/pull/265>`_)
- Added `TimestampAdapter` to the validation framework to deal with datasets that have different date/time specification fields (PR `#268 <https://github.com/TUW-GEO/pytesmo/pull/268>`_)

Version 0.13.4, 2022-01-12
==========================
- use `oldest-supported-numpy` in `pyproject.toml` to avoid errors due to incompatible numpy versions used in build-time.

Version 0.13.3, 2022-01-11
==========================
- rollback to old C-files due to numpy compatibility issues

Version 0.13.2, 2021-12-17
==========================
- monthly climatologies
- ``wraparound=True`` is the new default in the climatology calculation
- better error handling in validation framework

Version 0.13.1, 2021-12-16
==========================
- Merge pull request #249 from pstradio/monthly_clim
- Monthly climatology

Version 0.13.0, 2021-12-16
==========================
- monthly climatologies
- wraparound=True is the new default in the climatology calculation
- better error handling in validation framework

Version 0.12.0, 2021-08-19
==========================
- Add option for custom reading method name to all reading adapters (PR `#245 <https://github.com/TUW-GEO/pytesmo/pull/245>`_)
- `MonthMetricsAdapter` for new `PairwiseIntercomparisonMetrics` and `TripleCollocationMetrics` calculators (PR `#246 <https://github.com/TUW-GEO/pytesmo/pull/246>`_)

Version 0.11.4, 2021-07-13
==========================
- fixed some bugs in the confidence interval calculation, for some metrics only bootstrapped CIs are
  available now
- fixed bug in rolling metric calculation (now also works with float32 input data)
- minor bug fixes

Version 0.11.3, 2021-06-01
==========================
- removed ``ascat`` and ``ismn`` from the package dependencies
- fixed a few bugs in ``pytesmo.temporal_collocation.combined_temporal_collocation``.
- updated documentation
- added Python 3.9 to CI

Version 0.11.0, 2021-05-12
==========================
- Analytical and bootstrapping confidence intervals for metrics (PR #206). This
  includes some changes to the existing implementations (all old
  implementations are still available, but deprecated)

  - all pairwise metric functions take two arrays as input and return a single value
  - the correlation metrics (``pearsonr``, ``spearmanr``, ``kendalltau``) have new
    versions ``pearson_r``, ``spearman_r``, and ``kendall_tau`` which only return the
    correlation value, but not the p-value. The old functions have been
    deprecated. For calculating correlation + p-value, it is advised to use
    ``scipy.stats.pearsonr``, ``scipy.stats.spearmanr``, and
    ``scipy.stats.kendalltau``. Instead of p-values, confidence intervals for
    the correlation coefficients could be obtained with::

      r, lower, upper = with_analytical_ci(pearson_r, x, y)

  - ``pytesmo.metrics.tcol_error`` and ``pytesmo.metrics.tcol_snr`` have been
    deprecated. Use ``pytesmo.metrics.tcol_metrics`` instead (which is simply a
    renaming of ``tcol_snr``).
  - ``pytesmo.metrics.mse`` has been deprecated. There is a new, much faster
    implementation available (``pytesmo.metrics.mse_decomposition``).
    Individual values of the components can be calculated with
    ``pytesmo.metrics.mse``, ``pytesmo.metrics.mse_corr``,
    ``pytesmo.metrics.mse_bias``, ``pytesmo.metrics.mse_var``.
- Removed dependency on deprecated Numpy API
- added mean resampling in temporal collocation
- updated to ``ascat`` version 2.0


Version 0.10.0, 2021-02-15
==========================

- Drop Travis CI and Appveyor CI, replaced by Github Actions ci.yml workflow
- New implementation of temporal collocation based on pd.reindex (PR #204):
  :py:func:`pytesmo.temporal_matching.temporal_collocation`
- CDF matching function updated
- min_obs keyword added to MetricsCalculators


Version 0.9.1, 2020-09-14
=========================

- Include more combinations in validation framework, raise error if n < n_datasets
- `n_wise_apply` now can handle (drop) nans in a passed data frame correctly.
- TC metrics calculator has now option to calculate metrics for reference
- Fix deprecation warnings in anomaly adapter (Issue #198)
- Change combinations created by val framework, catch cases where scaling ref not in combinations (Issue #187)


Version 0.9, 2020-07-02
=======================

- Update pyscaffold v3.2.3
- Rolling Prearson's R and RMSD metrics calculator
- New results manager to handle rolling metrics
- Add deprecation warning for colormaps module
- Move timedate module to cadati, import from there
- Filter warnings in places where they are safe to ignore and not caused by pytesmo
- Update Docs and ipython notebooks

Version 0.8, 2019-11-21
=======================

- Update Triple Collocation metrics and metrics caluclator
- Update metrics apply function, use dictionaries instead of dataframe for results
- Remove io module and grids.grid.py and grids.netcdf.py modules
- Move resampling module to repurpose package (https://github.com/TUW-GEO/repurpose)
- Documentation updates

Version 0.7.1, 2019-09-03
==========================

- Change naming convention for Intercomparison metrics calculator results
- Use kdtree from pykdtree instead of scipy (faster)
- Add MetadataMetrics to metrics calculator
- Remove pybufr-ecmwf, use generic test data for testing spatial resampling

Version 0.7.0, 2019-05-15
=========================

- Add option for filling no data values to moving average
- Add option for minimum observations to moving average
- Default behavior of climatology calculation will now fill no_data values
  Nothing is changing if there are no no_data values in the time series

Version 0.6.11, 2019-05-15
==========================

- Add more metrics calculators
- Remove pytesmo.io.ismn module and move it to the ismn package
- Update tests for ascat v1.0 package
- Rename arguments for metrics functions to be more general
- Split linreg scaling function to determine and apply corrections separately
- Compatible with Python 3.7 on Windows

Version 0.6.10, 2018-04-09
==========================

- Update readme
- Fix bug in exponential filter when first value is a NaN value

Version 0.6.9, 2018-02-06
=========================

- Add extendent collocation metric
- Fix initial value for exponential filter
- Fix #123

Version 0.6.8, 2017-08-29
=========================

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

Version 0.6.7, 2017-07-25
=========================

-  Add respect leap years option for climatology calculation.

Version 0.6.6, 2017-07-14
=========================

-  Compatible with Python 3.6

Version 0.6.5, 2017-07-10
=========================

-  Add additonal functions for working with dekads.

Version 0.6.4, 2017-06-02
=========================

-  Refactor resampling routine to be more modular and better usable
   outside of pytesmo.

Version 0.6.3, 2017-04-28
=========================

-  temporal matching performance improvement of approx. 50%
-  Add functions for handling dekadal dates. See
   ``pytesmo.timedate.dekad``.

Version 0.6.2, 2017-01-13
=========================

-  Fix metadata for new version of pypi.

Version 0.6.1, 2017-01-13
=========================

-  Add return\_clim keyword to anomaly calculation. Useful for getting
   both anomaly and climatology in one pandas.DataFrame. Also used in
   time series anomaly plot.
-  Fix bug in julian2date which led to negative microseconds in some
   edge cases.

Version 0.6.0, 2016-07-29
=========================

-  Moved the ASCAT readers to the ascat package. The functionality is
   the same, just replace ``import pytesmo.io.sat.ascat`` by
   ``import ascat`` and everything should work the same as before.
-  The H07 reader now returns also ssm mean as a value between 0 and
   100. Before it was between 0 and 1 and inconsistent with the other
   ssm values.
-  Fix small bug in julian date calculation and add tests for it.
-  Add hamming window to resample module

Version 0.5.2, 2016-04-26
=========================

-  Fix bugs when the validation framework encountered empty datasets for
   various reasons.
-  Add dataset adapters for masking and anomaly calculation.
-  Improve performance of moving average calculation and ISMN readers.

Version 0.5.1, 2016-04-21
=========================

-  Fix bug in jobs argument passing to Validation class.
-  Add support to use a pre initialized DataManager instance in the
   Validation class.
-  Add support for per dataset reading method names in the DataManager.
   This relaxes the assumption that every dataset has a ``read_ts``
   method.

Version 0.5.0, 2016-04-20
=========================

-  Fix bug in temporal resampling if input was a pandas.Series
-  Major refactoring of validation framework. Please see updated
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

Version 0.4.0, 2016-03-24
=========================

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

Version 0.3.6, 2015-12-10
=========================

-  make sure that climatologies are always 366 elements
-  add new options to climatology calculation for filling NaN values
-  add option to climatology calculation for wraparound before the
   smoothing

Version 0.3.5, 2015-11-04
=========================

-  fix bug in anomaly calculation that occurred when the climatology
   series had a name already
-  add option in anomaly calculation to respect leap years during
   matching
-  improve testing of scaling functions
-  add linear CDF scaling based on stored percentiles
-  add utility function for MATLAB like percentile calculation
-  add utility function for making sure elements in an array are unique
   by using iterative interpolation

Version 0.3.4, 2015-10-23
=========================

-  fix #63 by moving data preparation before period checks
-  fix bug in exponential and boxcar filter. Problem was that nan values
   were not ignored correctly

Version 0.3.3, 2015-08-26
=========================

-  add option to temporal resampling to exclude window boundaries
-  fix #48 by reintroducting netcdf imports
-  fix #60 by importing correctly from pygeogrids
-  fix #56 by allowing read\_bulk keyword for ASCAT\_SSM
-  fix #58 by using cKDTree keyword if available
-  lookup table indexing fixed, see #59

Version 0.3.2, 2015-07-09
=========================

-  hotfix for temporal resampling problem when time series where of
   unequal lenghts

Version 0.3.1, 2015-07-09
=========================

-  added validation framework and example on how to use it
-  fix bug (issue #51) in temporal matching
-  added test data as git submodule

Version 0.3.0, 2015-05-26
=========================

-  added calculation of pearson R confidence intervals based on fisher z
   transform
-  ISMN reader can now get the data coverage for stations and networks
-  ISMN interface can now be restricted to a list of networks
-  added python3 support
-  moved grid functionality to pygeogrids package, pytesmo grids are
   deprecated and will be removed in future releases
-  include triple collocation example and improve documentation see
   issue #24

Version 0.2.5, 2014-12-15
=========================

-  fixed ASCAT verion detection for latest H25 dataset WARP55R22
-  added example for Soil Water Index calculation

Version 0.2.4, 2014-12-09
=========================

-  moved to pyscaffold structure
-  added tests for modules
-  added grid generation routines
-  fix for issue #15
-  updated classes to work with new base classes, does not change API
-  added travis CI support
-  changed theme of documentation, and enabled read the docs

Version 0.2.3, 2014-10-03
=========================

-  added grouping module

Version 0.2.2, 2014-10-03
=========================

-  fixed bug that lead to old grids without shape information not
   loading

Version 0.2.1, 2014-8-14
========================

-  added functionality to save grid as 2 dimensional array in
   grid.netcdf if grid is regular and shape information is given

Version 0.2.0, 2014-06-12
=========================

-  added readers, tests and examples for H-SAF image products H07, H08
   and H14
-  added resample method that makes using pyresample a easier for the
   dictionary structure that pytesmo uses for image data
-  added colormap reader for custom colormaps

Version 0.1.3, 2014-05-26
=========================

-  fixed bug in grid.nearest\_neighbour that caused different results on
   different systems. Radians are now always calculated at 64bit
   accuracy
-  ISMN routines now read the new ISMN download format
-  df\_metrics.bias now also returns a namedtuple

Version 0.1.2, 2014-04-16
=========================

-  Reader for different versions of netCDF H25 HSAF product
-  added functionality to save grid definitions to netCDF files
-  Fixed Bug that masked all data if snow probabilities did not exist
-  Added tests

Version 0.1.1, 2013-11-18
=========================

-  Added readers for netCDF H25 HSAF product
-  Added readers for netCDF ERS soil moisture product
-  Added general grid classes
-  Performance improvements for anomaly and climatology calculation
   through usage of cython
-  Introduced df\_metrics module for convienent calculation of metrics
   for data saved in pandas.DataFrames
