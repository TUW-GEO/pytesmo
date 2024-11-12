try:
    from itertools import izip as zip
except ImportError:
    # python 3
    pass

import numpy as np
import pandas as pd
from pygeogrids.grids import CellGrid
import warnings
import logging
from typing import Mapping, Tuple, List

from pytesmo.validation_framework.data_manager import DataManager
from pytesmo.validation_framework.data_manager import get_result_names
from pytesmo.validation_framework.data_manager import get_result_combinations
from pytesmo.validation_framework.data_scalers import DefaultScaler
import pytesmo.validation_framework.temporal_matchers as temporal_matchers
from pytesmo.utils import ensure_iterable
from distutils.version import LooseVersion

import pytesmo.validation_framework.error_handling as eh


class Validation(object):

    """
    Class for the validation process.

    Parameters
    ----------
    datasets : dict of dicts or DataManager
        :Keys: string, datasets names
        :Values: dict, containing the following fields
        :py:class:`pytesmo.validation_framework.data_manager.DataManager`

            'class': object
                Class containing the method read_ts for reading the data.
            'columns': list
                List of columns which will be used in the validation process.
            'args': list, optional
                Args for reading the data.
            'kwargs': dict, optional
                Kwargs for reading the data
            'grids_compatible': boolean, optional
                If set to True the grid point index is used directly when
                reading other, if False then lon, lat is used and a nearest
                neighbour search is necessary.
            'use_lut': boolean, optional
                If set to True the grid point index (obtained from a
                calculated lut between reference and other) is used when
                reading other, if False then lon, lat is used and a
                nearest neighbour search is necessary.
            'lut_max_dist': float, optional
                Maximum allowed distance in meters for the lut calculation.
    spatial_ref: string
        Name of the dataset used as a spatial, temporal and scaling reference.
        temporal and scaling references can be changed if needed. See the
        optional parameters ``temporal_ref`` and ``scaling_ref``.
    metrics_calculators : dict of functions
        The keys of the dict are tuples with the following structure: (n, k)
        with n >= 2 and n>=k. n must be equal to the number of datasets now.
        n is the number of datasets that should be temporally matched to the
        reference dataset and k is how many columns the metric calculator
        will get at once.
        What this means is that it is e.g. possible to temporally match 3
        datasets with 3 columns in total and then give the combinations of
        these columns to the metric calculator in sets of 2 by specifying the
        dictionary like:

        .. code::

            { (3, 2): metric_calculator}

        The values are functions that take an input DataFrame with the columns
        'ref' for the reference and 'n1', 'n2' and
        so on for other datasets as well as a dictionary mapping the column
        names to the names of the original datasets. In this way multiple
        metric calculators can be applied to different combinations of n
        input datasets.
    temporal_matcher: function, optional
        function that takes a dict of dataframes and a reference_key.
        It performs the temporal matching on the data and returns a dictionary
        of matched DataFrames that should be evaluated together by the metric
        calculator.
    temporal_window: float, optional
        Window to allow in temporal matching in days. The window is allowed
        on both sides of the timestamp of the temporal reference data.
        Only used with the standard temporal matcher.
    temporal_ref: string, optional
        If the temporal matching should use another dataset than the spatial
        reference as a reference dataset then give the dataset name here.
    period : list, optional
        Of type [datetime start, datetime end]. If given then the two input
        datasets will be truncated to start <= dates <= end.
    masking_datasets : dict of dictionaries
        Same format as the datasets with the difference that the `read`
        method of these datasets has to return pandas.DataFrames with only
        boolean columns. True means that the observations at this timestamp
        should be masked and False means that it should be kept.
    scaling : str or None or class instance
        - If set then the data will be scaled into the reference space using
          the method specified by the string using the
          :py:class:`pytesmo.validation_framework.data_scalers.DefaultScaler`
          class.
        - If set to None then no scaling will be performed.
        - It can also be set to a class instance that implements a
          ``scale(self, data, reference_index, gpi_info)`` method. See
          :py:class:`pytesmo.validation_framework.data_scalers.DefaultScaler`
          for an example.
    scaling_ref : string, optional
        If the scaling should be done to another dataset than the spatial
        reference then give the dataset name here.

    Methods
    -------
    calc(job)
        Takes either a cell or a gpi_info tuple and performs the validation.
    get_processing_jobs()
        Returns processing jobs that this process can understand.
    """

    def __init__(
        self,
        datasets,
        spatial_ref,
        metrics_calculators,
        temporal_matcher=None,
        temporal_window=1 / 24.0,
        temporal_ref=None,
        masking_datasets=None,
        period=None,
        scaling="cdf_match",
        scaling_ref=None,
    ):

        if isinstance(datasets, DataManager):
            self.data_manager = datasets
        else:
            self.data_manager = DataManager(datasets, spatial_ref, period)

        self.temp_matching = temporal_matcher
        if self.temp_matching is None:
            warnings.warn(
                "You are using the default temporal matcher. If you are using "
                "one of the newer metric calculators "
                "(PairwiseIntercomparisonMetrics, TripleCollocationMetrics) "
                "you should probably use `make_combined_temporal_matcher`"
                " instead. Have a look at the documentation of the metric "
                "calculators for more info."
            )
            self.temp_matching = temporal_matchers.BasicTemporalMatching(
                window=temporal_window
            ).combinatory_matcher

        self.temporal_ref = temporal_ref
        if self.temporal_ref is None:
            self.temporal_ref = self.data_manager.reference_name

        self.metrics_c = metrics_calculators
        for n, k in self.metrics_c:
            if n < len(self.data_manager.datasets.keys()):
                raise ValueError("n must be equal to the number of datasets")

        self.masking_dm = None
        if masking_datasets is not None:
            # add temporal reference dataset to the masking datasets since it
            # is necessary for temporally matching the masking datasets to the
            # common time stamps. Use _reference here to make a clash with the
            # names of the masking datasets unlikely
            masking_datasets.update(
                {"_reference": datasets[self.temporal_ref]}
            )
            self.masking_dm = DataManager(
                masking_datasets, "_reference", period=period
            )

        if type(scaling) == str:
            self.scaling = DefaultScaler(scaling)
        else:
            self.scaling = scaling
        self.scaling_ref = scaling_ref
        if self.scaling_ref is None:
            self.scaling_ref = self.data_manager.reference_name

        self.luts = self.data_manager.get_luts()

    def calc(
        self,
        gpis,
        lons,
        lats,
        *args,
        rename_cols=True,
        only_with_reference=False,
        handle_errors='raise',
    ) -> Mapping[Tuple[str], Mapping[str, np.ndarray]]:
        """
        The argument iterables (lists or numpy.ndarrays) are processed one
        after the other in tuples of the form (gpis[n], lons[n], lats[n],
        arg1[n], ..).

        Parameters
        ----------
        gpis: iterable
            The grid point indices is an identificator by which the
            spatial reference dataset can be read. This is either a list
            or a numpy.ndarray or any other iterable containing this indicator.
        lons: iterable
            Longitudes of the points identified by the gpis. Has to be the same
            size as gpis.
        lats: iterable
            latitudes of the points identified by the gpis. Has to be the same
            size as gpis.
        args: iterables
            any addiational arguments have to have the same size as the gpis
            iterable. They are given to the metrics calculators as
            metadata. Common usage is e.g. the long name or network name of an
            in situ station.
        rename_cols : bool, optional
            Whether to rename the columns to "ref", "k1", ... before passing
            the dataframe to the metrics calculators. Default is True.
        only_with_reference : bool, optional
            If this is enabled, only combinations that include the reference
            dataset (from the data manager) are calculated.
        handle_errors: str, optional (default: 'raise')
            Governs how to handle errors::

            * `raise`: If an error occurs during validation, raise exception.
            * `ignore`: If an error occurs, assign the correct return code
              to the result template and continue with the next GPI.

        Returns
        -------
        compact_results : dict of dicts
            :Keys: result names, combinations of
                  (referenceDataset.column, otherDataset.column)
            :Values: dict containing the elements returned
                  by metrics_calculator

        """
        handle_errors = handle_errors.lower()
        error_handling_options = ["raise", "ignore"]
        assert handle_errors in error_handling_options, (
            f"'handle_errors' must be one of {error_handling_options}"
        )

        if len(args) > 0:
            gpis, lons, lats, args = args_to_iterable(
                gpis, lons, lats, *args, n=3
            )
        else:
            gpis, lons, lats = args_to_iterable(gpis, lons, lats)

        results = {}
        for gpi_info in zip(gpis, lons, lats, *args):

            try:
                try:
                    df_dict = self.data_manager.get_data(
                        gpi_info[0], gpi_info[1], gpi_info[2]
                    )
                except Exception as e:
                    raise eh.DataManagerError(
                        f"Getting the data for gpi {gpi_info} failed with"
                        f" error: {e}")

                # if no data is available continue with the next gpi
                if len(df_dict) == 0:
                    raise eh.NoGpiDataError(f"No data for gpi {gpi_info}")
                matched_data, result, used_data = self.perform_validation(
                    df_dict,
                    gpi_info,
                    rename_cols=rename_cols,
                    only_with_reference=only_with_reference,
                    handle_errors=handle_errors,
                )
            except Exception as e:
                if handle_errors == 'raise':
                    raise e
                elif handle_errors == "ignore":
                    logging.error(f"{gpi_info}: {e}")
                    result = self.dummy_validation_result(
                        gpi_info, rename_cols=rename_cols,
                        only_with_reference=only_with_reference)
                    if isinstance(e, eh.ValidationError):
                        retcode = e.return_code
                    else:
                        retcode = eh.VALIDATION_FAILED
                    for key in result:
                        for k in result[key][0].keys():
                            # default case or subgroups status update
                            if (isinstance(k, str) and k == "status") or \
                               (isinstance(k, tuple) and k[1] == "status"):
                                result[key][0][k][0] = retcode

            # add result of one gpi to global results dictionary
            for r in result:
                if r not in results:
                    results[r] = []
                results[r] = results[r] + result[r]

        # So far, results is a dictionary mapping from a validation name/key
        # (the involved datasets) to lists of individual result dictionaries
        # for each gpi.
        # Here, these lists are summarized to have a single dictionary of numpy
        # arrays for each validation key
        compact_results = {}
        for key in results.keys():
            compact_results[key] = {}
            for field_name in results[key][0].keys():
                entries = []
                for result in results[key]:
                    entries.append(result[field_name][0])
                compact_results[key][field_name] = np.array(
                    entries, dtype=results[key][0][field_name].dtype
                )

        return compact_results

    def perform_validation(
        self,
        df_dict,
        gpi_info,
        rename_cols=True,
        only_with_reference=False,
        handle_errors="raise",
    ) -> Mapping[Tuple[str], List[Mapping[str, np.ndarray]]]:
        """
        Perform the validation for one grid point index and return the
        matched datasets as well as the calculated metrics.

        Parameters
        ----------
        df_dict: dict of pandas.DataFrames
            DataFrames read by the data readers for each dataset
        gpi_info: tuple
            tuple of at least, (gpi, lon, lat)
        rename_cols : bool, optional
            Whether to rename the columns to "ref", "k1", ... before passing
            the dataframe to the metrics calculators. Default is True.
        only_with_reference: bool, optional (default: False)
            Only compute metrics for dataset combinations where the reference
            is included.

        Returns
        -------
        matched_n: dict of pandas.DataFrames
            temporally matched data stored by (n, k) tuples
        results: dict
            Dictonary of calculated metrics stored by dataset combinations
            tuples.
        used_data: dict
            The DataFrame used for calculation of each set of metrics.

        Raises
        ------
        eh.TemporalMatchingError :
            If temporal matching failed
        eh.NoTempMatchedDataError :
            If there is insufficient data or the temporal matching did not
            return data.
        eh.ScalingError :
            If scaling failed
        """
        results = {}
        used_data = {}
        matched_n = {}

        if self.masking_dm is not None:
            ref_df = df_dict[self.temporal_ref]
            masked_ref_df = self.mask_dataset(ref_df, gpi_info)
            if len(masked_ref_df) == 0:
                return matched_n, results, used_data

            df_dict[self.temporal_ref] = masked_ref_df

        # we only need the data columns
        data_df_dict = {}
        for ds in df_dict:
            columns = self.data_manager.datasets[ds]["columns"]
            data_df_dict[ds] = df_dict[ds][columns]

        # matched_n is a dictionary mapping from dataset combinations as keys
        # to pandas dataframes
        try:
            matched_n = self.temporal_match_datasets(data_df_dict)
        except Exception:
            raise eh.TemporalMatchingError(
                f"Temporal matching failed for gpi {gpi_info}!"
            )

        for n, k in self.metrics_c:
            metrics_calculator = self.metrics_c[(n, k)]

            def dummy_result():
                # to get only an empty template returned, we pass an
                # empty dataframe here, with the correct column names
                dummy_df = pd.DataFrame([], columns=result_ds_names)
                metrics = metrics_calculator(dummy_df, gpi_info)
                return metrics

            n_matched_data = matched_n[(n, k)]
            if len(n_matched_data) == 0:
                # this would happen if self.temp_matching returns an empty list
                # or dictionary for n=n, k=k
                raise eh.NoTempMatchedDataError(
                    f"No temporally matched data for ({n}, {k})"
                    f" and metric calculator {self.metrics_c[(n,k)]}"
                    f" for gpi {gpi_info}!"
                )
            result_names = get_result_combinations(
                self.data_manager.ds_dict, n=k
            )
            for data, result_key in self.k_datasets_from(
                n_matched_data, result_names
            ):

                # it might also be a good idea to move this to
                # `get_result_combinations`
                result_ds_names = [key[0] for key in result_key]
                if only_with_reference:
                    if self.data_manager.reference_name not in result_ds_names:
                        continue

                if result_key not in results.keys():
                    results[result_key] = []

                if len(data) == 0:
                    if handle_errors == "raise":
                        raise eh.NoTempMatchedDataError(
                            f"Temporal matching resulted in empty dataset for"
                            f" {result_key} for gpi {gpi_info}"
                        )
                    else:
                        metrics = dummy_result()
                        metrics["status"][0] = eh.NO_TEMP_MATCHED_DATA
                        results[result_key].append(metrics)
                        continue

                # at this stage we can drop the column multiindex and just use
                # the dataset name
                data = data.rename(columns=lambda x: x[0])

                if self.scaling is not None:
                    # get scaling index by finding the column in the
                    # DataFrame that belongs to the scaling reference
                    scaling_index = data.columns.tolist().index(
                        self.scaling_ref
                    )
                    try:
                        data = self.scaling.scale(
                            data, scaling_index, gpi_info
                        )
                    except Exception as e:
                        raise eh.ScalingError(
                            f"Scaling failed for {result_key} for gpi"
                            f" {gpi_info} with error {e}!"
                        )

                    # Drop the scaling reference if it was not in the intended
                    # results
                    if self.scaling_ref not in [key[0] for key in result_key]:
                        data = data.drop(columns=[self.scaling_ref])

                # Rename the columns to 'ref', 'k1', 'k2', ...
                if rename_cols:
                    rename_dict = {}
                    for i, r in enumerate(result_key):
                        rename_dict[r[0]] = f"k{i}" if i > 0 else "ref"
                    data.rename(columns=rename_dict, inplace=True)

                used_data[result_key] = data
                try:
                    metrics = metrics_calculator(data, gpi_info)
                except Exception:
                    if handle_errors == "raise":
                        raise eh.MetricsCalculationError(
                            f"Metrics calculation failed for {result_key}"
                            f" for gpi {gpi_info}!"
                        )
                    else:
                        metrics = dummy_result()
                        metrics["status"][0] = eh.METRICS_CALCULATION_FAILED
                results[result_key].append(metrics)

        return matched_n, results, used_data

    def dummy_validation_result(
        self,
        gpi_info,
        rename_cols=True,
        only_with_reference=False,
    ) -> Mapping[Tuple[str], List[Mapping[str, np.ndarray]]]:
        """
        Creates an empty result dictionary to be used if perform_validation
        fails
        """
        results = {}
        for n, k in self.metrics_c:
            result_names = get_result_combinations(
                self.data_manager.ds_dict, n=k
            )
            for result_key in result_names:
                # it might also be a good idea to move this to
                # `get_result_combinations`
                result_ds_names = [key[0] for key in result_key]
                if only_with_reference:
                    if self.data_manager.reference_name not in result_ds_names:
                        continue
                metrics_calculator = self.metrics_c[(n, k)]
                # to get only an empty template returned, we pass an empty
                # dataframe here, with the correct column names
                dummy_df = pd.DataFrame([], columns=result_ds_names)
                metrics = metrics_calculator(dummy_df, gpi_info)
                if result_key not in results.keys():
                    results[result_key] = []
                results[result_key].append(metrics)
        return results

    def mask_dataset(self, ref_df, gpi_info):
        """
        Mask the temporal reference dataset with the data read
        through the masking datasets.

        Parameters
        ----------
        gpi_info: tuple
            tuple of at least, (gpi, lon, lat)

        Returns
        -------
        mask: numpy.ndarray
            boolean array of the size of the temporal reference read
        """

        matched_masking = self.temporal_match_masking_data(ref_df, gpi_info)
        # this will only be one element since n is the same as the
        # number of masking datasets
        result_names = get_result_names(
            self.masking_dm.ds_dict, "_reference", n=2
        )
        choose_all = pd.DataFrame(index=ref_df.index)
        for data, result in self.k_datasets_from(
            matched_masking, result_names, include_scaling_ref=False
        ):
            if len(data) == 0:
                continue

            for key in result:
                if key[0] != "_reference":
                    # this is necessary since the boolean datatype might have
                    # been changed to float 1.0 and 0.0 issue with temporal
                    # resampling that is not easily resolved since most
                    # datatypes have no nan representation.
                    choose = pd.Series((data[key] is False), index=data.index)
                    choose = choose.reindex(
                        index=choose_all.index, fill_value=True
                    )
                    choose_all[key] = choose.copy()
        choosing = choose_all.apply(np.all, axis=1)

        return ref_df[choosing]

    def temporal_match_masking_data(self, ref_df, gpi_info):
        """
        Temporal match the masking data to the reference DataFrame

        Parameters
        ----------
        ref_df: pandas.DataFrame
            Reference data
        gpi_info: tuple or list
            contains, (gpi, lon, lat)

        Returns
        -------
        matched_masking: dict of pandas.DataFrames
            Contains temporally matched masking data. This dict has only one
            key being a tuple that contains the matched datasets.
        """

        # read only masking datasets and use the already read reference
        masking_df_dict = self.masking_dm.get_other_data(
            gpi_info[0], gpi_info[1], gpi_info[2]
        )
        masking_df_dict.update({"_reference": ref_df})
        matched_masking = self.temp_matching(
            masking_df_dict, "_reference", n=2
        )
        return matched_masking

    def temporal_match_datasets(self, df_dict):
        """
        Temporally match all the requested combinations of datasets.

        Parameters
        ----------
        df_dict: dict of pandas.DataFrames
            DataFrames read by the data readers for each dataset

        Returns
        -------
        matched_n: dict of pandas.DataFrames
            for each (n, k) in the metrics calculators the n temporally
            matched dataframes
        """

        matched_n = {}
        for n, k in self.metrics_c:
            matched_data = self.temp_matching(
                df_dict, self.temporal_ref, n=n, k=k
            )

            matched_n[(n, k)] = matched_data

        return matched_n

    def k_datasets_from(
        self, n_matched_data, result_names, include_scaling_ref=True
    ):
        """
        Extract k datasets from n temporally matched ones.

        This is used to send combinations of k datasets to
        metrics calculators expecting only k datasets.

        Parameters
        ----------
        n_matched_data: dict of pandas.DataFrames
            DataFrames in which n datasets were temporally matched.
            The key is a tuple of the dataset names.
        result_names: list
            result names to extract
        include_scaling_ref: boolean, optional
            if set the scaling reference will always be included.
            Should only be disabled for getting the masking datasets

        Yields
        ------
        data: pd.DataFrame
            pandas DataFrame with k columns extracted from the
            temporally matched datasets
        result: tuple
            Tuple describing which datasets and columns are in
            the returned data. ((dataset_name, column_name),
            (dataset_name2, column_name2))
        """

        for result in result_names:
            result_extract = result
            if self.scaling is not None and include_scaling_ref:
                # always make sure the scaling reference is included in the
                # results otherwise the scaling will fail
                scaling_ref_column = self.data_manager.datasets[
                    self.scaling_ref
                ]["columns"][0]
                scaling_result_name = (self.scaling_ref, scaling_ref_column)
                if scaling_result_name not in result:
                    result_extract = result + (scaling_result_name,)
            data = self.get_data_for_result_tuple(
                n_matched_data, result_extract
            )
            yield data, result

    def get_data_for_result_tuple(self, n_matched_data, result_tuple):
        """
        Extract a dataframe for a given result tuple from the
        matched dataframes.

        Parameters
        ----------
        n_matched_data: dict of pandas.DataFrames
            DataFrames in which n datasets were temporally matched.
            The key is a tuple of the dataset names.
        result_tuple: tuple
            Tuple describing which datasets and columns should be
            extracted. ((dataset_name, column_name),
                        (dataset_name2, column_name2))

        Returns
        -------
        data: pd.DataFrame
            pandas DataFrame with columns extracted from the
            temporally matched datasets
        """
        # find the key into the temporally matched dataset by combining the
        # dataset parts of the result_names
        dskey = []
        for i, r in enumerate(result_tuple):
            dskey.append(r[0])

        dskey = tuple(dskey)
        if len(list(n_matched_data)[0]) == len(dskey):
            # we should have an exact match of datasets and
            # temporal matches

            # still need to make sure that dskey is in the right order and
            # contains all the same datasets as the n_matched_data
            for key in n_matched_data:
                if sorted(dskey) == sorted(key):
                    dskey = key
                    break
            if dskey not in n_matched_data:
                return []
            else:
                data = n_matched_data[dskey]
        else:
            # more datasets were temporally matched than are
            # requested now so we select a temporally matched
            # dataset that has the first key in common with the
            # temporal reference.

            # This guarantees that we only select columns from dataframes for
            # which the temporal reference dataset was included in the
            # temporal matching

            first_match = [
                key for key in n_matched_data if self.temporal_ref == key[0]
            ]
            found_key = None
            for key in first_match:
                for dsk in dskey:
                    if dsk not in key:
                        continue
                found_key = key
            data = n_matched_data[found_key]

        # extract only the relevant columns from matched DataFrame
        data = data[[x for x in result_tuple]]
        # drop values if one column is NaN
        data = data.dropna()
        return data

    def get_processing_jobs(self):
        """
        Returns processing jobs that this process can understand.

        Returns
        -------
        jobs : list
            List of cells or gpis to process.
        """
        jobs = []
        if self.data_manager.reference_grid is not None:
            if type(self.data_manager.reference_grid) is CellGrid:
                cells = self.data_manager.reference_grid.get_cells()
                for cell in cells:
                    (
                        cell_gpis,
                        cell_lons,
                        cell_lats,
                    ) = self.data_manager.reference_grid.grid_points_for_cell(
                        cell
                    )
                    jobs.append([cell_gpis, cell_lons, cell_lats])
            else:
                (
                    gpis,
                    lons,
                    lats,
                ) = self.data_manager.reference_grid.get_grid_points()
                jobs = [gpis, lons, lats]

        return jobs


def args_to_iterable(*args, **kwargs):
    """
    Convert arguments to iterables.


    Parameters
    ----------
    args: iterables or not
        arguments
    n : int, optional
        number of explicit arguments
    """
    if "n" in kwargs:
        n = kwargs["n"]
    else:
        n = len(args)

    arguments = []
    for i, arg in enumerate(args):
        arguments.append(ensure_iterable(arg))

    it = iter(arguments)
    for _ in range(n):
        yield next(it, None)
    if n < len(args):
        yield tuple(it)
