try:
    from itertools import izip as zip
except ImportError:
    # python 3
    pass

import numpy as np
import pandas as pd
from pygeogrids.grids import CellGrid

from pytesmo.validation_framework.data_manager import DataManager
from pytesmo.validation_framework.data_manager import get_result_names
from pytesmo.validation_framework.data_scalers import DefaultScaler
import pytesmo.validation_framework.temporal_matchers as temporal_matchers
from pytesmo.utils import ensure_iterable
from distutils.version import LooseVersion

class Validation(object):

    """
    Class for the validation process.

    Parameters
    ----------
    datasets : dict of dicts, or pytesmo.validation_framwork.data_manager.DataManager
        Keys: string, datasets names
        Values: dict, containing the following fields
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
        temporal and scaling references can be changed if needed. See the optional parameters
        ``temporal_ref`` and ``scaling_ref``.
    metrics_calculators : dict of functions
        The keys of the dict are tuples with the following structure: (n, k) with n >= 2
        and n>=k. n is the number of datasets that should be temporally matched to the
        reference dataset and k is how many columns the metric calculator will get at once.
        What this means is that it is e.g. possible to temporally match 3 datasets with
        3 columns in total and then give the combinations of these columns to the metric
        calculator in sets of 2 by specifying the dictionary like:

        .. code::

            { (3, 2): metric_calculator}

        The values are functions that take an input DataFrame with the columns 'ref'
        for the reference and 'n1', 'n2' and
        so on for other datasets as well as a dictionary mapping the column names
        to the names of the original datasets. In this way multiple metric calculators
        can be applied to different combinations of n input datasets.
    temporal_matcher: function, optional
        function that takes a dict of dataframes and a reference_key.
        It performs the temporal matching on the data and returns a dictionary
        of matched DataFrames that should be evaluated together by the metric calculator.
    temporal_window: float, optional
        Window to allow in temporal matching in days. The window is allowed on both
        sides of the timestamp of the temporal reference data.
        Only used with the standard temporal matcher.
    temporal_ref: string, optional
        If the temporal matching should use another dataset than the spatial reference
        as a reference dataset then give the dataset name here.
    period : list, optional
        Of type [datetime start, datetime end]. If given then the two input
        datasets will be truncated to start <= dates <= end.
    masking_datasets : dict of dictionaries
        Same format as the datasets with the difference that the read_ts method of these
        datasets has to return pandas.DataFrames with only boolean columns. True means that the
        observations at this timestamp should be masked and False means that it should be kept.
    scaling : string, None or class instance
        - If set then the data will be scaled into the reference space using the
          method specified by the string using the
          :py:class:`pytesmo.validation_framework.data_scalers.DefaultScaler` class.
        - If set to None then no scaling will be performed.
        - It can also be set to a class instance that implements a
          ``scale(self, data, reference_index, gpi_info)`` method. See
          :py:class:`pytesmo.validation_framework.data_scalers.DefaultScaler` for an example.
    scaling_ref : string, optional
        If the scaling should be done to another dataset than the spatial reference then
        give the dataset name here.

    Methods
    -------
    calc(job)
        Takes either a cell or a gpi_info tuple and performs the validation.
    get_processing_jobs()
        Returns processing jobs that this process can understand.
    """

    def __init__(self, datasets, spatial_ref, metrics_calculators,
                 temporal_matcher=None, temporal_window=1 / 24.0,
                 temporal_ref=None,
                 masking_datasets=None,
                 period=None,
                 scaling='lin_cdf_match', scaling_ref=None):

        if type(datasets) is DataManager:
            self.data_manager = datasets
        else:
            self.data_manager = DataManager(datasets, spatial_ref, period)

        self.temp_matching = temporal_matcher
        if self.temp_matching is None:
            self.temp_matching = temporal_matchers.BasicTemporalMatching(
                window=temporal_window).combinatory_matcher

        self.temporal_ref = temporal_ref
        if self.temporal_ref is None:
            self.temporal_ref = self.data_manager.reference_name

        self.metrics_c = metrics_calculators

        self.masking_dm = None
        if masking_datasets is not None:
            # add temporal reference dataset to the masking datasets since it
            # is necessary for temporally matching the masking datasets to the
            # common time stamps. Use _reference here to make a clash with the
            # names of the masking datasets unlikely
            masking_datasets.update(
                {'_reference': datasets[self.temporal_ref]})
            self.masking_dm = DataManager(masking_datasets, '_reference',
                                          period=period)

        if type(scaling) == str:
            self.scaling = DefaultScaler(scaling)
        else:
            self.scaling = scaling
        self.scaling_ref = scaling_ref
        if self.scaling_ref is None:
            self.scaling_ref = self.data_manager.reference_name

        self.luts = self.data_manager.get_luts()

    def calc(self, gpis, lons, lats, *args):
        """
        The argument iterables (lists or numpy.ndarrays) are processed one after the other in
        tuples of the form (gpis[n], lons[n], lats[n], arg1[n], ..).

        Parameters
        ----------
        gpis : iterable
            The grid point indices is an identificator by which the
            spatial reference dataset can be read. This is either a list
            or a numpy.ndarray or any other iterable containing this indicator.
        lons: iterable
            Longitudes of the points identified by the gpis. Has to be the same size as gpis.
        lats: iterable
            latitudes of the points identified by the gpis. Has to be the same size as gpis.
        args: iterables
            any addiational arguments have to have the same size as the gpis iterable. They are
            given to the metrics calculators as metadata. Common usage is e.g. the long name
            or network name of an in situ station.

        Returns
        -------
        compact_results : dict of dicts
            Keys: result names, combinations of
                  (referenceDataset.column, otherDataset.column)
            Values: dict containing the elements returned by metrics_calculator
        """
        results = {}
        if len(args) > 0:
            gpis, lons, lats, args = args_to_iterable(gpis,
                                                      lons,
                                                      lats,
                                                      *args,
                                                      n=3)
        else:
            gpis, lons, lats = args_to_iterable(gpis, lons, lats)

        for gpi_info in zip(gpis, lons, lats, *args):

            df_dict = self.data_manager.get_data(gpi_info[0],
                                                 gpi_info[1],
                                                 gpi_info[2])

            # if no data is available continue with the next gpi
            if len(df_dict) == 0:
                continue
            matched_data, result, used_data = self.perform_validation(
                df_dict, gpi_info)

            # add result of one gpi to global results dictionary
            for r in result:
                if r not in results:
                    results[r] = []
                results[r] = results[r] + result[r]

        compact_results = {}
        for key in results.keys():
            compact_results[key] = {}
            for field_name in results[key][0].keys():
                entries = []
                for result in results[key]:
                    entries.append(result[field_name][0])
                compact_results[key][field_name] = \
                    np.array(entries, dtype=results[key][0][field_name].dtype)

        return compact_results

    def perform_validation(self,
                           df_dict,
                           gpi_info):
        """
        Perform the validation for one grid point index and return the
        matched datasets as well as the calculated metrics.

        Parameters
        ----------
        df_dict: dict of pandas.DataFrames
            DataFrames read by the data readers for each dataset
        gpi_info: tuple
            tuple of at least, (gpi, lon, lat)

        Returns
        -------
        matched_n: dict of pandas.DataFrames
            temporally matched data stored by (n, k) tuples
        results: dict
            Dictonary of calculated metrics stored by dataset combinations tuples.
        used_data: dict
            The DataFrame used for calculation of each set of metrics.
        """
        results = {}
        used_data = {}
        matched_n = {}

        if self.masking_dm is not None:
            ref_df = df_dict[self.temporal_ref]
            masked_ref_df = self.mask_dataset(ref_df,
                                              gpi_info)
            if len(masked_ref_df) == 0:
                return matched_n, results, used_data

            df_dict[self.temporal_ref] = masked_ref_df

        matched_n = self.temporal_match_datasets(df_dict)

        for n, k in self.metrics_c:
            n_matched_data = matched_n[(n, k)]
            if len(n_matched_data) == 0:
                continue
            result_names = get_result_names(self.data_manager.ds_dict,
                                            self.temporal_ref,
                                            n=k)
            for data, result_key in self.k_datasets_from(n_matched_data,
                                                         result_names):

                if len(data) == 0:
                    continue

                # at this stage we can drop the column multiindex and just use
                # the dataset name
                if LooseVersion(pd.__version__) < LooseVersion('0.23'):
                    data.columns = data.columns.droplevel(level=1)
                else:
                    data = data.rename(columns=lambda x: x[0])

                if self.scaling is not None:
                    # get scaling index by finding the column in the
                    # DataFrame that belongs to the scaling reference
                    scaling_index = data.columns.tolist().index(self.scaling_ref)
                    try:
                        data = self.scaling.scale(data,
                                                  scaling_index,
                                                  gpi_info)
                    except ValueError:
                        continue
                # Rename the columns to 'ref', 'k1', 'k2', ...
                rename_dict = {}
                f = lambda x: "k{}".format(x) if x > 0 else 'ref'
                for i, r in enumerate(result_key):
                    rename_dict[r[0]] = f(i)
                data.rename(columns=rename_dict, inplace=True)

                if result_key not in results.keys():
                    results[result_key] = []

                metrics_calculator = self.metrics_c[(n, k)]
                used_data[result_key] = data
                metrics = metrics_calculator(data, gpi_info)
                results[result_key].append(metrics)

        return matched_n, results, used_data

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
        result_names = get_result_names(self.masking_dm.ds_dict,
                                        '_reference',
                                        n=2)
        choose_all = pd.DataFrame(index=ref_df.index)
        for data, result in self.k_datasets_from(matched_masking,
                                                 result_names):
            if len(data) == 0:
                continue

            for key in result:
                if key[0] != '_reference':
                    # this is necessary since the boolean datatype might have
                    # been changed to float 1.0 and 0.0 issue with temporal
                    # resampling that is not easily resolved since most
                    # datatypes have no nan representation.
                    choose = pd.Series((data[key] == False), index=data.index)
                    choose = choose.reindex(index=choose_all.index,
                                            fill_value=True)
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
            Contains temporally matched masking data. This dict has only one key
            being a tuple that contains the matched datasets.
        """

        # read only masking datasets and use the already read reference
        masking_df_dict = self.masking_dm.get_other_data(gpi_info[0],
                                                         gpi_info[1],
                                                         gpi_info[2])
        masking_df_dict.update({'_reference': ref_df})
        matched_masking = self.temp_matching(masking_df_dict,
                                             '_reference',
                                             n=2)
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
            matched_data = self.temp_matching(df_dict,
                                              self.temporal_ref,
                                              n=n)

            matched_n[(n, k)] = matched_data

        return matched_n

    def k_datasets_from(self, n_matched_data, result_names):
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

        Yields
        ------
        data: pd.DataFrame
            pandas DataFrame with k columns extracted from the
            temporally matched datasets
        result: tuple
            Tuple describing which datasets and columns are in
            the returned data. ((dataset_name, column_name), (dataset_name2, column_name2))
        """

        for result in result_names:
            data = self.get_data_for_result_tuple(n_matched_data, result)
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
            extracted. ((dataset_name, column_name), (dataset_name2, column_name2))

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
            try:
                data = n_matched_data[dskey]
            except KeyError:
                # if not then temporal matching between two datasets was
                # unsuccessful
                return []
        else:
            # more datasets were temporally matched than are
            # requested now so we select a temporally matched
            # dataset that has the first key in common with the
            # requested one ensuring that it was used as a
            # reference and also has the rest of the requested
            # datasets in the key
            first_match = [
                key for key in n_matched_data if dskey[0] == key[0]]
            found_key = None
            for key in first_match:
                for dsk in dskey[1:]:
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
                    (cell_gpis,
                     cell_lons,
                     cell_lats) = self.data_manager.reference_grid.grid_points_for_cell(cell)
                    jobs.append([cell_gpis, cell_lons, cell_lats])
            else:
                gpis, lons, lats = self.data_manager.reference_grid.get_grid_points()
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
    if 'n' in kwargs:
        n = kwargs['n']
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
