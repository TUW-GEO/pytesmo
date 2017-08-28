# Copyright (c) 2015, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department
#     of Geodesy and Geoinformation nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

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

import itertools
import warnings

import pandas as pd

from pygeobase.object_base import TS


class DataManager(object):

    """
    Class to handle the data management.

    Parameters
    ----------
    datasets : dict of dicts
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
                default: False
            'use_lut': boolean, optional
                If set to True the grid point index (obtained from a
                calculated lut between reference and other) is used when
                reading other, if False then lon, lat is used and a
                nearest neighbour search is necessary.
                default: False
            'lut_max_dist': float, optional
                Maximum allowed distance in meters for the lut calculation.
                default: None
    ref_name: string
        Name of the reference dataset
    period : list, optional
        Of type [datetime start, datetime end]. If given then the two input
        datasets will be truncated to start <= dates <= end.
    read_ts_names: string or dict of strings, optional
        if another method name than 'read_ts' should be used for reading the data
        then it can be specified here. If it is a dict then specify a
        function name for each dataset.

    Methods
    -------
    use_lut(other_name)
        Returns lut between reference and other if use_lut for other dataset
        was set to True.
    get_result_names()
        Return results names based on reference and others names.
    read_reference(*args)
        Function to read and prepare the reference dataset.
    read_other(other_name, *args)
        Function to read and prepare the other datasets.
    """

    def __init__(self, datasets, ref_name,
                 period=None,
                 read_ts_names='read_ts'):
        """
        Initialize parameters.
        """
        self.datasets = datasets
        self._add_default_values()
        self.reference_name = ref_name

        self.other_name = []
        for dataset in datasets.keys():
            if dataset != ref_name:
                self.other_name.append(dataset)
                if 'use_lut' not in self.datasets[dataset]:
                    self.datasets[dataset]['use_lut'] = False

        try:
            self.reference_grid = self.datasets[
                self.reference_name]['class'].grid
        except AttributeError:
            self.reference_grid = None

        self.period = period
        self.luts = self.get_luts()
        if type(read_ts_names) is dict:
            self.read_ts_names = read_ts_names
        else:
            d = {}
            for dataset in datasets:
                d[dataset] = read_ts_names
            self.read_ts_names = d

    def _add_default_values(self):
        """
        Add defaults for args, kwargs, grids_compatible, use_lut and
        lut_max_dist to dataset dictionary.
        """
        defaults = {'use_lut': False,
                    'args': [],
                    'kwargs': {},
                    'grids_compatible': False,
                    'lut_max_dist': None}
        for dataset in self.datasets.keys():
            new_defaults = dict(defaults)
            new_defaults.update(self.datasets[dataset])
            self.datasets[dataset] = new_defaults

    def get_luts(self):
        """
        Returns luts between reference and others if use_lut for other datasets
        was set to True.

        Returns
        -------
        luts : dict
            Keys: other datasets names
            Values: lut between reference and other, or None
        """
        luts = {}
        for other_name in self.other_name:
            if self.datasets[other_name]['use_lut']:
                luts[other_name] = self.reference_grid.calc_lut(
                    self.datasets[other_name]['class'].grid,
                    max_dist=self.datasets[other_name]['lut_max_dist'])
            else:
                luts[other_name] = None

        return luts

    @property
    def ds_dict(self):
        ds_dict = {}
        for dataset in self.datasets.keys():
            ds_dict[dataset] = self.datasets[dataset]['columns']
        return ds_dict

    def get_results_names(self, n=2):

        return get_result_names(self.ds_dict, self.reference_name, n=n)

    def read_reference(self, *args):
        """
        Function to read and prepare the reference dataset.

        Calls read_ts of the dataset.
        Takes either 1 (gpi) or 2 (lon, lat) arguments.

        Parameters
        ----------
        gpi : int
            Grid point index
        lon : float
            Longitude of point
        lat : float
            Latitude of point

        Returns
        -------
        ref_df : pandas.DataFrame or None
            Reference dataframe.
        """
        return self.read_ds(self.reference_name, *args)

    def read_other(self, name, *args):
        """
        Function to read and prepare a datasets.

        Calls read_ts of the dataset.

        Takes either 1 (gpi) or 2 (lon, lat) arguments.

        Parameters
        ----------
        name : string
            Name of the other dataset.
        gpi : int
            Grid point index
        lon : float
            Longitude of point
        lat : float
            Latitude of point

        Returns
        -------
        data_df : pandas.DataFrame or None
            Data DataFrame.
        """
        return self.read_ds(name, *args)

    def read_ds(self, name, *args):
        """
        Function to read and prepare a datasets.

        Calls read_ts of the dataset.

        Takes either 1 (gpi) or 2 (lon, lat) arguments.

        Parameters
        ----------
        name : string
            Name of the other dataset.
        gpi : int
            Grid point index
        lon : float
            Longitude of point
        lat : float
            Latitude of point

        Returns
        -------
        data_df : pandas.DataFrame or None
            Data DataFrame.

        """
        ds = self.datasets[name]
        args = list(args)
        args.extend(ds['args'])

        try:
            func = getattr(ds['class'], self.read_ts_names[name])
            data_df = func(*args, **ds['kwargs'])
            if type(data_df) is TS or issubclass(type(data_df), TS):
                data_df = data_df.data
        except IOError:
            warnings.warn(
                "IOError while reading dataset {} with args {:}".format(name,
                                                                        args))
            return None
        except RuntimeError as e:
            if e.args[0] == "No such file or directory":
                warnings.warn(
                    "IOError while reading dataset {} with args {:}".format(name,
                                                                            args))
                return None
            else:
                raise e

        if len(data_df) == 0:
            warnings.warn("No data for dataset {}".format(name))
            return None

        if isinstance(data_df, pd.DataFrame) == False:
            warnings.warn("Data is not a DataFrame {:}".format(args))
            return None

        if self.period is not None:
            # here we use the isoformat since pandas slice behavior is
            # different when using datetime objects.
            data_df = data_df[
                self.period[0].isoformat():self.period[1].isoformat()]

        if len(data_df) == 0:
            warnings.warn("No data for dataset {} with arguments {:}".format(name,
                                                                             args))
            return None

        else:
            return data_df

    def get_data(self, gpi, lon, lat):
        """
        Get all the data from this manager for a certain
        grid point, longitude, latidude combination.

        Parameters
        ----------
        gpi: int
            grid point indices
        lon: float
            grid point longitude
        lat: type
            grid point latitude

        Returns
        -------
        df_dict: dict of pandas.DataFrames
            Dictionary with dataset names as the key and
            pandas.DataFrames containing the data for the point
            as values.
            The dict will be empty if no data is available.
        """
        df_dict = {}

        ref_dataframe = self.read_reference(gpi)
        # if no reference data available continue with the next gpi
        if ref_dataframe is None:
            return df_dict

        other_dataframes = self.get_other_data(gpi, lon, lat)
        # if no other data available continue with the next gpi
        if len(other_dataframes) == 0:
            return df_dict

        df_dict = other_dataframes
        df_dict.update({self.reference_name: ref_dataframe})

        return df_dict

    def get_other_data(self, gpi, lon, lat):
        """
        Get all the data for non reference datasets
        from this manager for a certain
        grid point, longitude, latidude combination.

        Parameters
        ----------
        gpi: int
            grid point indices
        lon: float
            grid point longitude
        lat: type
            grid point latitude

        Returns
        -------
        other_dataframes: dict of pandas.DataFrames
            Dictionary with dataset names as the key and
            pandas.DataFrames containing the data for the point
            as values.
            The dict will be empty if no data is available.
        """

        other_dataframes = {}
        for other_name in self.other_name:
            grids_compatible = self.datasets[
                other_name]['grids_compatible']
            if grids_compatible:
                other_dataframe = self.read_other(
                    other_name, gpi)
            elif self.luts[other_name] is not None:
                other_gpi = self.luts[other_name][gpi]
                if other_gpi == -1:
                    continue
                other_dataframe = self.read_other(
                    other_name, other_gpi)
            else:
                other_dataframe = self.read_other(
                    other_name, lon, lat)

            if other_dataframe is not None:
                other_dataframes[other_name] = other_dataframe
        return other_dataframes


def flatten(seq):
    l = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):
                l.append(elt2)
        else:
            l.append(elt)
    return l


def get_result_names(ds_dict, refkey, n=2):
    """
    Return result names based on all possible combinations based on a
    reference dataset.

    Parameters
    ----------
    ds_dict: dict
       Dict of lists containing the dataset names as keys and a list of the
       columns to read from the dataset as values.
    refkey: string
       dataset name to use as a reference
    n: int
        Number of datasets for combine with each other.
        If n=2 always two datasets will be combined into one result.
        If n=3 always three datasets will be combined into one results and so on.
        n has to be <= the number of total datasets.

    Returns
    -------
    results_names : list of tuples
        Containing all combinations of
        (referenceDataset.column, otherDataset.column)
    """
    results_names = []

    ref_columns = []
    for column in ds_dict[refkey]:
        ref_columns.append((refkey, column))

    other_columns = []
    other_names = list(ds_dict)
    del other_names[other_names.index(refkey)]
    for other in sorted(other_names):
        for column in ds_dict[other]:
            other_columns.append((other, column))

    for comb in itertools.product(ref_columns,
                                  itertools.combinations(other_columns, n - 1)):
        results_names.append(comb)

    # flatten to one level and remove those that do not have n unique
    # datasets
    results_names = flatten(results_names)

    # iterate in chunks of n*2 over the list
    result_combos = []
    for chunk in [results_names[pos:pos + n * 2] for pos in range(0, len(results_names), n * 2)]:
        combo = []
        datasets = chunk[::2]
        columns = chunk[1::2]
        # if datasets are compared to themselves then don't include the
        # combination
        if len(set(datasets)) != n:
            continue
        for dataset, column in zip(datasets, columns):
            combo.append((dataset, column))
        result_combos.append(tuple(combo))

    return result_combos
