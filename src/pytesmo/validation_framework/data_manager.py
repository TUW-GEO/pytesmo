# Copyright (c) 2016,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without
#     specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT
# OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import itertools
import numpy as np

from pytesmo.validation_framework.upscaling import Upscaling, MixinReadTs


class DataManager(MixinReadTs):

    """
    Class to handle the data management.

    Parameters
    ----------
    datasets : dict of dicts
        :Keys : string, datasets names
        :Values : dict, containing the following fields

            'class' : object
                Class containing the method `read` for reading the data.
            'columns' : list
                List of columns which will be used in the validation process.
            'args' : list, optional
                Args that are passed to the reading function.
            'kwargs' : dict, optional
                Kwargs that are passed to the reading function.
            'grids_compatible' : boolean, optional
                If set to True the grid point index is used directly when
                reading other, if False then lon, lat is used and a nearest
                neighbour search is necessary.
                default: False
            'use_lut' : boolean, optional
                If set to True the grid point index (obtained from a
                calculated lut between reference and other) is used when
                reading other, if False then lon, lat is used and a
                nearest neighbour search is necessary.
                default: False
            'max_dist' : float, optional
                Maximum allowed distance in meters for the lut calculation.
                default: None
    ref_name : string
        Name of the reference dataset. The reference dataset is used as spatial
        reference, i.e. all other dataset will be interpolated to the locations
        of the reference dataset.
    period : list, optional
        Of type [datetime start, datetime end]. If given then the two input
        datasets will be truncated to start <= dates <= end.
    read_ts_names : string or dict of strings, optional
        if another method name than 'read' should be used for reading the data
        then it can be specified here. If it is a dict then specify a
        function name for each dataset.
    upscale_parms : dict, optional. Default is None.
        dictionary with parameters for the upscaling methods. Keys:
            * 'upscaling_method': method for upscaling
            * 'temporal_stability': bool for using temporal stability
            * 'upscaling_lut': dict of shape
                {'other_name':{ref gpi: [other gpis]}}

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

    def __init__(
        self,
        datasets,
        ref_name,
        period=None,
        read_ts_names="read",
        upscale_parms=None,
    ):
        self.datasets = datasets
        self._add_default_values()
        self.reference_name = ref_name
        self.upscale_parms = upscale_parms

        self.other_name = []
        for dataset in datasets.keys():
            if dataset != ref_name:
                self.other_name.append(dataset)
                if "use_lut" not in self.datasets[dataset]:
                    self.datasets[dataset]["use_lut"] = False

        try:
            self.reference_grid = self.datasets[self.reference_name][
                "class"
            ].grid
        except AttributeError:
            self.reference_grid = None

        self.period = period

        # get reading functions
        if type(read_ts_names) is dict:
            self.read_ts_names = read_ts_names
        else:
            d = {}
            for dataset in datasets:
                d[dataset] = read_ts_names
            self.read_ts_names = d

        # match points in space
        if upscale_parms:
            # initialize class that performs upscaling operations
            others_class = {}
            for other in self.other_name:
                others_class[other] = self.datasets[other]["class"]
            self.luts = Upscaling(
                ref_class=datasets[self.reference_name]["class"],
                others_class=others_class,
                upscaling_lut=upscale_parms["upscaling_lut"],
                manager_parms=self.__dict__,
            )
        else:
            # combine ref to NNs only
            self.luts = self.get_luts()

    def _add_default_values(self):
        """
        Add defaults for args, kwargs, grids_compatible, use_lut and
        max_dist to dataset dictionary.
        """
        defaults = {
            "use_lut": False,
            "args": [],
            "kwargs": {},
            "grids_compatible": False,
            "max_dist": np.inf,
        }
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
            if self.datasets[other_name]["use_lut"]:
                luts[other_name] = self.reference_grid.calc_lut(
                    self.datasets[other_name]["class"].grid,
                    max_dist=self.datasets[other_name]["max_dist"],
                )
            else:
                luts[other_name] = None

        return luts

    @property
    def ds_dict(self):
        ds_dict = {}
        for dataset in self.datasets.keys():
            ds_dict[dataset] = self.datasets[dataset]["columns"]
        return ds_dict

    def get_results_names(self, n=2):

        return get_result_names(self.ds_dict, self.reference_name, n=n)

    def read_reference(self, *args):
        """
        Function to read and prepare the reference dataset.

        Calls read of the dataset.
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
        Function to read and prepare non-reference datasets.

        Calls read of the dataset.

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
            grids_compatible = self.datasets[other_name]["grids_compatible"]
            if grids_compatible:
                other_dataframe = self.read_other(other_name, gpi)
            elif isinstance(self.luts, Upscaling):
                other_dataframe = self.luts.get_upscaled_ts(
                    gpi=gpi, other_name=other_name, **self.upscale_parms
                )
            elif self.luts[other_name] is not None:
                other_gpi = self.luts[other_name][gpi]
                if other_gpi == -1:
                    continue
                other_dataframe = self.read_other(other_name, other_gpi)
            else:
                # other_dataframe = self.read_other(other_name, lon, lat)
                max_dist = self.datasets[other_name].get("max_dist", np.inf)
                grid = self.datasets[other_name]["class"].grid
                other_gpi, dist = grid.find_nearest_gpi(
                    lon, lat, max_dist=max_dist
                )
                if np.isinf(dist):
                    # no other point found in range, currently this is handled
                    # by returning None
                    other_dataframe = None
                else:
                    other_dataframe = self.read_other(other_name, other_gpi)

            if other_dataframe is not None:
                other_dataframes[other_name] = other_dataframe
        return other_dataframes


def flatten(seq):
    eltl = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):
                eltl.append(elt2)
        else:
            eltl.append(elt)
    return eltl


def get_result_combinations(ds_dict, n=2):
    """
    Get all possible combinations dataset columns

    Parameters
    ----------
    ds_dict: dict
       Dict of lists containing the dataset names as keys and a list of the
       columns to read from the dataset as values.
    n: int
        Number of datasets for combine with each other.
        If n=2 always two datasets will be combined into one result.
        If n=3 always three datasets will be combined into one results and
        so on.
        n has to be <= the number of total datasets.

    Returns
    -------
    results_names : list of tuples
        Containing all possible combinations of
        (dataset_x.column, dataset_y.column)
        for all datasets in ds_dict
    """
    combis = []
    for key in ds_dict:
        combis.extend(get_result_names(ds_dict, key, n=n))
    return sorted(list(set(combis)))


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
        If n=3 always three datasets will be combined into one results and
        so on.
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

    for comb in itertools.product(
        ref_columns, itertools.combinations(other_columns, n - 1)
    ):
        results_names.append(comb)

    # flatten to one level and remove those that do not have n unique
    # datasets
    results_names = flatten(results_names)

    # iterate in chunks of n*2 over the list
    result_combos = []
    for chunk in [
        results_names[pos: pos + n * 2]
        for pos in range(0, len(results_names), n * 2)
    ]:
        combo = []
        datasets = chunk[::2]
        columns = chunk[1::2]
        # if datasets are compared to themselves then don't include the
        # combination
        if len(set(datasets)) != n:
            continue
        for dataset, column in zip(datasets, columns):
            combo.append((dataset, column))
        result_combos.append(tuple(sorted(combo)))

    return result_combos
