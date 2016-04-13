try:
    from itertools import izip as zip
except ImportError:
    # python 3
    pass

import numpy as np

import pytesmo.scaling as scaling
from pytesmo.validation_framework.data_manager import DataManager


class Validation(object):

    """
    Class for the validation process.

    Parameters
    ----------
    datasets : dict of dicts
        Keys: string, datasets names
        Values: dict, containing the following fields
            'class': object
                Class containing the method read_ts for reading the data.
            'columns': list
                List of columns which will be used in the validation process.
            'type': string
                'reference' or 'other'.
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
    temporal_matcher: function
        function that takes a dict of dataframes and a reference_key.
        It performs the temporal matching on the data and returns a dictionary
        of matched DataFrames that should be evaluated together by the metric calculator.
    metrics_calculators : dict of functions
        The keys of the dict are integers >= 2 and the values are functions that take
        an input DataFrame with the columns 'ref' for the reference and 'n1', 'n2' and
        so on for other datasets. In this way multiple metric calculators can be applied to
        different combinations of n input datasets.
    data_prep: object
        Object that provides the methods prep_reference and prep_other
        which take the pandas.Dataframe provided by the read_ts methods (plus
        other_name for prep_other) and do some data preparation on it before
        temporal matching etc. can be used e.g. for special masking or anomaly
        calculations.
    period : list, optional
        Of type [datetime start, datetime end]. If given then the two input
        datasets will be truncated to start <= dates <= end.
    scaling : string
        If set then the data will be scaled into the reference space using the
        method specified by the string.
    scale_to_other : boolean, optional
        If True the reference dataset is scaled to the other dataset instead
        of the default behavior.
    cell_based_jobs : boolean, optional
        If True then the jobs will be cell based, if false jobs will be tuples
        of (gpi, lon, lat).

    Methods
    -------
    calc(job)
        Takes either a cell or a gpi_info tuple and performs the validation.
    get_processing_jobs()
        Returns processing jobs that this process can understand.
    """

    def __init__(self, datasets, temporal_matcher, metrics_calculators,
                 data_prep=None, period=None, scaling='lin_cdf_match',
                 scale_to_other=False, cell_based_jobs=True):
        """
        Initialize parameters.
        """
        self.data_manager = DataManager(datasets, data_prep, period)

        self.temp_matching = temporal_matcher
        self.metrics_c = metrics_calculators

        self.scaling = scaling
        self.scale_to_index = 0
        if scale_to_other:
            self.scale_to_index = 1

        self.cell_based_jobs = cell_based_jobs

        self.luts = self.data_manager.get_luts()

    def calc(self, job):
        """
        Takes either a cell or a gpi_info tuple and performs the validation.

        Parameters
        ----------
        job : object
            Job of type that self.get_processing_jobs() returns.

        Returns
        -------
        compact_results : dict of dicts
            Keys: result names, combinations of
                  (referenceDataset.column, otherDataset.column)
            Values: dict containing the elements returned by metrics_calculator
        """
        results = {}

        if self.cell_based_jobs:
            process_gpis, process_lons, process_lats = self.data_manager.\
                reference_grid.grid_points_for_cell(job)
        else:
            process_gpis, process_lons, process_lats = [
                job[0]], [job[1]], [job[2]]

        for gpi_info in zip(process_gpis, process_lons, process_lats):
            # if processing is cell based gpi_metainfo is limited to gpi, lon,
            # lat at the moment
            if self.cell_based_jobs:
                gpi_meta = gpi_info
            else:
                gpi_meta = job

            ref_dataframe = self.data_manager.read_reference(gpi_info[0])
            # if no reference data available continue with the next gpi
            if ref_dataframe is None:
                continue

            other_dataframes = {}
            for other_name in self.data_manager.other_name:
                grids_compatible = self.data_manager.datasets[
                    other_name]['grids_compatible']
                if grids_compatible:
                    other_dataframe = self.data_manager.read_other(
                        other_name, gpi_info[0])
                elif self.luts[other_name] is not None:
                    other_gpi = self.luts[other_name][gpi_info[0]]
                    if other_gpi == -1:
                        continue
                    other_dataframe = self.data_manager.read_other(
                        other_name, other_gpi)
                else:
                    other_dataframe = self.data_manager.read_other(
                        other_name, gpi_info[1], gpi_info[2])

                if other_dataframe is not None:
                    other_dataframes[other_name] = other_dataframe

            # if no other data available continue with the next gpi
            if len(other_dataframes) == 0:
                continue

            df_dict = other_dataframes
            df_dict.update({self.data_manager.reference_name: ref_dataframe})

            # compute results for combinations as requested by the metrics
            # calculator dict
            # First temporal match all the combinations
            matched_n = {}
            for n in self.metrics_c:
                matched_data = self.temp_matching(df_dict,
                                                  self.data_manager.reference_name,
                                                  n=n)
                matched_n[n] = matched_data

            for n in self.metrics_c:
                n_matched_data = matched_n[n]
                for result in self.data_manager.get_results_names(n):
                    # find the key into the temporally matched dataset by combining the
                    # dataset parts of the result_names
                    dskey = []
                    rename_dict = {}
                    f = lambda x: "n{}".format(x) if x > 0 else 'ref'
                    for i, r in enumerate(result):
                        dskey.append(r[0])
                        rename_dict[r[0]] = f(i)

                    dskey = tuple(dskey)
                    data = n_matched_data[dskey]

                    # extract only the relevant columns from matched DataFrame
                    data = data[[x for x in result]]

                    # at this stage we can drop the column multiindex and just use
                    # the dataset name
                    data.columns = data.columns.droplevel(level=1)

                    data.rename(columns=rename_dict, inplace=True)

                    if len(data) == 0:
                        continue

                    if self.scaling is not None:
                        try:
                            data = scaling.scale(data,
                                                 method=self.scaling,
                                                 reference_index=0)
                        except ValueError:
                            continue

                    if result not in results.keys():
                        results[result] = []

                    metrics_calculator = self.metrics_c[n]
                    results[result].append(metrics_calculator(data, gpi_meta))

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

    def get_processing_jobs(self):
        """
        Returns processing jobs that this process can understand.

        Returns
        -------
        jobs : list
            List of cells or gpis to process.
        """
        if self.data_manager.reference_grid is not None:
            if self.cell_based_jobs:
                return self.data_manager.reference_grid.get_cells()
            else:
                return zip(self.data_manager.reference_grid.get_grid_points())
        else:
            return []
