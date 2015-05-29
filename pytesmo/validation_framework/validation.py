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
Created on Aug 30, 2013

@author: Christoph Paulik Christoph.Paulik@geo.tuwien.ac.at
'''

from itertools import izip
import itertools
import numpy as np

import pytesmo.scaling as scaling
import general.io.compr_pickle as pickle

from pytesmo.validation_framework.data_manager import DataManager


class Validation(object):

    """
    this class is the basis for comparing datasets that both have
    have a class based reader that follow some guidelines.
    *  implement an read_ts function that takes either gpi
       or lon,lat and returns a pandas.DataFrame with datetimeindex
    *  The readers of the datasets should also be initialized with the read_bulk keyword
       to make reading of cell based datasets or netCDF files faster

    Parameters
    ----------
    datasets : dict of dicts
        dictonary containing the following fields for each dataset
            'class': object
                class based reader
            'columns': list
                list of columns which will be used in the validation process
            'type': string
                'reference' or 'other'
            'args': list, optional
                args for reading the data
            'kwargs': dict, optional
                kwargs for reading the data
            'grids_compatible': boolean, optional
                if set to True the grid point index is used directly
                when reading other, if False then lon, lat is used and a
                nearest neighbour search is necessary
            'use_lut': boolean, optional
                if set to True the grid point index (obtained from a
                calculated lut between reference and other) is used when
                reading other, if False then lon, lat is used and a
                nearest neighbour search is necessary
            'lut_max_dist': float, optional
                maximum allowed distance in meters for the lut calculation
    temporal_matcher: object
        class instance that has a match method that takes a
        reference and a other DataFrame. It's match method
        should return a DataFrame with the index of the reference DataFrame
        and all columns of both DataFrames
    metrics_calculator : object
        class that has the attribute result_template and a calc_metrics method
        that takes a pandas.DataFrame with 2 columns named 'ref' and 'other'
        and returns a filled result_template
    data_prep: object
        object that provides the methods prep_reference and prep_other
        which take the pandas.Dataframe provided by the read_ts methods
        and do some data preparation on it before temporal matching etc.
        can be used e.g. for special masking or anomaly calculations
    period : list, optional
        of type [datetime start,datetime end] if given then the two input
        datasets will be truncated to start <= dates <= end
    scaling : string
        if set then the data will be scaled into the reference space using the
        method specified by the string
    scale_to_other : boolean, optional
        if True the reference dataset is scaled to the other dataset instead
        of the default behavior
    cell_based_jobs : boolean, optional
        if True then the jobs will be cell based, if false jobs will be tuples
        of (gpi, lon, lat)
    """

    def __init__(self, datasets, temporal_matcher, metrics_calculator,
                 data_prep=None, period=None, scaling='lin_cdf_match',
                 scale_to_other=False, cell_based_jobs=True):

        self.data_manager = DataManager(datasets, data_prep, period)

        self.temp_matching = temporal_matcher.match
        self.calc_metrics = metrics_calculator.calc_metrics

        self.scaling = scaling
        self.scale_to_index = 0
        if scale_to_other:
            self.scale_to_index = 1

        self.cell_based_jobs = cell_based_jobs

    def calc(self, job):
        """
        takes either a cell or a gpi_info tuple and
        matches and compares the 2 datasets according to
        the objects given to self.metrics_calculator and
        self.temporal_matcher

        Parameters
        ----------
        job : object
            job as understood by the process, is of type that self.get_processing_jobs()
            returns

        Returns
        -------
        results : dict
            dictionary with keys self.result_names
            each dict element is a list of elements returned by
            self.calc_metrics
        """
        result_names = self.data_manager.get_result_names()
        results = {}

        if self.cell_based_jobs:
            process_gpis, process_lons, process_lats = self.data_manager.\
                reference_grid.grid_points_for_cell(job)
        else:
            process_gpis, process_lons, process_lats = [
                job[0]], [job[1]], [job[2]]

        i = 0
        for gpi_info in izip(process_gpis, process_lons, process_lats):
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
                grids_compatible = self.data_manager.datasets[other_name]['grids_compatible']
                use_lut = self.data_manager.use_lut(other_name)
                if grids_compatible:
                    other_dataframe = self.data_manager.read_other(
                        other_name, gpi_info[0])
                elif use_lut is not None:
                    other_gpi = use_lut[gpi_info[0]]
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

            joined_data = {}
            for other in other_dataframes.keys():
                joined = self.temp_matching(ref_dataframe,
                                            other_dataframes[other])

                if len(joined) != 0:
                    joined_data[other] = joined

            if len(joined_data) == 0:
                continue

            i += 1

            # compute results for each requested columns
            for result in result_names:
                ref_col = result[0].split('.')[1]
                other_col = result[1].split('.')[1]
                other_name = result[1].split('.')[0]

                try:
                    data = joined_data[other_name][[ref_col, other_col]].dropna()
                except KeyError:
                    continue

                data.rename(
                    columns={ref_col: 'ref', other_col: 'other'}, inplace=True)

                if len(data) == 0:
                    continue

                if self.scaling is not None:
                    try:
                        data = scaling.scale(
                            data, method=self.scaling, reference_index=self.scale_to_index)
                    except ValueError:
                        continue

                if result not in results.keys():
                    results[result] = []

                results[result].append(self.calc_metrics(data, gpi_meta))

            if i == 3:
                break

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
        returns processing jobs that this process can understand

        Returns
        -------
        jobs : list
            list of cells or gpis to process
        """
        if self.cell_based_jobs:
            return self.data_manager.reference_grid.get_cells()
        else:
            return zip(self.data_manager.reference_grid.get_grid_points())
