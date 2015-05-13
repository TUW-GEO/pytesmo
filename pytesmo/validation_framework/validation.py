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
    datasets : dict
        dictonary of dataset class instances
    reference_name : string
        name of the key in datasets of the initalized reader object of reference dataset whose grid will
        be used and compared against the nearest grid point of other
    other_name : string
        name of key in datasets of the initalized reader object of other
    reference_column : string, or list of strings
        column name of the pandas.Dataframe returned by the reader of
        the reference object to be used in the comparison.
        if multiple_ref is True this has to be a list of columns
        that are compared to other_column
    other_column : string
        column name of the pandas.Dataframe returned by the reader of
        the other object to be used in the comparison.
    metrics_calculator : object
        class that has the attribute result_template and a calc_metrics method
        that takes a pandas.DataFrame with 2 columns named 'ref' and 'other'
        and returns a filled result_template
    grids_compatible : boolean, optional
        if set to True the grid point index is used directly
        when reading other, if False then lon, lat is used and a
        nearest neighbour search is necessary
    data_prep: object
        object that provides the methods prep_reference and prep_other
        which take the pandas.Dataframe provided by the read_ts methods
        and do some data preparation on it before temporal matching etc.
        can be used e.g. for special masking or anomaly calculations
    temporal_matcher: object
        class instance that has a match method that takes a
        reference and a other DataFrame. It's match method
        should return a DataFrame with the index of the reference DataFrame
        and all columns of both DataFrames
    scaling : string
        if set then the data will be scaled into the reference space using the
        method specified by the string
    scale_to_other : boolean, optional
        if True the reference dataset is scaled to the other dataset instead
        of the default behavior
    result_names : list, optional
        if given the result dictionary will have the keys in this list, if not given
        the result dictionary will have keys built out of reference_column and other_column
    lut : numpy.array
        if given this look up table will be used instead of nearest neighbour search
    period : list, optional
        of type [datetime start,datetime end] if given then the two input datasets will be truncated to
        start <= dates <= end
    cell_based_jobs : boolean, optional
        if True then the jobs will be cell based, if false jobs will be tuples of
        (gpi, lon, lat)
    """

    def __init__(self, datasets, reference_name, other_name,
                 reference_column=None, other_column=None,
                 reference_args=[], other_args=[],
                 reference_kwargs={}, other_kwargs={},
                 grids_compatible=False, data_prep=None,
                 temporal_matcher=None, scaling=None,
                 scale_to_other=False, metrics_calculator=None,
                 result_names=None, use_lut=True, lut_max_dist=30000,
                 period=None, cell_based_jobs=True, save_path=None,
                 result_man=None):

        self.reference = datasets[reference_name]
        self.other = datasets[other_name]
        self.reference_column = reference_column
        if type(self.reference_column) is not list:
            self.reference_column = [self.reference_column]
        self.other_column = other_column
        if type(self.other_column) is not list:
            self.other_column = [self.other_column]
        self.reference_args = reference_args
        self.other_args = other_args
        self.reference_kwargs = reference_kwargs
        self.other_kwargs = other_kwargs

        self.cell_based_jobs = cell_based_jobs

        self.grids_compatible = grids_compatible
        self.temp_matching = temporal_matcher.match

        self.use_data_prep = False
        if data_prep is not None:
            self.data_prep = data_prep
            self.use_data_prep = True

        self.scaling = scaling
        self.use_scaling = False
        if self.scaling is not None:
            self.use_scaling = True

        self.scale_to_index = 0
        if scale_to_other:
            self.scale_to_index = 1

        self.metrics_calculator = metrics_calculator
        self.calc_metrics = self.metrics_calculator.calc_metrics
        self.result_names = result_names
        self.results = {}
        self.period = period
        self.use_period = False
        if self.period is not None:
            self.use_period = True
        self.use_lut = use_lut
        if self.use_lut:
            self.lut = self.reference.grid.calc_lut(
                self.other.grid, max_dist=lut_max_dist)
        else:
            self.lut = None

        self.save_path = save_path

        if result_man is not None:
            self.result_man = result_man(metrics_calculator=self.metrics_calculator,
                                         grid=self.reference.grid)
        else:
            self.result_man = result_man

        if self.result_names is None:
            self.result_names = []
            for columns in itertools.product(reference_column, other_column):
                self.result_names.append('_'.join(columns))

        if len(self.result_names) != len(self.reference_column) * len(self.other_column):
            raise ValueError("Wrong number of result names. There should be a result_name for every"
                             " combination of reference_column and other_column")

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
        results = {}
        for result_name in self.result_names:
            results[result_name] = []

        if self.cell_based_jobs:
            process_gpis, process_lons, process_lats = self.reference.grid.grid_points_for_cell(
                job)
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

            # print self.name, gpi_info[0]
            # if i==50: return results
            try:
                ref_dataframe = self.read_reference(gpi_info[0])
            except IOError:
                continue

            if self.use_period:
                ref_dataframe = ref_dataframe[self.period[0]:self.period[1]]

            if len(ref_dataframe) == 0:
                continue

            if self.use_data_prep:
                ref_dataframe = self.data_prep.prep_reference(ref_dataframe)

            try:
                if self.grids_compatible:
                    other_data = self.read_other(gpi_info[0])
                elif self.use_lut:
                    other_gpi = self.lut[gpi_info[0]]
                    if other_gpi == -1:
                        continue
                    other_data = self.read_other(other_gpi)
                else:
                    other_data = self.read_other(gpi_info[1], gpi_info[2])
            except IOError:
                continue

            # if no data available continue with the next gpi
            if len(other_data) == 0:
                continue

            if self.use_period:
                other_data = other_data[self.period[0]:self.period[1]]

            # if no data available continue with the next gpi
            if len(other_data) == 0:
                continue

            if self.use_data_prep:
                other_data = self.data_prep.prep_other(other_data)

            joined_data = self.temp_matching(ref_dataframe, other_data)

            if len(joined_data) == 0:
                continue

            i += 1
            if i == 20:
                return results

            # compute results for each requested column in DataFrame
            for j, columns in enumerate(itertools.product(self.reference_column, self.other_column)):

                data = joined_data[list(columns)].dropna()

                data.rename(
                    columns={columns[0]: 'ref', columns[1]: 'other'}, inplace=True)

                if len(data) == 0:
                    continue

                if self.use_scaling:
                    try:
                        data = scaling.scale(
                            data, method=self.scaling, reference_index=self.scale_to_index)
                    except ValueError:
                        continue

                results[self.result_names[j]].append(
                    self.calc_metrics(data, gpi_meta))

        return results

    def read_reference(self, *args):
        """
        function to read the reference dataset
        can be overridden if another function than read_ts is used
        """
        args = list(args)
        args.extend(self.reference_args)
        return self.reference.read_ts(*args, **self.reference_kwargs)

    def read_other(self, *args):
        """
        function to read the other dataset
        can be overridden if another function than read_ts is used
        """
        args = list(args)
        args.extend(self.other_args)
        return self.other.read_ts(*args, **self.other_kwargs)

    def get_processing_jobs(self):
        """
        returns processing jobs that this process can understand

        Returns
        -------
        jobs : list
            list of cells or gpis to process
        """
        if self.cell_based_jobs:
            return self.reference.grid.get_cells()
        else:
            return zip(self.reference.grid.get_grid_points())

    def manage_results(self, results):
        """
        manages the results after processing, this could mean saving or returning them
        or just removing success or error messages from the result queue

        Parameters
        ----------
        results : list
            results in list as returned by self.calc
        """
        result_dict = {}
        for result_name in self.result_names:
            result_dict[result_name] = []

        for result in results:
            for key in result:
                result_dict[key].extend(result[key])

        for key in result_dict:
            result_dict[key] = np.squeeze(np.dstack(result_dict[key]))

        if self.save_path is not None:

            if self.result_man is None:
                pickle.pickle(self.save_path, result_dict)
            else:
                self.result_man.manage_results(result_dict, self.save_path)

        else:
            return result_dict
