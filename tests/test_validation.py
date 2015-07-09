# coding: utf-8
# Copyright (c) 2015,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology,
#      Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
Tests for the validation framework
Created on Mon Jul  6 12:49:07 2015
'''

import os
import tempfile
import netCDF4 as nc
import numpy as np
import numpy.testing as nptest

import pytesmo.validation_framework.temporal_matchers as temporal_matchers
import pytesmo.validation_framework.metric_calculators as metrics_calculators
from pytesmo.validation_framework.results_manager import netcdf_results_manager

from datetime import datetime

from pytesmo.io.sat.ascat import AscatH25_SSM
from pytesmo.io.ismn.interface import ISMN_Interface
from pytesmo.validation_framework.validation import Validation


class DataPreparation(object):
    """
    Class for preparing the data before validation.
    """

    @staticmethod
    def prep_reference(reference):
        """
        Static method used to prepare the reference dataset (ISMN).

        Parameters
        ----------
        reference : pandas.DataFrame
            ISMN data.

        Returns
        -------
        reference : pandas.DataFrame
            Masked reference.
        """
        return reference

    @staticmethod
    def prep_other(other, other_name,
                   mask_snow=80,
                   mask_frozen=80,
                   mask_ssf=[0, 1]):
        """
        Static method used to prepare the other datasets (ASCAT).

        Parameters
        ----------
        other : pandas.DataFrame
            Containing at least the fields: sm, frozen_prob, snow_prob, ssf.
        other_name : string
            ASCAT.
        mask_snow : int, optional
            If set, all the observations with snow probability > mask_snow
            are removed from the result. Default: 80.
        mask_frozen : int, optional
            If set, all the observations with frozen probability > mask_frozen
            are removed from the result. Default: 80.
        mask_ssf : list, optional
            If set, all the observations with ssf != mask_ssf are removed from
            the result. Default: [0, 1].

        Returns
        -------
        reference : pandas.DataFrame
            Masked reference.
        """
        if other_name == 'ASCAT':

            # mask frozen
            if mask_frozen is not None:
                other = other[other['frozen_prob'] < mask_frozen]

            # mask snow
            if mask_snow is not None:
                other = other[other['snow_prob'] < mask_snow]

            # mask ssf
            if mask_ssf is not None:
                other = other[(other['ssf'] == mask_ssf[0]) |
                              (other['ssf'] == mask_ssf[1])]
        return other


def test_ascat_ismn_validation():
    """
    Test processing framework with some ISMN and ASCAT sample data
    """
    ascat_data_folder = os.path.join(os.path.dirname(__file__), 'test-data',
                                     'sat', 'ascat', 'netcdf', '55R22')

    ascat_grid_folder = os.path.join(os.path.dirname(__file__), 'test-data',
                                     'sat', 'ascat', 'netcdf', 'grid')

    ascat_reader = AscatH25_SSM(ascat_data_folder, ascat_grid_folder)
    ascat_reader.read_bulk = True
    ascat_reader._load_grid_info()

    # Initialize ISMN reader

    ismn_data_folder = os.path.join(os.path.dirname(__file__), 'test-data',
                                    'ismn', 'multinetwork', 'header_values')
    ismn_reader = ISMN_Interface(ismn_data_folder)

    jobs = []

    ids = ismn_reader.get_dataset_ids(
        variable='soil moisture',
        min_depth=0,
        max_depth=0.1)
    for idx in ids:
        metadata = ismn_reader.metadata[idx]
        jobs.append((idx, metadata['longitude'], metadata['latitude']))

    # Create the variable ***save_path*** which is a string representing the
    # path where the results will be saved. **DO NOT CHANGE** the name
    # ***save_path*** because it will be searched during the parallel
    # processing!

    save_path = tempfile.mkdtemp()

    # Create the validation object.

    datasets = {
        'ISMN': {
            'class': ismn_reader, 'columns': [
                'soil moisture'
            ], 'type': 'reference', 'args': [], 'kwargs': {}
        },
        'ASCAT': {
            'class': ascat_reader, 'columns': [
                'sm'
            ], 'type': 'other', 'args': [], 'kwargs': {}, 'grids_compatible':
            False, 'use_lut': False, 'lut_max_dist': 30000
        }
    }

    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

    process = Validation(
        datasets=datasets,
        data_prep=DataPreparation(),
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0,
            reverse=True),
        scaling='lin_cdf_match',
        scale_to_other=True,
        metrics_calculator=metrics_calculators.BasicMetrics(),
        period=period,
        cell_based_jobs=False)

    for job in jobs:
        results = process.calc(job)
        netcdf_results_manager(results, save_path)

    results_fname = os.path.join(
        save_path, 'ISMN.soil moisture_with_ASCAT.sm.nc')

    vars_should = [u'n_obs', u'tau', u'gpi', u'RMSD', u'lon', u'p_tau',
                   u'BIAS', u'p_rho', u'rho', u'lat', u'R', u'p_R']
    n_obs_should = [360, 385, 1644, 1881, 1927, 479, 140, 251]
    rho_should = np.array([0.54618734, 0.71739876, 0.62089276, 0.53246528,
                           0.30299741, 0.69647062, 0.840593, 0.73913699],
                          dtype=np.float32)

    rmsd_should = np.array([11.53626347, 7.54565048, 17.45193481, 21.19371414,
                            14.24668026, 14.27493, 13.173215, 12.59192371],
                           dtype=np.float32)
    with nc.Dataset(results_fname) as results:
        assert sorted(results.variables.keys()) == sorted(vars_should)
        assert sorted(results.variables['n_obs'][:].tolist()) == sorted(
            n_obs_should)
        nptest.assert_allclose(sorted(rho_should),
                               sorted(results.variables['rho'][:]))
        nptest.assert_allclose(sorted(rmsd_should),
                               sorted(results.variables['RMSD'][:]))
