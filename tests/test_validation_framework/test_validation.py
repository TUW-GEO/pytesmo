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
"""
Tests for the validation framework
Created on Mon Jul  6 12:49:07 2015
"""

from datetime import datetime
import netCDF4 as nc
import numpy as np
import numpy.testing as nptest
import os
import pytest
import tempfile
import warnings
import pandas as pd

import pygeogrids.grids as grids
from pygeobase.io_base import GriddedTsBase

import pytesmo.validation_framework.temporal_matchers as temporal_matchers
import pytesmo.validation_framework.metric_calculators as metrics_calculators
from pytesmo.validation_framework.results_manager import netcdf_results_manager
from pytesmo.validation_framework.data_manager import DataManager
from pytesmo.validation_framework.results_manager import PointDataResults
from pytesmo.validation_framework.validation import Validation
from pytesmo.validation_framework.validation import args_to_iterable

from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics
)
from pytesmo.validation_framework.temporal_matchers import (
    make_combined_temporal_matcher
)

from ismn.interface import ISMN_Interface

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from ascat.read_native.cdr import AscatGriddedNcTs

if __name__ != "__main__":
    from tests.test_validation_framework.test_datasets import (
        setup_TestDatasets,
        setup_two_without_overlap,
        setup_three_with_two_overlapping,
        MaskingTestDataset,
    )


@pytest.fixture
def ascat_reader():
    ascat_data_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test-data",
        "sat",
        "ascat",
        "netcdf",
        "55R22",
    )

    ascat_grid_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test-data",
        "sat",
        "ascat",
        "netcdf",
        "grid",
    )
    grid_fname = os.path.join(ascat_grid_folder, "TUW_WARP5_grid_info_2_1.nc")

    static_layers_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test-data",
        "sat",
        "h_saf",
        "static_layer",
    )

    ascat_reader = AscatGriddedNcTs(
        ascat_data_folder,
        "TUW_METOP_ASCAT_WARP55R22_{:04d}",
        grid_filename=grid_fname,
        static_layer_path=static_layers_folder,
    )
    ascat_reader.read_bulk = True
    return ascat_reader


@pytest.fixture
def ismn_reader():
    # Initialize ISMN reader

    ismn_data_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test-data",
        "ismn",
        "multinetwork",
        "header_values",
    )
    ismn_reader = ISMN_Interface(ismn_data_folder)

    return ismn_reader


def check_results(
        filename: str,
        target_vars: dict,
        variables: list = None,
):
    """
    Check that standard vars are present and that nobs, rho and rmsd match the given values. Vars can be optionally
    specified
    """
    if variables is not None:
        vars_should = variables
    else:
        vars_should = [u"n_obs", u"tau", u"gpi", u"RMSD", u"lon", u"p_tau", u"BIAS", u"p_rho",
                       u"rho", u"lat", u"R", u"p_R", u"time", u"idx", u"_row_size"]

    with nc.Dataset(filename, mode="r") as results:
        vars = results.variables.keys()
        assert sorted(vars) == sorted(vars_should)

        for varname, should_values in target_vars.items():
            values = results.variables[varname][:]
            if varname == "n_obs":
                values = list(values)
                assert sorted(values) == sorted(should_values)
            elif varname == "network":
                nptest.assert_equal(sorted(values), sorted(should_values))
            else:
                nptest.assert_allclose(sorted(values), sorted(should_values), rtol=1e-4)


# @pytest.mark.slow
# @pytest.mark.full_framework
def test_ascat_ismn_validation(ascat_reader, ismn_reader):
    """
    Test processing framework with some ISMN and ASCAT sample data
    """
    jobs = []

    ids = ismn_reader.get_dataset_ids(
        variable="soil moisture", min_depth=0, max_depth=0.1
    )
    for idx in ids:
        metadata = ismn_reader.metadata[idx]
        jobs.append((idx, metadata["longitude"], metadata["latitude"]))

    # Create the variable ***save_path*** which is a string representing the
    # path where the results will be saved. **DO NOT CHANGE** the name
    # ***save_path*** because it will be searched during the parallel
    # processing!

    save_path = tempfile.mkdtemp()

    # Create the validation object.

    datasets = {
        "ISMN": {"class": ismn_reader, "columns": ["soil moisture"]},
        "ASCAT": {
            "class": ascat_reader,
            "columns": ["sm"],
            "kwargs": {
                "mask_frozen_prob": 80,
                "mask_snow_prob": 80,
                "mask_ssf": True,
            },
        },
    }

    read_ts_names = {"ASCAT": "read", "ISMN": "read"}
    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

    datasets = DataManager(
        datasets, "ISMN", period, read_ts_names=read_ts_names
    )

    process = Validation(
        datasets,
        "ISMN",
        temporal_ref="ASCAT",
        scaling="lin_cdf_match",
        scaling_ref="ASCAT",
        metrics_calculators={
            (2, 2): metrics_calculators.BasicMetrics(
                other_name="k1"
            ).calc_metrics
        },
        period=period,
    )

    for job in jobs:
        results = process.calc(*job)
        netcdf_results_manager(results, save_path)

    results_fname = os.path.join(
        save_path, "ASCAT.sm_with_ISMN.soil moisture.nc"
    )
    # targets
    target_vars = {
        "n_obs": [
            357,
            384,
            1646,
            1875,
            1915,
            467,
            141,
            251
        ],
        "rho": np.array(
            [0.53934574,
             0.7002289,
             0.62200236,
             0.53647155,
             0.30413666,
             0.6740655,
             0.8418981,
             0.74206454
             ], dtype=np.float32),
        "RMSD": np.array(
            [11.583476,
             7.729667,
             17.441547,
             21.125721,
             14.31557,
             14.187225,
             13.0622425,
             12.903898
             ], dtype=np.float32)}

    check_results(
        filename=results_fname,
        target_vars=target_vars,
    )


# @pytest.mark.slow
# @pytest.mark.full_framework
def test_ascat_ismn_validation_metadata(ascat_reader, ismn_reader):
    """
    Test processing framework with some ISMN and ASCAT sample data
    """
    jobs = []

    ids = ismn_reader.get_dataset_ids(
        variable="soil moisture", min_depth=0, max_depth=0.1
    )

    metadata_dict_template = {
        "network": np.array(["None"], dtype="U256"),
        "station": np.array(["None"], dtype="U256"),
        "landcover": np.float32([np.nan]),
        "climate": np.array(["None"], dtype="U4"),
    }

    for idx in ids:
        metadata = ismn_reader.metadata[idx]
        metadata_dict = [
            {
                "network": metadata["network"],
                "station": metadata["station"],
                "landcover": metadata["landcover_2010"],
                "climate": metadata["climate"],
            }
        ]
        jobs.append(
            (idx, metadata["longitude"], metadata["latitude"], metadata_dict)
        )

    # Create the variable ***save_path*** which is a string representing the
    # path where the results will be saved. **DO NOT CHANGE** the name
    # ***save_path*** because it will be searched during the parallel
    # processing!

    save_path = tempfile.mkdtemp()

    # Create the validation object.

    datasets = {
        "ISMN": {
            "class": ismn_reader,
            "columns": ["soil moisture"],
        },
        "ASCAT": {
            "class": ascat_reader,
            "columns": ["sm"],
            "kwargs": {
                "mask_frozen_prob": 80,
                "mask_snow_prob": 80,
                "mask_ssf": True,
            },
        },
    }

    read_ts_names = {"ASCAT": "read", "ISMN": "read"}
    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

    datasets = DataManager(
        datasets, "ISMN", period, read_ts_names=read_ts_names
    )
    process = Validation(
        datasets,
        "ISMN",
        temporal_ref="ASCAT",
        scaling="lin_cdf_match",
        scaling_ref="ASCAT",
        metrics_calculators={
            (2, 2): metrics_calculators.BasicMetrics(
                other_name="k1", metadata_template=metadata_dict_template
            ).calc_metrics
        },
        period=period,
    )

    for job in jobs:
        results = process.calc(*job)
        netcdf_results_manager(results, save_path)

    results_fname = os.path.join(
        save_path, "ASCAT.sm_with_ISMN.soil moisture.nc"
    )
    target_vars = {
        "n_obs": [
            357,
            384,
            1646,
            1875,
            1915,
            467,
            141,
            251
        ],
        "rho": np.array(
            [0.53934574,
             0.7002289,
             0.62200236,
             0.53647155,
             0.30413666,
             0.6740655,
             0.8418981,
             0.74206454,
             ], dtype=np.float32),
        "RMSD": np.array(
            [11.583476,
             7.729667,
             17.441547,
             21.125721,
             14.31557,
             14.187225,
             13.0622425,
             12.903898,
             ], dtype=np.float32),
        "network": np.array([
            "MAQU",
            "MAQU",
            "SCAN",
            "SCAN",
            "SCAN",
            "SOILSCAPE",
            "SOILSCAPE",
            "SOILSCAPE",
        ], dtype="U256",)
    }
    vars_should = [
        'BIAS',
        'R',
        'RMSD',
        '_row_size',
        'climate',
        'gpi',
        'idx',
        'landcover',
        'lat',
        'lon',
        'n_obs',
        'network',
        'p_R',
        'p_rho',
        'p_tau',
        'rho',
        'station',
        'tau',
        'time'
    ]

    check_results(
        filename=results_fname,
        target_vars=target_vars,
        variables=vars_should
    )


def test_validation_with_averager(ascat_reader, ismn_reader):
    """
    Test processing framework with averaging module. ASCAT and ISMN data are used here with no geographical
    considerations (the lut is provided more upstream and contains this information already)
    """
    while hasattr(ascat_reader, 'cls'):
        ascat_reader = ascat_reader.cls
    # lookup table between the ascat and ismn points - not geographically correct
    upscaling_lut = {
        "ISMN": {
            1814367: [(0, 102.1333, 33.8833), (1, 102.1333, 33.6666)],
            1803695: [(2, -86.55, 34.783), (3, -97.083, 37.133), (4, -105.417, 34.25)],
            1856312: [(5, -120.9675, 38.43003), (6, -120.78559, 38.14956), (7, -120.80639, 38.17353)]
        }}
    gpis = (1814367, 1803695, 1856312)
    lons, lats = [], []
    for gpi in gpis:
        lon, lat = ascat_reader.grid.gpi2lonlat(gpi)
        lons.append(lon)
        lats.append(lat)

    jobs = [(gpis, lons, lats)]

    # Create the variable ***save_path*** which is a string representing the
    # path where the results will be saved. **DO NOT CHANGE** the name
    # ***save_path*** because it will be searched during the parallel
    # processing!

    save_path = tempfile.mkdtemp()

    # Create the validation object.

    datasets = {
        "ASCAT": {
            "class": ascat_reader,
            "columns": ["sm"],
            "kwargs": {
                "mask_frozen_prob": 80,
                "mask_snow_prob": 80,
                "mask_ssf": True,
            }
        },
        "ISMN": {
            "class": ismn_reader,
            "columns": ["soil moisture"],
        },
    }

    read_ts_names = {"ASCAT": "read", "ISMN": "read"}
    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

    datasets = DataManager(
        datasets,
        "ASCAT",
        period,
        read_ts_names=read_ts_names,
        upscale_parms={
            "upscaling_method": "average",
            "temporal_stability": True,
            "upscaling_lut": upscaling_lut,
        },
    )
    process = Validation(
        datasets,
        "ASCAT",
        temporal_ref="ISMN",
        scaling="lin_cdf_match",
        scaling_ref="ISMN",
        metrics_calculators={
            (2, 2): metrics_calculators.BasicMetrics(
                other_name="k1"
            ).calc_metrics
        },
        period=period,
    )

    for job in jobs:
        results = process.calc(*job)
        netcdf_results_manager(results, save_path)

    results_fname = os.path.join(
        save_path, "ASCAT.sm_with_ISMN.soil moisture.nc"
    )

    target_vars = {
        "n_obs": [
            764,
            2392,
            904
        ],
        "rho": np.array(
            [-0.012487,
             0.255156,
             0.635517
             ], dtype=np.float32),
        "RMSD": np.array(
            [0.056428,
             0.056508,
             0.116294
             ], dtype=np.float32),
        "R": np.array(
            [-0.012335,
             0.257671,
             0.657239
             ], dtype=np.float32)
    }

    check_results(
        filename=results_fname,
        target_vars=target_vars,
    )


def test_validation_error_n2_k2():
    datasets = setup_TestDatasets()

    dm = DataManager(
        datasets,
        "DS1",
        read_ts_names={d: "read" for d in ["DS1", "DS2", "DS3"]},
    )

    # n less than number of datasets is no longer allowed
    with pytest.raises(ValueError):
        Validation(
            dm,
            "DS1",
            temporal_matcher=temporal_matchers.BasicTemporalMatching(
                window=1 / 24.0
            ).combinatory_matcher,
            scaling="lin_cdf_match",
            metrics_calculators={
                (2, 2): metrics_calculators.BasicMetrics(
                    other_name="k1"
                ).calc_metrics
            },
        )


def test_validation_n3_k2_temporal_matching_no_matches():
    tst_results = {}

    datasets = setup_two_without_overlap()

    dm = DataManager(
        datasets,
        "DS1",
        read_ts_names={d: "read" for d in ["DS1", "DS2", "DS3"]},
    )

    process = Validation(
        dm,
        "DS1",
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0
        ).combinatory_matcher,
        scaling="lin_cdf_match",
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(
                other_name="k1"
            ).calc_metrics
        },
    )

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n3_k2_data_manager_argument():
    tst_results = {
        (("DS1", "x"), ("DS3", "y")): {
            "n_obs": np.array([1000], dtype=np.int32),
            "tau": np.array([np.nan], dtype=np.float32),
            "gpi": np.array([4], dtype=np.int32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "lon": np.array([4.0]),
            "p_tau": np.array([np.nan], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "lat": np.array([4.0]),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
        },
        (("DS1", "x"), ("DS2", "y")): {
            "n_obs": np.array([1000], dtype=np.int32),
            "tau": np.array([np.nan], dtype=np.float32),
            "gpi": np.array([4], dtype=np.int32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "lon": np.array([4.0]),
            "p_tau": np.array([np.nan], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "lat": np.array([4.0]),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
        },
        (("DS1", "x"), ("DS3", "x")): {
            "n_obs": np.array([1000], dtype=np.int32),
            "tau": np.array([np.nan], dtype=np.float32),
            "gpi": np.array([4], dtype=np.int32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "lon": np.array([4.0]),
            "p_tau": np.array([np.nan], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "lat": np.array([4.0]),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
        },
        (("DS2", "y"), ("DS3", "x")): {
            "gpi": np.array([4], dtype=np.int32),
            "lon": np.array([4.0]),
            "lat": np.array([4.0]),
            "n_obs": np.array([1000], dtype=np.int32),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "tau": np.array([np.nan], dtype=np.float32),
            "p_tau": np.array([np.nan], dtype=np.float32),
        },
        (("DS2", "y"), ("DS3", "y")): {
            "gpi": np.array([4], dtype=np.int32),
            "lon": np.array([4.0]),
            "lat": np.array([4.0]),
            "n_obs": np.array([1000], dtype=np.int32),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "tau": np.array([np.nan], dtype=np.float32),
            "p_tau": np.array([np.nan], dtype=np.float32),
        },
    }

    datasets = setup_TestDatasets()
    dm = DataManager(
        datasets,
        "DS1",
        read_ts_names={d: "read" for d in ["DS1", "DS2", "DS3"]},
    )

    process = Validation(
        dm,
        "DS1",
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0
        ).combinatory_matcher,
        scaling="lin_cdf_match",
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(
                other_name="k1"
            ).calc_metrics
        },
    )

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))

    datasets = setup_TestDatasets()
    dm = DataManager(
        datasets,
        "DS1",
        read_ts_names={d: "read" for d in ["DS1", "DS2", "DS3"]},
    )

    process = Validation(
        dm,
        "DS1",
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0
        ).combinatory_matcher,
        scaling="lin_cdf_match",
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(
                other_name="k1"
            ).calc_metrics
        },
    )

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n3_k2():
    tst_results = {
        (("DS1", "x"), ("DS3", "y")): {
            "n_obs": np.array([1000], dtype=np.int32),
            "tau": np.array([np.nan], dtype=np.float32),
            "gpi": np.array([4], dtype=np.int32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "lon": np.array([4.0]),
            "p_tau": np.array([np.nan], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "lat": np.array([4.0]),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
        },
        (("DS1", "x"), ("DS2", "y")): {
            "n_obs": np.array([1000], dtype=np.int32),
            "tau": np.array([np.nan], dtype=np.float32),
            "gpi": np.array([4], dtype=np.int32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "lon": np.array([4.0]),
            "p_tau": np.array([np.nan], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "lat": np.array([4.0]),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
        },
        (("DS1", "x"), ("DS3", "x")): {
            "n_obs": np.array([1000], dtype=np.int32),
            "tau": np.array([np.nan], dtype=np.float32),
            "gpi": np.array([4], dtype=np.int32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "lon": np.array([4.0]),
            "p_tau": np.array([np.nan], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "lat": np.array([4.0]),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
        },
        (("DS2", "y"), ("DS3", "x")): {
            "gpi": np.array([4], dtype=np.int32),
            "lon": np.array([4.0]),
            "lat": np.array([4.0]),
            "n_obs": np.array([1000], dtype=np.int32),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "tau": np.array([np.nan], dtype=np.float32),
            "p_tau": np.array([np.nan], dtype=np.float32),
        },
        (("DS2", "y"), ("DS3", "y")): {
            "gpi": np.array([4], dtype=np.int32),
            "lon": np.array([4.0]),
            "lat": np.array([4.0]),
            "n_obs": np.array([1000], dtype=np.int32),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "tau": np.array([np.nan], dtype=np.float32),
            "p_tau": np.array([np.nan], dtype=np.float32),
        },
    }

    datasets = setup_TestDatasets()
    dm = DataManager(
        datasets,
        "DS1",
        read_ts_names={d: "read" for d in ["DS1", "DS2", "DS3"]},
    )

    process = Validation(
        dm,
        "DS1",
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0
        ).combinatory_matcher,
        scaling="lin_cdf_match",
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(
                other_name="k1"
            ).calc_metrics
        },
    )

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n3_k2_temporal_matching_no_matches2():
    tst_results = {
        (("DS1", "x"), ("DS3", "y")): {
            "n_obs": np.array([1000], dtype=np.int32),
            "tau": np.array([np.nan], dtype=np.float32),
            "gpi": np.array([4], dtype=np.int32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "lon": np.array([4.0]),
            "p_tau": np.array([np.nan], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "lat": np.array([4.0]),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
        },
        (("DS1", "x"), ("DS3", "x")): {
            "n_obs": np.array([1000], dtype=np.int32),
            "tau": np.array([np.nan], dtype=np.float32),
            "gpi": np.array([4], dtype=np.int32),
            "RMSD": np.array([0.0], dtype=np.float32),
            "lon": np.array([4.0]),
            "p_tau": np.array([np.nan], dtype=np.float32),
            "BIAS": np.array([0.0], dtype=np.float32),
            "p_rho": np.array([0.0], dtype=np.float32),
            "rho": np.array([1.0], dtype=np.float32),
            "lat": np.array([4.0]),
            "R": np.array([1.0], dtype=np.float32),
            "p_R": np.array([0.0], dtype=np.float32),
        },
    }

    datasets = setup_three_with_two_overlapping()
    dm = DataManager(
        datasets,
        "DS1",
        read_ts_names={d: "read" for d in ["DS1", "DS2", "DS3"]},
    )

    process = Validation(
        dm,
        "DS1",
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0
        ).combinatory_matcher,
        scaling="lin_cdf_match",
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(
                other_name="k1"
            ).calc_metrics
        },
    )

    jobs = process.get_processing_jobs()
    for job in jobs:
        results = process.calc(*job)
        assert sorted(list(results)) == sorted(list(tst_results))


def test_validation_n3_k2_masking_no_data_remains():
    datasets = setup_TestDatasets()

    # setup masking datasets

    grid = grids.CellGrid(
        np.array([1, 2, 3, 4]),
        np.array([1, 2, 3, 4]),
        np.array([4, 4, 2, 1]),
        gpis=np.array([1, 2, 3, 4]),
    )

    mds1 = GriddedTsBase("", grid, MaskingTestDataset)
    mds2 = GriddedTsBase("", grid, MaskingTestDataset)

    mds = {
        "masking1": {
            "class": mds1,
            "columns": ["x"],
            "args": [],
            "kwargs": {"limit": 500},
            "use_lut": False,
            "grids_compatible": True,
        },
        "masking2": {
            "class": mds2,
            "columns": ["x"],
            "args": [],
            "kwargs": {"limit": 1000},
            "use_lut": False,
            "grids_compatible": True,
        },
    }

    process = Validation(
        datasets,
        "DS1",
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0
        ).combinatory_matcher,
        scaling="lin_cdf_match",
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(
                other_name="k1"
            ).calc_metrics
        },
        masking_datasets=mds,
    )

    gpi_info = (1, 1, 1)
    ref_df = datasets["DS1"]["class"].read(1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        new_ref_df = process.mask_dataset(ref_df, gpi_info)
    assert len(new_ref_df) == 0
    nptest.assert_allclose(new_ref_df.x.values, np.arange(1000, 1000))
    jobs = process.get_processing_jobs()
    for job in jobs:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            results = process.calc(*job)
        tst = []
        assert sorted(list(results)) == sorted(list(tst))
        for key, tst_key in zip(sorted(results), sorted(tst)):
            nptest.assert_almost_equal(
                results[key]["n_obs"], tst[tst_key]["n_obs"]
            )


def test_validation_n3_k2_masking():
    # test result for one gpi in a cell
    tst_results_one = {
        (("DS1", "x"), ("DS3", "y")): {
            "n_obs": np.array([250], dtype=np.int32)
        },
        (("DS1", "x"), ("DS2", "y")): {
            "n_obs": np.array([250], dtype=np.int32)
        },
        (("DS1", "x"), ("DS3", "x")): {
            "n_obs": np.array([250], dtype=np.int32)
        },
        (("DS2", "y"), ("DS3", "x")): {
            "n_obs": np.array([250], dtype=np.int32)
        },
        (("DS2", "y"), ("DS3", "y")): {
            "n_obs": np.array([250], dtype=np.int32)
        },
    }

    # test result for two gpis in a cell
    tst_results_two = {
        (("DS1", "x"), ("DS3", "y")): {
            "n_obs": np.array([250, 250], dtype=np.int32)
        },
        (("DS1", "x"), ("DS2", "y")): {
            "n_obs": np.array([250, 250], dtype=np.int32)
        },
        (("DS1", "x"), ("DS3", "x")): {
            "n_obs": np.array([250, 250], dtype=np.int32)
        },
        (("DS2", "y"), ("DS3", "x")): {
            "n_obs": np.array([250, 250], dtype=np.int32)
        },
        (("DS2", "y"), ("DS3", "y")): {
            "n_obs": np.array([250, 250], dtype=np.int32)
        },
    }

    # cell 4 in this example has two gpis so it returns different results.
    tst_results = {1: tst_results_one, 1: tst_results_one, 2: tst_results_two}

    datasets = setup_TestDatasets()

    # setup masking datasets

    grid = grids.CellGrid(
        np.array([1, 2, 3, 4]),
        np.array([1, 2, 3, 4]),
        np.array([4, 4, 2, 1]),
        gpis=np.array([1, 2, 3, 4]),
    )

    mds1 = GriddedTsBase("", grid, MaskingTestDataset)
    mds2 = GriddedTsBase("", grid, MaskingTestDataset)

    mds = {
        "masking1": {
            "class": mds1,
            "columns": ["x"],
            "args": [],
            "kwargs": {"limit": 500},
            "use_lut": False,
            "grids_compatible": True,
        },
        "masking2": {
            "class": mds2,
            "columns": ["x"],
            "args": [],
            "kwargs": {"limit": 750},
            "use_lut": False,
            "grids_compatible": True,
        },
    }

    process = Validation(
        datasets,
        "DS1",
        temporal_matcher=temporal_matchers.BasicTemporalMatching(
            window=1 / 24.0
        ).combinatory_matcher,
        scaling="lin_cdf_match",
        metrics_calculators={
            (3, 2): metrics_calculators.BasicMetrics(
                other_name="k1"
            ).calc_metrics
        },
        masking_datasets=mds,
    )

    gpi_info = (1, 1, 1)
    ref_df = datasets["DS1"]["class"].read(1)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=DeprecationWarning
        )  # read_ts is hard coded when using mask_data
        new_ref_df = process.mask_dataset(ref_df, gpi_info)
    assert len(new_ref_df) == 250
    nptest.assert_allclose(new_ref_df.x.values, np.arange(750, 1000))
    jobs = process.get_processing_jobs()
    for job in jobs:

        with warnings.catch_warnings():
            # most warnings here are caused by the read_ts function that cannot
            # be changed when using a masking data set
            warnings.simplefilter("ignore", category=DeprecationWarning)
            results = process.calc(*job)

        tst = tst_results[len(job[0])]
        assert sorted(list(results)) == sorted(list(tst))
        for key, tst_key in zip(sorted(results), sorted(tst)):
            nptest.assert_almost_equal(
                results[key]["n_obs"], tst[tst_key]["n_obs"]
            )


# @pytest.mark.slow
# @pytest.mark.full_framework
def test_ascat_ismn_validation_metadata_rolling(ascat_reader, ismn_reader):
    """
    Test processing framework with some ISMN and ASCAT sample data
    """
    jobs = []

    ids = ismn_reader.get_dataset_ids(
        variable="soil moisture", min_depth=0, max_depth=0.1
    )

    metadata_dict_template = {
        "network": np.array(["None"], dtype="U256"),
        "station": np.array(["None"], dtype="U256"),
        "landcover": np.float32([np.nan]),
        "climate": np.array(["None"], dtype="U4"),
    }

    for idx in ids:
        metadata = ismn_reader.metadata[idx]
        metadata_dict = [
            {
                "network": metadata["network"],
                "station": metadata["station"],
                "landcover": metadata["landcover_2010"],
                "climate": metadata["climate"],
            }
        ]
        jobs.append(
            (idx, metadata["longitude"], metadata["latitude"], metadata_dict)
        )

    save_path = tempfile.mkdtemp()

    # Create the validation object.

    datasets = {
        "ISMN": {"class": ismn_reader, "columns": ["soil moisture"]},
        "ASCAT": {
            "class": ascat_reader,
            "columns": ["sm"],
            "kwargs": {
                "mask_frozen_prob": 80,
                "mask_snow_prob": 80,
                "mask_ssf": True,
            },
        },
    }

    read_ts_names = {"ASCAT": "read", "ISMN": "read"}
    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

    datasets = DataManager(
        datasets, "ISMN", period, read_ts_names=read_ts_names
    )

    process = Validation(
        datasets,
        "ISMN",
        temporal_ref="ASCAT",
        scaling="lin_cdf_match",
        scaling_ref="ASCAT",
        metrics_calculators={
            (2, 2): metrics_calculators.RollingMetrics(
                other_name="k1", metadata_template=metadata_dict_template
            ).calc_metrics
        },
        period=period,
    )

    for job in jobs:
        results = process.calc(*job)
        netcdf_results_manager(
            results, save_path, ts_vars=["R", "p_R", "RMSD"]
        )

    results_fname = os.path.join(
        save_path, "ASCAT.sm_with_ISMN.soil moisture.nc"
    )

    target_vars = {
        "network": np.array([
            "MAQU",
            "MAQU",
            "SCAN",
            "SCAN",
            "SCAN",
            "SOILSCAPE",
            "SOILSCAPE",
            "SOILSCAPE",
        ],
            dtype="U256",)
    }
    vars_should = [
        u"gpi",
        u"RMSD",
        u"lon",
        u"lat",
        u"R",
        u"p_R",
        u"time",
        u"idx",
        u"_row_size"
    ]
    for key, value in metadata_dict_template.items():
        vars_should.append(key)

    check_results(
        filename=results_fname,
        target_vars=target_vars,
        variables=vars_should
    )

    reader = PointDataResults(results_fname, read_only=True)
    df = reader.read_loc(None)
    assert np.all(df.gpi.values == np.arange(8))
    assert reader.read_ts(0).index.size == 357
    assert np.all(
        reader.read_ts(1).columns.values == np.array(["R", "p_R", "RMSD"])
    )


def test_args_to_iterable_non_iterables():
    gpis = 1
    lons = 1
    lats = 1
    arg1 = 1
    arg2 = 2
    arg3 = 3
    gpis_, lons_, lats_, args = args_to_iterable(
        gpis, lons, lats, arg1, arg2, arg3, n=3
    )

    assert gpis_ == [gpis]
    assert lons_ == [lons]
    assert lats_ == [lats]
    assert args == ([arg1], [arg2], [arg3])


def test_args_to_iterable_n3():
    gpis = [1, 2, 3]
    lons = [2, 3, 4]
    lats = [3, 4, 5]
    arg1 = [1, 1, 1]
    arg2 = [1, 1, 1]
    gpis_, lons_, lats_, args = args_to_iterable(
        gpis, lons, lats, arg1, arg2, n=3
    )

    assert gpis_ == gpis
    assert lons_ == lons
    assert lats_ == lats
    assert args == (arg1, arg2)

    zipped_should = [(1, 2, 3, 1, 1), (2, 3, 4, 1, 1), (3, 4, 5, 1, 1)]

    for i, t in enumerate(zip(gpis_, lons_, lats_, *args)):
        assert zipped_should[i] == t


def test_args_to_iterable_mixed():
    gpis = [1, 2, 3]
    lons = [2, 3, 4]
    lats = 1
    arg1 = 1
    gpis_, lons_, lats_, args = args_to_iterable(gpis, lons, lats, arg1)

    assert gpis_ == gpis
    assert lons_ == lons
    assert lats_ == [lats]
    assert args == [arg1]


def test_args_to_iterable_mixed_strings():
    gpis = [1, 2, 3]
    lons = [2, 3, 4]
    lats = 1
    arg1 = "test"
    gpis_, lons_, lats_, args = args_to_iterable(gpis, lons, lats, arg1)

    assert gpis_ == gpis
    assert lons_ == lons
    assert lats_ == [lats]
    assert args == [arg1]


#######################################################################
# Tests for new temporal matcher & metric calculators
#######################################################################


def create_correlated_data(n_datasets, n, r):
    """Creates n_datasets random timeseries with specified correlation"""
    C = np.ones((n_datasets, n_datasets)) * r
    for i in range(n_datasets):
        C[i, i] = 1
    A = np.linalg.cholesky(C)

    return (A @ np.random.randn(n_datasets, n)).T


class DummyReader:
    def __init__(self, dfs, name):
        self.data = [pd.DataFrame(dfs[i][name]) for i in range(len(dfs))]

    def read(self, gpi, *args, **kwargs):
        return self.data[gpi]


class DummyNoneReader:
    def __init__(self, dfs, name):
        self.data = [pd.DataFrame(dfs[i][name]) for i in range(len(dfs))]

    def read(self, gpi, *args, **kwargs):
        names = self.data[gpi].columns
        return pd.DataFrame(np.zeros((0, len(names))), columns=names)


def create_datasets(n_datasets, npoints, nsamples, missing=False):
    """
    Creates three datasets with given number of points to compare, each
    having number of samples given
    """
    dfs = []
    for gpi in range(npoints):
        r = np.random.rand()
        data = create_correlated_data(n_datasets, nsamples, r)
        index = pd.date_range("1980", periods=nsamples, freq="D")
        dfs.append(pd.DataFrame(
            data, index=index, columns=(
                    ["refcol"] + [f"other{i}col" for i in range(1, n_datasets)]
            )
        ))

    datasets = {}
    datasets["0-ERA5"] = {
        "columns": ["refcol"],
        "class": DummyReader(dfs, "refcol")
    }
    for i in range(1, n_datasets - 1):
        datasets[f"{i}-ESA_CCI_SM_combined"] = {
            "columns": [f"other{i}col"],
            "class": DummyReader(dfs, f"other{i}col")
        }
    if missing:
        datasets[f"{n_datasets - 1}-missing"] = {
            "columns": [f"other{n_datasets - 1}col"],
            "class": DummyNoneReader(dfs, f"other{n_datasets - 1}col")
        }
    else:
        datasets[f"{n_datasets - 1}-ESA_CCI_SM_combined"] = {
            "columns": [f"other{n_datasets - 1}col"],
            "class": DummyReader(dfs, f"other{n_datasets - 1}col")
        }
    return datasets


def test_missing_data():
    n_datasets = 5
    npoints = 5
    nsamples = 100

    datasets = create_datasets(n_datasets, npoints, nsamples, missing=True)

    metric_calculator = PairwiseIntercomparisonMetrics()

    val = Validation(
        datasets,
        spatial_ref="0-ERA5",
        metrics_calculators={(n_datasets, 2): metric_calculator.calc_metrics},
        temporal_matcher=make_combined_temporal_matcher(pd.Timedelta(12, "H")),
    )
    gpis = list(range(npoints))
    val.calc(gpis, gpis, gpis, rename_cols=False, only_with_temporal_ref=True)


def test_combined_matching_scaling():
    n_datasets = 5
    npoints = 5
    nsamples = 100

    datasets = create_datasets(n_datasets, npoints, nsamples)

    metric_calculator = PairwiseIntercomparisonMetrics()

    val = Validation(
        datasets,
        spatial_ref="0-ERA5",
        metrics_calculators={(n_datasets, 2): metric_calculator.calc_metrics},
        temporal_matcher=make_combined_temporal_matcher(pd.Timedelta(12, "H")),
        scaling="mean_std",
        scaling_ref="0-ERA5",
    )
    gpis = list(range(npoints))
    val.calc(gpis, gpis, gpis, rename_cols=False, only_with_temporal_ref=True)


if __name__ == "__main__":
    # for profiling with cProfile, on the command line run
    # python -m cProfile -o ascat_ismn_validation.profile test_validation.py
    # output can be investigated with snakeviz
    test_ascat_ismn_validation()
    # profiling result: most of the time is spend loading the data (110s), next
    # bigger chunk of computation time is temporal matching (6.7s, old
    # implementation, 0.3s new implementation)
