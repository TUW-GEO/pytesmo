# Copyright (c) 2021,Vienna University of Technology,
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
#     to endorse or promote products derived from this software without specific
#     prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Test for the data averaging class"""

import warnings

import pandas as pd
import numpy as np

import pytest
import os
from datetime import datetime

from pytesmo.validation_framework.data_averaging import DataAverager

from ismn.interface import ISMN_Interface
from esa_cci_sm.interface import CCITs

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from ascat.read_native.cdr import AscatGriddedNcTs


@pytest.fixture
def ismn_reader():
    ismn_data_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test-data",
        "ismn",
        "multinetwork",
        "header_values",
    )

    return ISMN_Interface(ismn_data_folder)


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
def cci_reader():
    cci_data_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test-data",
        "sat",
        "ESA_CCI_SM_combined",
        "ESA_CCI_SM_C_V04_7",
    )
    cci_reader = CCITs(cci_data_folder)
    cci_reader.read_bulk = True

    return cci_reader


@pytest.fixture
def cci_test(ismn_reader, cci_reader):
    """Setup of test for ISMN averaged in cci (regular) grid"""
    manager_parms = {
        "datasets": {
            "ISMN": {
                "class": ismn_reader,
                "columns": ["soil moisture"],
                "args": [],
                "kwargs": {},
            },
            "ESA_CCI_SM_combined": {
                "class": cci_reader,
                "columns": ["ESA_CCI_SM_C_sm"],
                "kwargs": {},
            },
        }, "read_ts_names": {
            "ESA_CCI_SM_combined": "read", "ISMN": "read_ts"
        },
        "period": [datetime(2007, 1, 1), datetime(2014, 12, 31)],
        "ref_class": cci_reader,
        "others_class": {"ISMN": ismn_reader}
    }

    cci_subset = (
            41.08659705984312,
            41.545589036668105,
            -5.722503662109376,
            -5.060577392578126,
        )

    cci_points = cci_reader.grid.get_bbox_grid_points(*cci_subset)

    cci_test_averager = DataAverager(
        ref_class=cci_reader,
        others_class={"ISMN": ismn_reader},
        geo_subset=cci_subset,
        manager_parms=manager_parms
    )

    return cci_test_averager, cci_points


def test_cci(cci_test):
    """Test that averaging doesn't produce errors with a cci (regular) grid"""
    cci_test_averager, cci_points = cci_test

    ismn_upscaled = []
    for gpi in cci_points:
        upscaled = cci_test_averager.get_upscaled_ts(
            gpi=gpi,
            other_name="ISMN",
            temporal_stability=True
        )
        ismn_upscaled.append(upscaled)

    for upscaled in ismn_upscaled:
        if upscaled is not None:
            assert isinstance(upscaled, pd.DataFrame), "get_upscaled_ts should always return a Dataframe or None"


@pytest.fixture
def synthetic_test():
    """Create test with synthetic readers"""
    # todo


def test_upscale(cci_test):
    """Test all upscaling functions"""
    cci_test_averager, cci_points = cci_test

    to_upscale = pd.concat(
        [pd.Series(2, index=np.linspace(1,10), name='sm'),
         pd.Series(4, index=np.linspace(1,10), name='sm')],
        axis=1
    )
    # simple check of series averaging
    upscaled = cci_test_averager.upscale(to_upscale, method="average")
    should = pd.Series(float(3), index=np.linspace(1,10))
    assert upscaled.equals(should)


def test_tstability(cci_test):
    """Test temporal stability filtering with noisy or uncorrelated series"""
    cci_test_averager, cci_points = cci_test
    n_obs = 1000
    points = np.linspace(0, 2*np.pi, n_obs)
    ts = np.sin(points)
    low_corr = np.sin(points+np.pi)
    high_sterr = np.sin(points) + np.random.normal(0, 2, n_obs)
    to_filter = pd.concat(
        [pd.Series(ts, name='sm_1'),
         pd.Series(ts, name='sm_2'),
         pd.Series(ts, name='sm_3'),
         pd.Series(low_corr, name='low_corr'),
         pd.Series(high_sterr, name='high_sterr')],
        axis=1
    )

    filtered = cci_test_averager.tstability_filter(to_filter)
    should = to_filter.drop(["low_corr", "high_sterr"], axis="columns")
    assert filtered.equals(should)


def test_temporal_matching(cci_test):
    """Test temporal matching"""
    cci_test_averager, cci_points = cci_test
    data_ref = np.arange(30.)
    data2match = data_ref[:-1]
    data2match[2] = np.nan

    ref_ser = pd.Series(
        data_ref,
        index=pd.date_range(datetime(2007, 1, 1, 0), datetime(2007, 1, 30, 0), freq="D"),
    ).to_frame()
    match_ser = pd.Series(
        data2match,
        index=pd.date_range(datetime(2007, 1, 1, 5), datetime(2007, 1, 29, 5), freq="D"),
    ).to_frame()
    to_match = [ref_ser, match_ser]

    matched = cci_test_averager.temporal_match(to_match, drop_missing=False)
    assert len(matched.index) == 29, "Should be matched to the longest timeseries"

    matched = cci_test_averager.temporal_match(to_match, drop_missing=True)
    assert len(matched.index) == 28, matched

    matched = cci_test_averager.temporal_match(to_match, hours=4)
    assert matched[matched.columns[1]].dropna().empty, "Should not be matched"
