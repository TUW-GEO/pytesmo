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
from datetime import datetime

from pytesmo.validation_framework.data_averaging import DataAverager

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")


def return_averager():
    manager_parms = {
        "datasets": None,
        "period": None,
        "read_ts_names": None,
    }
    averager = DataAverager(
        ref_class=None,
        others_class={},
        upscaling_lut={},
        manager_parms=manager_parms
    )

    return averager


def test_upscale():
    """Test all upscaling functions"""
    averager = return_averager()

    to_upscale = pd.concat(
        [pd.Series(2, index=np.linspace(1,10), name='sm'),
         pd.Series(4, index=np.linspace(1,10), name='sm')],
        axis=1
    )
    # simple check of series averaging
    upscaled = averager.upscale(to_upscale, method="average")
    should = pd.Series(float(3), index=np.linspace(1,10))
    assert upscaled.equals(should)


def test_tstability():
    """Test temporal stability filtering with noisy or uncorrelated series"""
    averager = return_averager()
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

    filtered = averager.tstability_filter(to_filter)
    should = to_filter.drop(["low_corr", "high_sterr"], axis="columns")
    assert filtered.equals(should)


def test_temporal_matching():
    """Test temporal matching"""
    averager = return_averager()
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

    matched = averager.temporal_match(to_match, drop_missing=False)
    assert len(matched.index) == 30, "Should be matched to the longest timeseries"

    matched = averager.temporal_match(to_match, drop_missing=True)
    assert len(matched.index) == 28, "Should drop the row and the missing timestep with a missing value"

    matched = averager.temporal_match(to_match, hours=4)
    assert matched[matched.columns[1]].dropna().empty, "Should not be matched"
