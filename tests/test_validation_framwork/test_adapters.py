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

'''
Test for the adapters.
'''

from pytesmo.validation_framework.adapters import MaskingAdapter
from pytesmo.validation_framework.adapters import SelfMaskingAdapter
from pytesmo.validation_framework.adapters import AnomalyAdapter
from pytesmo.validation_framework.adapters import AnomalyClimAdapter

from tests.test_validation_framwork.test_datasets import TestDataset

import numpy as np
import numpy.testing as nptest


def test_masking_adapter():
    for col in (None, 'x'):
        ds = TestDataset('', n=20)
        ds_mask = MaskingAdapter(ds, '<', 10, col)
        data_masked = ds_mask.read_ts()
        data_masked2 = ds_mask.read()

        nptest.assert_almost_equal(data_masked['x'].values,
                                   np.concatenate([np.ones((10), dtype=bool),
                                                   np.zeros((10), dtype=bool)]))
        nptest.assert_almost_equal(data_masked2['x'].values,
                                   np.concatenate([np.ones((10), dtype=bool),
                                                   np.zeros((10), dtype=bool)]))

        if col is None:
            nptest.assert_almost_equal(
                data_masked['y'].values, np.ones((20), dtype=bool))
            nptest.assert_almost_equal(
                data_masked2['y'].values, np.ones((20), dtype=bool))

def test_self_masking_adapter():
    ref_x = np.arange(10)
    ref_y = np.arange(10) * 0.5
    ds = TestDataset('', n=20)
    
    ds_mask = SelfMaskingAdapter(ds, '<', 10, 'x')
    data_masked = ds_mask.read_ts()
    data_masked2 = ds_mask.read()

    nptest.assert_almost_equal(data_masked['x'].values,ref_x)
    nptest.assert_almost_equal(data_masked2['x'].values,ref_x)
    nptest.assert_almost_equal(data_masked['y'].values,ref_y)
    nptest.assert_almost_equal(data_masked2['y'].values,ref_y)

def test_anomaly_adapter():
    ds = TestDataset('', n=20)
    ds_anom = AnomalyAdapter(ds)
    data_anom = ds_anom.read_ts()
    data_anom2 = ds_anom.read()
    nptest.assert_almost_equal(data_anom['x'].values[0], -8.5)
    nptest.assert_almost_equal(data_anom['y'].values[0], -4.25)
    nptest.assert_almost_equal(data_anom2['x'].values[0], -8.5)
    nptest.assert_almost_equal(data_anom2['y'].values[0], -4.25)


def test_anomaly_adapter_one_column():
    ds = TestDataset('', n=20)
    ds_anom = AnomalyAdapter(ds, columns=['x'])
    data_anom = ds_anom.read_ts()
    nptest.assert_almost_equal(data_anom['x'].values[0], -8.5)
    nptest.assert_almost_equal(data_anom['y'].values[0], 0)


def test_anomaly_clim_adapter():
    ds = TestDataset('', n=20)
    ds_anom = AnomalyClimAdapter(ds)
    data_anom = ds_anom.read_ts()
    data_anom2 = ds_anom.read()
    nptest.assert_almost_equal(data_anom['x'].values[4], -5.5)
    nptest.assert_almost_equal(data_anom['y'].values[4], -2.75)
    nptest.assert_almost_equal(data_anom2['x'].values[4], -5.5)
    nptest.assert_almost_equal(data_anom2['y'].values[4], -2.75)


def test_anomaly_clim_adapter_one_column():
    ds = TestDataset('', n=20)
    ds_anom = AnomalyClimAdapter(ds, columns=['x'])
    data_anom = ds_anom.read_ts()
    nptest.assert_almost_equal(data_anom['x'].values[4], -5.5)
    nptest.assert_almost_equal(data_anom['y'].values[4], 2)
