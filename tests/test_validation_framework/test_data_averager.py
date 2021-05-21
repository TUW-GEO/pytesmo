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

'''
Test for the data averaging class:

* First test uses ISMN as a non-reference dataset
* Second test averages a higher-resolution satellite dataset, used as non-reference with a lower
    res satellite as reference
'''
import warnings
import pytest
import os
from datetime import datetime
import numpy as np

from pytesmo.validation_framework.data_averaging import DataAverager

from ismn.interface import ISMN_Interface
from smecv_grid.grid import SMECV_Grid_v052
from esa_cci_sm.interface import CCITs
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from ascat.read_native.cdr import AscatGriddedNcTs


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
def first_test_manager_parms(cci_reader):
    """Imputs of data averaging class"""
    ismn_data_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test-data",
        "ismn",
        "multinetwork",
        "header_values",
    )
    ismn_reader = ISMN_Interface(ismn_data_folder)
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

    return manager_parms


def test_ismn_averaging(first_test_manager_parms, cci_reader):
    ref_class = first_test_manager_parms["ref_class"]
    others_class = first_test_manager_parms["others_class"]

    averager = DataAverager(
        ref_class=ref_class,
        others_class=others_class,
        geo_subset=(
            41.08659705984312,
            41.545589036668105,
            -5.722503662109376,
            -5.060577392578126,
        ),
        manager_parms=first_test_manager_parms
    )
    print(averager.lut)
    remedhus_bbox = [
        41.08659705984312,
        41.545589036668105,
        -5.722503662109376,
        -5.060577392578126,
    ]
    cci_points = cci_reader.grid.get_bbox_grid_points(*remedhus_bbox)

    ismn_upscaled = []
    for gpi in cci_points:
        upscaled = averager.wrapper(
            gpi=gpi,
            other_name="ISMN",
        )
        ismn_upscaled.append(upscaled)
    print(ismn_upscaled)
