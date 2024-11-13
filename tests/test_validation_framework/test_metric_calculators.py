# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of TU Wien, Department of Geodesy and Geoinformation nor
#     the names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.

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

from datetime import datetime
import numpy as np
from numpy.testing import (
    assert_equal,
    assert_almost_equal,
    assert_allclose,
)
import pandas as pd
from pathlib import Path
import pytest
import shutil

from pytesmo.metrics import with_analytical_ci, with_bootstrapped_ci, pairwise
from pytesmo.validation_framework.validation import Validation
import pytesmo.validation_framework.error_handling as eh
from pytesmo.validation_framework.metric_calculators import (
    MetadataMetrics,
    BasicMetrics,
    BasicMetricsPlusMSE,
    IntercomparisonMetrics,
    TCMetrics,
    FTMetrics,
    HSAF_Metrics,
    RollingMetrics,
    PairwiseIntercomparisonMetrics,
    TripleCollocationMetrics,
)
from pytesmo.validation_framework.metric_calculators_adapters import (
    MonthsMetricsAdapter,
)

from pytesmo.validation_framework.temporal_matchers import (
    make_combined_temporal_matcher,
    BasicTemporalMatching,
)
from pytesmo.validation_framework.results_manager import netcdf_results_manager
import pytesmo.metrics as metrics

from .utils import DummyReader


def make_some_data():
    """
    Create a data frame with 3 columns and a pre defined bias.
    """
    startdate = datetime(2000, 1, 1)
    enddate = datetime(2000, 12, 31)
    dt_index = pd.date_range(start=startdate, end=enddate, freq="D")

    names = ["ref", "k1", "k2", "k3"]
    # always 0.5
    df = pd.DataFrame(
        index=dt_index,
        data={name: np.repeat(0.5, dt_index.size) for name in names},
    )

    df["k1"] += 0.2  # some positive bias
    df["k2"] -= 0.2  # some negative bias
    df["k3"] -= 0.3  # some more negative bias

    return df


def test_MetadataMetrics_calculator():
    """
    Test MetadataMetrics.
    """
    df = make_some_data()
    data = df[["ref", "k1"]]

    metriccalc = MetadataMetrics(other_name="k1")
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    assert sorted(list(res.keys())) == sorted(["gpi", "lon", "lat", "status"])

    metadata_dict_template = {
        "network": np.array(["None"], dtype="U256"),
        "station": np.array(["None"], dtype="U256"),
        "landcover": np.int32([-1]),
        "climate": np.array(["None"], dtype="U4"),
    }

    metadata_dict = {
        "network": "SOILSCAPE",
        "station": "node1200",
        "landcover": 110,
        "climate": "Csa",
    }

    metriccalc = MetadataMetrics(
        other_name="k1", metadata_template=metadata_dict_template
    )
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0, metadata_dict))
    for key, value in metadata_dict.items():
        assert res[key] == metadata_dict[key]


def test_BasicMetrics_calculator():
    """
    Test BasicMetrics.
    """
    df = make_some_data()
    data = df[["ref", "k1"]]

    metriccalc = BasicMetrics(other_name="k1", calc_tau=False)
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(
        n_obs=np.array([366]),
        RMSD=np.array([0.2], dtype="float32"),
        BIAS=np.array([-0.2], dtype="float32"),
    )

    assert res["n_obs"] == should["n_obs"]
    assert np.isnan(res["rho"])
    assert res["RMSD"] == should["RMSD"]
    assert res["BIAS"] == should["BIAS"]
    assert np.isnan(res["R"])

    # scipy 1.3.0 is not built for python 2.7 so we allow both for now
    assert np.isnan(res["p_R"]) or res["p_R"] == 1.0


def test_BasicMetrics_calculator_metadata():
    """
    Test BasicMetrics with metadata.
    """
    df = make_some_data()
    data = df[["ref", "k1"]]

    metadata_dict_template = {
        "network": np.array(["None"], dtype="U256")}

    metriccalc = BasicMetrics(
        other_name="k1",
        calc_tau=False,
        metadata_template=metadata_dict_template,
    )

    res = metriccalc.calc_metrics(
        data, gpi_info=(0, 0, 0, {"network": "SOILSCAPE"})
    )

    should = dict(
        network=np.array(["SOILSCAPE"], dtype="U256"),
        n_obs=np.array([366]),
        RMSD=np.array([0.2], dtype="float32"),
        BIAS=np.array([-0.2], dtype="float32"),
        dtype="float32",
    )

    assert res["n_obs"] == should["n_obs"]
    assert np.isnan(res["rho"])
    assert res["RMSD"] == should["RMSD"]
    assert res["BIAS"] == should["BIAS"]
    assert res["network"] == should["network"]
    assert np.isnan(res["R"])
    # depends on scipy version changed after v1.2.1
    assert res["p_R"] == np.array([1.0]) or np.isnan(res["R"])


def test_BasicMetricsPlusMSE_calculator():
    """
    Test BasicMetricsPlusMSE.
    """
    df = make_some_data()
    data = df[["ref", "k1"]]

    metriccalc = BasicMetricsPlusMSE(other_name="k1")
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(
        network=np.array(["SOILSCAPE"], dtype="U256"),
        n_obs=np.array([366]),
        RMSD=np.array([0.2], dtype="float32"),
        BIAS=np.array([-0.2], dtype="float32"),
        dtype="float32",
    )

    assert res["n_obs"] == should["n_obs"]
    assert np.isnan(res["rho"])
    assert res["RMSD"] == should["RMSD"]
    assert res["BIAS"] == should["BIAS"]
    assert np.isnan(res["R"])
    # depends on scipy version changed after v1.2.1
    assert res["p_R"] == np.array([1.0]) or np.isnan(res["R"])


def test_BasicMetricsPlusMSE_calculator_metadata():
    """
    Test BasicMetricsPlusMSE with metadata.
    """
    df = make_some_data()
    data = df[["ref", "k1"]]

    metadata_dict_template = {
        "network": np.array(["None"], dtype="U256")}

    metriccalc = BasicMetricsPlusMSE(
        other_name="k1", metadata_template=metadata_dict_template
    )
    res = metriccalc.calc_metrics(
        data, gpi_info=(0, 0, 0, {"network": "SOILSCAPE"})
    )

    should = dict(
        network=np.array(["SOILSCAPE"], dtype="U256"),
        n_obs=np.array([366]),
        RMSD=np.array([0.2], dtype="float32"),
        BIAS=np.array([-0.2], dtype="float32"),
        dtype="float32",
    )

    assert res["n_obs"] == should["n_obs"]
    assert np.isnan(res["rho"])
    assert res["RMSD"] == should["RMSD"]
    assert res["BIAS"] == should["BIAS"]
    assert res["network"] == should["network"]
    assert np.isnan(res["R"])
    # depends on scipy version changed after v1.2.1
    assert res["p_R"] == np.array([1.0]) or np.isnan(res["R"])


def test_IntercompMetrics_calculator():
    """
    Test IntercompMetrics.
    """
    df = make_some_data()
    data = df[["ref", "k1", "k2"]]

    with pytest.warns(DeprecationWarning):
        metriccalc = IntercomparisonMetrics(
            other_names=("k1", "k2"), calc_tau=True
        )

    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    assert res["n_obs"] == np.array([366])

    assert np.isnan(res["R_between_ref_and_k1"])
    assert np.isnan(res["R_between_ref_and_k2"])

    assert np.isnan(res["rho_between_ref_and_k1"])
    assert np.isnan(res["rho_between_ref_and_k2"])

    np.testing.assert_almost_equal(
        res["mse_between_ref_and_k1"], np.array([0.04], dtype=np.float32)
    )
    np.testing.assert_almost_equal(
        res["mse_between_ref_and_k2"], np.array([0.04], dtype=np.float32)
    )

    np.testing.assert_almost_equal(
        res["mse_corr_between_ref_and_k1"],
        np.array([0], dtype=np.float32)
    )
    np.testing.assert_almost_equal(
        res["mse_corr_between_ref_and_k2"],
        np.array([0], dtype=np.float32)
    )

    np.testing.assert_almost_equal(
        res["mse_bias_between_ref_and_k1"],
        np.array([0.04], dtype=np.float32)
    )
    np.testing.assert_almost_equal(
        res["mse_bias_between_ref_and_k2"],
        np.array([0.04], dtype=np.float32)
    )

    # scipy 1.3.0 is not built for python 2.7 so we allow both for now
    assert (
            np.isnan(res["p_R_between_ref_and_k1"])
            or res["p_R_between_ref_and_k1"] == 1.0
    )
    assert (
            np.isnan(res["p_R_between_ref_and_k2"])
            or res["p_R_between_ref_and_k2"] == 1.0
    )

    assert (res["RMSD_between_ref_and_k1"] ==
            np.array([0.2], dtype="float32"))
    assert (res["RMSD_between_ref_and_k2"] ==
            np.array([0.2], dtype="float32"))

    assert (res["BIAS_between_ref_and_k1"] ==
            np.array([-0.2], dtype="float32"))
    assert (res["BIAS_between_ref_and_k2"] ==
            np.array([0.2], dtype="float32"))

    np.testing.assert_almost_equal(
        res["urmsd_between_ref_and_k1"], np.array([0.0], dtype="float32")
    )
    np.testing.assert_almost_equal(
        res["urmsd_between_ref_and_k2"], np.array([0.0], dtype="float32")
    )

    assert "RSS_between_ref_and_k1" in res.keys()
    assert "RSS_between_ref_and_k2" in res.keys()


def test_IntercompMetrics_calculator_metadata():
    """
    Test IntercompMetrics with metadata.
    """
    df = make_some_data()
    data = df[["ref", "k1", "k2"]]

    metadata_dict_template = {
        "network": np.array(["None"], dtype="U256")}

    with pytest.warns(DeprecationWarning):
        metriccalc = IntercomparisonMetrics(
            other_names=("k1", "k2"),
            calc_tau=True,
            metadata_template=metadata_dict_template,
        )
    res = metriccalc.calc_metrics(
        data, gpi_info=(0, 0, 0, {"network": "SOILSCAPE"})
    )

    assert res["network"] == np.array(["SOILSCAPE"], dtype="U256")


def test_TC_metrics_calculator():
    """
    Test TC metrics.
    """
    # this calculator uses a reference data set that is part of ALL triples.
    df = make_some_data()
    data = df[["ref", "k1", "k2", "k3"]]

    with pytest.warns(DeprecationWarning):
        metriccalc = TCMetrics(
            other_names=("k1", "k2", "k3"),
            calc_tau=True,
            dataset_names=("ref", "k1", "k2", "k3"),
        )

    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    assert res["n_obs"] == np.array([366])

    assert np.isnan(res["R_between_ref_and_k1"])
    assert np.isnan(res["R_between_ref_and_k2"])

    assert np.isnan(res["rho_between_ref_and_k1"])
    assert np.isnan(res["rho_between_ref_and_k2"])

    np.testing.assert_almost_equal(
        res["mse_between_ref_and_k1"],
        np.array([0.04], dtype=np.float32)
    )
    np.testing.assert_almost_equal(
        res["mse_between_ref_and_k2"],
        np.array([0.04], dtype=np.float32)
    )

    np.testing.assert_almost_equal(
        res["mse_corr_between_ref_and_k1"],
        np.array([0], dtype=np.float32)
    )
    np.testing.assert_almost_equal(
        res["mse_corr_between_ref_and_k2"],
        np.array([0], dtype=np.float32)
    )

    np.testing.assert_almost_equal(
        res["mse_bias_between_ref_and_k1"],
        np.array([0.04], dtype=np.float32)
    )
    np.testing.assert_almost_equal(
        res["mse_bias_between_ref_and_k2"],
        np.array([0.04], dtype=np.float32)
    )

    # scipy 1.3.0 is not built for python 2.7 so we allow both for now
    assert (
            np.isnan(res["p_R_between_ref_and_k1"])
            or res["p_R_between_ref_and_k1"] == 1.0
    )
    assert (
            np.isnan(res["p_R_between_ref_and_k2"])
            or res["p_R_between_ref_and_k2"] == 1.0
    )

    assert (res["RMSD_between_ref_and_k1"] ==
            np.array([0.2], dtype="float32"))
    assert (res["RMSD_between_ref_and_k2"] ==
            np.array([0.2], dtype="float32"))

    assert (res["BIAS_between_ref_and_k1"] ==
            np.array([-0.2], dtype="float32"))
    assert (res["BIAS_between_ref_and_k2"] ==
            np.array([0.2], dtype="float32"))

    np.testing.assert_almost_equal(
        res["urmsd_between_ref_and_k1"],
        np.array([0.0], dtype="float32")
    )
    np.testing.assert_almost_equal(
        res["urmsd_between_ref_and_k2"],
        np.array([0.0], dtype="float32")
    )

    assert "RSS_between_ref_and_k1" in res.keys()
    assert "RSS_between_ref_and_k2" in res.keys()
    # each non-ref dataset has a snr, err and beta

    assert np.isnan(res["snr_k1_between_ref_and_k1_and_k2"])
    assert np.isnan(res["snr_k2_between_ref_and_k1_and_k2"])
    assert np.isnan(res["snr_k2_between_ref_and_k2_and_k3"])
    assert np.isnan(res["err_std_k1_between_ref_and_k1_and_k2"])
    np.testing.assert_almost_equal(
        res["beta_k1_between_ref_and_k1_and_k2"][0], 0.0
    )
    np.testing.assert_almost_equal(
        res["beta_k2_between_ref_and_k1_and_k2"][0], 0.0
    )
    np.testing.assert_almost_equal(
        res["beta_k3_between_ref_and_k1_and_k3"][0], 0.0
    )


def test_TC_metrics_calculator_metadata():
    """
    Test TC metrics with metadata.
    """
    df = make_some_data()
    data = df[["ref", "k1", "k2"]]

    metadata_dict_template = {
        "network": np.array(["None"], dtype="U256")}

    with pytest.warns(DeprecationWarning):
        metriccalc = TCMetrics(
            other_names=("k1", "k2"),
            calc_tau=True,
            dataset_names=["ref", "k1", "k2"],
            metadata_template=metadata_dict_template,
        )
    res = metriccalc.calc_metrics(
        data, gpi_info=(0, 0, 0, {"network": "SOILSCAPE"})
    )

    assert res["network"] == np.array(["SOILSCAPE"], dtype="U256")


def test_FTMetrics():
    """
    Test FT metrics.
    """
    df = make_some_data()
    data = df[["ref", "k1"]]

    metriccalc = FTMetrics(frozen_flag=2, other_name="k1")
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(
        n_obs=np.array([366]),
        ssf_fr_temp_un=np.array([0.0], dtype="float32"),
        dtype="float32",
    )

    assert res["n_obs"] == should["n_obs"]
    assert res["ssf_fr_temp_un"] == should["ssf_fr_temp_un"]


def test_FTMetrics_metadata():
    """
    Test FT metrics with metadata.
    """
    df = make_some_data()
    data = df[["ref", "k1"]]

    metadata_dict_template = {
        "network": np.array(["None"], dtype="U256")}

    metriccalc = FTMetrics(
        frozen_flag=2,
        other_name="k1",
        metadata_template=metadata_dict_template,
    )
    res = metriccalc.calc_metrics(
        data, gpi_info=(0, 0, 0, {"network": "SOILSCAPE"})
    )

    assert res["network"] == np.array(["SOILSCAPE"], dtype="U256")


def test_BasicSeasonalMetrics():
    """
    Test BasicSeasonalMetrics.
    """
    df = make_some_data()
    data = df[["ref", "k1"]]

    with pytest.warns(UserWarning):
        metriccalc = MonthsMetricsAdapter(BasicMetrics(other_name="k1"))
        res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = {("ALL", "n_obs"): np.array([366], dtype="float32")}

    assert res[("ALL", "n_obs")] == should[("ALL", "n_obs")]
    assert np.isnan(res[("ALL", "rho")])


def test_BasicSeasonalMetrics_metadata():
    """
    Test BasicSeasonalMetrics with metadata.
    """
    df = make_some_data()
    data = df[["ref", "k1"]]

    metadata_dict_template = {
        "network": np.array(["None"], dtype="U256")}

    with pytest.warns(UserWarning):
        metriccalc = MonthsMetricsAdapter(
            BasicMetrics(
                other_name="k1", metadata_template=metadata_dict_template
            )
        )
        res = metriccalc.calc_metrics(
            data, gpi_info=(0, 0, 0, {"network": "SOILSCAPE"})
        )

    assert res["network"] == np.array(["SOILSCAPE"], dtype="U256")


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide*:RuntimeWarning")
def test_HSAF_Metrics():
    """
    Test HSAF Metrics
    """
    df = make_some_data()
    data = df[["ref", "k1", "k2"]]

    metriccalc = HSAF_Metrics(other_name1="k1", other_name2="k2")
    res = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0))

    should = dict(ALL_n_obs=np.array([366]), dtype="float32")

    assert res["ALL_n_obs"] == should["ALL_n_obs"]
    assert np.isnan(res["ref_k1_ALL_rho"])
    assert np.isnan(res["ref_k2_ALL_rho"])


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide*:RuntimeWarning")
def test_HSAF_Metrics_metadata():
    """
    Test HSAF Metrics with metadata.
    """
    df = make_some_data()
    data = df[["ref", "k1", "k2"]]

    metadata_dict_template = {"network": np.array(["None"], dtype="U256")}

    metriccalc = HSAF_Metrics(
        other_name1="k1", metadata_template=metadata_dict_template
    )
    res = metriccalc.calc_metrics(
        data, gpi_info=(0, 0, 0, {"network": "SOILSCAPE"})
    )

    assert res["network"] == np.array(["SOILSCAPE"], dtype="U256")


def test_RollingMetrics():
    """
    Test RollingMetrics.
    """
    df = make_some_data()
    df["ref"] += np.random.rand(len(df))
    df["k1"] += np.random.rand(len(df))
    data = df[["ref", "k1"]]

    metriccalc = RollingMetrics(other_name="k1")
    dataset = metriccalc.calc_metrics(data, gpi_info=(0, 0, 0), center=False)

    # test pearson r
    ref_array = df["ref"].rolling("30d").corr(df["k1"])
    np.testing.assert_almost_equal(dataset["R"][0], ref_array.values)

    # test rmsd
    indexer = np.arange(30)[None, :] + np.arange(len(df) - 30)[:, None]
    rmsd_arr = []
    for i in range(indexer.shape[0]):
        rmsd_arr.append(
            metrics.rmsd(
                df.iloc[indexer[i, :], df.columns.get_loc("ref")].values,
                df.iloc[indexer[i, :], df.columns.get_loc("k1")].values
            )
        )

    rmsd_arr = np.array(rmsd_arr)
    np.testing.assert_almost_equal(dataset["RMSD"][0][29:-1], rmsd_arr)


###########################################################################
# Tests for new QA4SM metrics calculators


# class DummyReader:
#     def __init__(self, df, name):
#         self.data = pd.DataFrame(df[name])

#     def read(self, *args, **kwargs):
#         return self.data


def make_datasets(df):
    datasets = {}
    for key in df:
        ds = {"columns": [key], "class": DummyReader(df, key)}
        datasets[key + "_name"] = ds
    return datasets


def make_testdata_known_results():
    dr = pd.date_range("2000", "2020", freq="D")
    n = len(dr)
    x = np.ones(n) * 2
    x[10] = np.nan
    df = pd.DataFrame(
        {
            "reference": x,
            # starting with a letter > r here, so we can check how
            # reliable the sorting is
            "plus2": x + 2,
            "minus4": x - 4,
            "plus1": x + 1,
        },
        index=dr,
    )

    expected = {
        (("plus2_name", "plus2"), ("reference_name", "reference")): {
            "n_obs": n - 1,
            "R": np.nan,
            "p_R": np.nan,
            "BIAS": 2,
            "RMSD": 2,
            "mse": 4,
            "RSS": (n - 1) * 4,
            "mse_corr": 0,
            "mse_bias": 4,
            "mse_var": 0,
            "urmsd": 0,
            "gpi": 0,
            "lon": 1,
            "lat": 1,
        },
        (("minus4_name", "minus4"), ("reference_name", "reference")): {
            "n_obs": n - 1,
            "R": np.nan,
            "p_R": np.nan,
            "BIAS": -4,
            "RMSD": 4,
            "mse": 16,
            "RSS": (n - 1) * 16,
            "mse_corr": 0,
            "mse_bias": 16,
            "mse_var": 0,
            "urmsd": 0,
            "gpi": 0,
            "lon": 1,
            "lat": 1,
        },
        (("plus1_name", "plus1"), ("reference_name", "reference")): {
            "n_obs": n - 1,
            "R": np.nan,
            "p_R": np.nan,
            "BIAS": 1,
            "RMSD": 1,
            "mse": 1,
            "RSS": (n - 1) * 1,
            "mse_corr": 0,
            "mse_bias": 1,
            "mse_var": 0,
            "urmsd": 0,
            "gpi": 0,
            "lon": 1,
            "lat": 1,
        },
    }

    # make all arrays of np.float32, except n_obs, gpi (int32) and lat, lon
    # (float64)
    for ck in expected:
        for m in expected[ck]:
            if m in ["n_obs", "gpi"]:
                expected[ck][m] = np.array([expected[ck][m]],
                                           dtype=np.int32)
            elif m in ["lat", "lon"]:
                expected[ck][m] = np.array([expected[ck][m]],
                                           dtype=np.float64)
            else:
                expected[ck][m] = np.array([expected[ck][m]],
                                           dtype=np.float32)

    return make_datasets(df), expected


def make_testdata_random():
    np.random.seed(42)
    dr = pd.date_range("2000", "2020", freq="D")
    n = len(dr)
    n_datasets = 4

    # generating random correlated data
    r = 0.8
    C = np.ones((n_datasets, n_datasets)) * r
    for i in range(n_datasets):
        C[i, i] = 1
    A = np.linalg.cholesky(C)
    X = (A @ np.random.randn(n_datasets, n)).T

    ref = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]
    x3 = X[:, 3]
    ref[10] = np.nan
    x2[50] = np.nan
    df = pd.DataFrame(
        {
            "reference": ref,
            "col1": x1,
            "col2": x2,
            "zcol3": x3,
        },
        index=dr,
    )

    expected = {
        (("col1_name", "col1"), ("reference_name", "reference")): {
            "n_obs": n - 2,
            "gpi": 0,
            "lon": 1,
            "lat": 1,
        },
        (("col2_name", "col2"), ("reference_name", "reference")): {
            "n_obs": n - 2,
            "gpi": 0,
            "lon": 1,
            "lat": 1,
        },
        (("reference_name", "reference"), ("zcol3_name", "zcol3")): {
            "n_obs": n - 2,
            "gpi": 0,
            "lon": 1,
            "lat": 1,
        },
    }

    # make all arrays of np.float32, except n_obs, gpi (int32) and lat, lon
    # (float64)
    for ck in expected:
        for m in expected[ck]:
            if m in ["n_obs", "gpi"]:
                expected[ck][m] = np.array([expected[ck][m]],
                                           dtype=np.int32)
            elif m in ["lat", "lon"]:
                expected[ck][m] = np.array([expected[ck][m]],
                                           dtype=np.float64)
            else:
                expected[ck][m] = np.array([expected[ck][m]],
                                           dtype=np.float32)

    return make_datasets(df), expected


@pytest.mark.parametrize(
    "testdata_generator", [make_testdata_known_results, make_testdata_random]
)
@pytest.mark.parametrize("seas_metrics", [None, MonthsMetricsAdapter])
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide.*:RuntimeWarning")
def test_PairwiseIntercomparisonMetrics(testdata_generator, seas_metrics):
    # This test first compares the PairwiseIntercomparisonMetrics to known
    # results and then confirms that it agrees with IntercomparisonMetrics as
    # expected

    datasets, expected = testdata_generator()

    # for the pairwise intercomparison metrics it's important that we use
    # make_combined_temporal_matcher

    metrics_calculator = PairwiseIntercomparisonMetrics(
        calc_spearman=True, analytical_cis=False
    )

    if seas_metrics:
        metrics_calculator = seas_metrics(metrics_calculator)

    val = Validation(
        datasets,
        "reference_name",
        scaling=None,  # doesn't work with the constant test data
        temporal_matcher=make_combined_temporal_matcher(pd.Timedelta(6, "h")),
        metrics_calculators={(4, 2): (metrics_calculator.calc_metrics)},
    )
    results_pw = val.calc(
        [0], [1], [1], rename_cols=False, only_with_reference=True
    )

    # in results_pw, there are four entries with keys (("c1name", "c1"),
    # ("refname", "ref"), and so on.
    # Each value is a single dictionary with the values of the metrics

    expected_metrics = [
        "R",
        "p_R",
        "BIAS",
        "RMSD",
        "mse",
        "RSS",
        "mse_corr",
        "mse_bias",
        "urmsd",
        "mse_var",
        "n_obs",
        "gpi",
        "lat",
        "lon",
        "rho",
        "p_rho",
        "tau",
        "p_tau",
    ]
    seasons = ["ALL", "DJF", "MAM", "JJA", "SON"]

    if seas_metrics:
        metrics = []
        for seas in seasons:
            metrics += list(map(lambda x: (seas, x), expected_metrics))
    else:
        metrics = expected_metrics

    for key in results_pw:
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert all(map(lambda x: isinstance(x, tuple), key))
        assert isinstance(results_pw[key], dict)
        res_metrics = list(results_pw[key].keys())
        assert all([v in res_metrics for v in ["lon", "lat", "gpi"]])
        for m in metrics:
            if m in expected[key]:
                assert_equal(results_pw[key][m], expected[key][m])

    # preparation of IntercomparisonMetrics run for comparison
    ds_names = list(datasets.keys())
    with pytest.warns(DeprecationWarning):
        metrics = IntercomparisonMetrics(
            dataset_names=ds_names,
            # passing the names here explicitly, see GH issue #220
            refname="reference_name",
            other_names=ds_names[1:],
            calc_tau=True,
        )
    if seas_metrics:
        with pytest.warns(UserWarning):
            metrics = seas_metrics(metrics)

    with pytest.warns(UserWarning):
        val = Validation(
            datasets,
            "reference_name",
            scaling=None,
            temporal_matcher=None,  # use default here
            metrics_calculators={(4, 4): metrics.calc_metrics},
        )
    results = val.calc(0, 1, 1, rename_cols=False)

    # results is a dictionary with one entry and key
    # (('c1name', 'c1'), ('c2name', 'c2'), ('c3name', 'c3'), ('refname',
    # 'ref')), the value is a list of length 0, which contains a dictionary
    # with all the results, where the metrics are joined with "_between_" with
    # the combination of datasets, which is joined with "_and_", e.g. for R
    # between ``refname`` and ``c1name`` the key is
    # "R_between_refname_and_c1name"
    if seas_metrics:
        common_metrics = ["gpi", "lat", "lon"]
    else:
        common_metrics = ["n_obs", "gpi", "lat", "lon"]
    pw_metrics = list(set(expected_metrics) - set(common_metrics))
    # there's some sorting done at some point in pytesmo
    oldkey = tuple(sorted([(name, name.split("_")[0]) for name in ds_names]))
    res_old = results[oldkey]

    if seas_metrics:
        metrics = []
        pw_metrics.pop(pw_metrics.index("n_obs"))
        for seas in seasons:
            metrics += list(map(lambda x: (seas, x), pw_metrics))
    else:
        metrics = pw_metrics

    for key in results_pw:
        res = results_pw[key]
        # handle the full dataset metrics
        for m in common_metrics:
            assert_equal(res[m], res_old[m])
        # now get the metrics and compare to the right combination
        for m in metrics:
            othername = key[0][0]
            refname = key[1][0]
            if othername == "reference_name":
                # sorting might be different, see GH #220
                othername = key[1][0]
                refname = key[0][0]
            if isinstance(m, tuple):
                seas, m = m
                old_m_key = (
                    f"{seas}",
                    f"{m}_between_{refname}_and_{othername}",
                )
            else:
                seas = None
                old_m_key = f"{m}_between_{refname}_and_{othername}"
            if m == "BIAS":
                # PairwiseIntercomparisonMetrics has the result as (other,
                # ref), and therefore "bias between other and ref", compared to
                # "bias between ref and bias" in IntercomparisonMetrics
                # this is related to issue #220
                assert_equal(
                    np.abs(res[(seas, m)] if seas else res[m]),
                    np.abs(res_old[old_m_key]),
                )
            elif m == "urmsd":
                # the old implementation differs from the new implementation
                pass
            else:
                assert_equal(
                    res[(seas, m)] if seas else res[m], res_old[old_m_key]
                )


def test_PairwiseIntercomparisonMetrics_confidence_intervals():
    # tests if the correct confidence intervals are returned

    datasets, _ = make_testdata_random()
    matcher = make_combined_temporal_matcher(pd.Timedelta(6, "h"))
    val = Validation(
        datasets,
        "reference_name",
        scaling=None,  # doesn't work with the constant test data
        temporal_matcher=matcher,
        metrics_calculators={
            (4, 2): (
                PairwiseIntercomparisonMetrics(
                    calc_spearman=True,
                    calc_kendall=True,
                    analytical_cis=True,
                    bootstrap_cis=True,
                ).calc_metrics
            )
        },
    )
    results_pw = val.calc(
        [0], [1], [1], rename_cols=False,
        only_with_reference=True
    )

    metrics_with_ci = {
        "BIAS": "bias",
        "R": "pearson_r",
        "rho": "spearman_r",
        "tau": "kendall_tau",
    }
    metrics_with_bs_ci = {
        "mse": "msd",
        "mse_bias": "mse_bias",
        "RMSD": "rmsd",
        "urmsd": "ubrmsd",
        "mse_corr": "mse_corr",
        "mse_var": "mse_var",
    }

    # reconstruct dataframe
    frames = []
    for key in datasets:
        frames.append(datasets[key]["class"].data[0])
    data = pd.concat(frames, axis=1)
    data.dropna(how="any", inplace=True)

    for key in results_pw:
        othername = key[0][0]
        other_col = othername.split("_")[0]
        other = data[other_col].values
        refname = key[1][0]
        ref_col = refname.split("_")[0]
        ref = data[ref_col].values
        for metric_key in metrics_with_ci:
            lower = results_pw[key][f"{metric_key}_ci_lower"]
            upper = results_pw[key][f"{metric_key}_ci_upper"]

            # calculate manually from data
            metric_func = getattr(pairwise, metrics_with_ci[metric_key])
            m, lb, ub = with_analytical_ci(metric_func, other, ref)
            # difference due to float32 vs. float64
            assert_almost_equal(upper, ub, 6)
            assert_almost_equal(lower, lb, 6)

        for metric_key in metrics_with_bs_ci:
            lower = results_pw[key][f"{metric_key}_ci_lower"]
            upper = results_pw[key][f"{metric_key}_ci_upper"]

            # calculate manually from data
            metric_func = getattr(pairwise, metrics_with_bs_ci[metric_key])
            m, lb, ub = with_bootstrapped_ci(metric_func, other, ref)
            assert_allclose(upper, ub, rtol=1e-1, atol=1e-4)
            assert_allclose(lower, lb, rtol=1e-1, atol=1e-4)


@pytest.mark.parametrize(
    "testdata_generator", [make_testdata_known_results, make_testdata_random]
)
@pytest.mark.parametrize("seas_metrics", [None, MonthsMetricsAdapter])
def test_TripleCollocationMetrics(testdata_generator, seas_metrics):
    # tests by comparison of pairwise metrics to triplet metrics

    datasets, expected = testdata_generator()

    refname = "reference_name"
    othernames = list(datasets.keys())
    othernames.remove(refname)

    triplet_metrics_calculator = TripleCollocationMetrics(
        refname, bootstrap_cis=False
    )

    if seas_metrics:
        triplet_metrics_calculator = seas_metrics(triplet_metrics_calculator)

    matcher = make_combined_temporal_matcher(pd.Timedelta(6, "h"))

    val_triplet = Validation(
        datasets,
        "reference_name",
        scaling=None,  # doesn't work with the constant test data
        temporal_matcher=matcher,
        metrics_calculators={(4, 3): triplet_metrics_calculator.calc_metrics},
    )
    results_triplet = val_triplet.calc(
        [0], [1], [1], rename_cols=False,
        only_with_reference=True
    )

    if "col1_name" in datasets.keys():
        # we only test the TCA results with the random data, since for the
        # constant data all covariances are zero and TCA therefore doesn't
        # work.
        for metric in ["snr", "err_std", "beta"]:
            for dset in datasets:
                values = []
                dkey = (dset, datasets[dset]["columns"][0])
                for tkey in results_triplet:
                    if dkey in tkey:
                        if seas_metrics is None:
                            values.append(
                                results_triplet[tkey][(metric, dset)][0]
                            )
                        else:
                            values.append(
                                results_triplet[tkey][("DJF", metric, dset)][0]
                            )
                diff = np.abs(np.diff(values))
                assert diff.max() / values[0] < 0.1

    # check if writing to file works
    results_path = Path("__test_results")
    # if this throws, there's either some data left over from previous tests,
    # or some data is named __test_results. Remove the __test_results directory
    # from your current directory to make the test work again.
    assert not results_path.exists()
    results_path.mkdir(exist_ok=True, parents=True)
    netcdf_results_manager(results_triplet, results_path.name)
    assert results_path.exists()
    for key in results_triplet:
        fname = "_with_".join(map(lambda t: ".".join(t), key)) + ".nc"
        assert (results_path / fname).exists()
        # res = xr.open_dataset(results_path / fname)
        # for metric in ["snr", "err_std", "beta"]:
        #     for dset, _ in key:
        #         mkey = metric + "__" + dset
        #         assert mkey in res.data_vars
    shutil.rmtree(results_path)

    # now with CIs, again only for random data
    if "col1_name" in datasets.keys():
        triplet_metrics_calculator = TripleCollocationMetrics(
            refname, bootstrap_cis=True
        )

        if seas_metrics:
            triplet_metrics_calculator = seas_metrics(
                triplet_metrics_calculator
            )

        val_triplet = Validation(
            datasets,
            "reference_name",
            scaling=None,  # doesn't work with the constant test data
            temporal_matcher=matcher,
            metrics_calculators={
                (4, 3): triplet_metrics_calculator.calc_metrics
            },
        )
        results_triplet = val_triplet.calc(
            [0], [1], [1], rename_cols=False,
            only_with_reference=True
        )
        for key in results_triplet:
            for dset, _ in key:
                for metric in ["snr", "err_std", "beta"]:
                    if seas_metrics is not None:
                        lkey = ("DJF", f"{metric}_ci_lower")
                        ukey = ("DJF", f"{metric}_ci_upper")
                        mkey = ("DJF", f"{metric}")
                    else:
                        lkey = (f"{metric}_ci_lower",)
                        ukey = (f"{metric}_ci_upper",)
                        mkey = (f"{metric}",)
                    assert (*lkey, dset) in results_triplet[key]
                    assert (*ukey, dset) in results_triplet[key]
                    assert (
                            results_triplet[key][(*lkey, dset)]
                            <= results_triplet[key][(*mkey, dset)]
                    )
                    assert (
                            results_triplet[key][(*mkey, dset)]
                            <= results_triplet[key][(*ukey, dset)]
                    )


def test_temporal_matching_ascat_ismn():
    """
    This test uses a CSV file of ASCAT and ISMN data to test if the temporal
    matching within the validation works as epxected in a "real" setup.
    This only tests whether the number of observations matches, because this is
    the main thing the temporal matching influences.
    """

    # test with ASCAT and ISMN data
    here = Path(__file__).resolve().parent
    ascat = pd.read_csv(here / "ASCAT.csv", index_col=0, parse_dates=True)
    ismn = pd.read_csv(here / "ISMN.csv", index_col=0, parse_dates=True)
    dfs = {"ASCAT": ascat, "ISMN": ismn}
    columns = {"ASCAT": "sm", "ISMN": "soil_moisture"}
    refname = "ISMN"
    window = pd.Timedelta(12, "h")

    old_matcher = BasicTemporalMatching().combinatory_matcher
    new_matcher = make_combined_temporal_matcher(window)

    datasets = {}
    for key in ["ISMN", "ASCAT"]:
        all_columns = list(dfs[key].columns)
        ds = {
            "columns": [columns[key]],
            "class": DummyReader(dfs[key], all_columns),
        }
        datasets[key] = ds

    new_val = Validation(
        datasets,
        refname,
        scaling=None,  # doesn't work with the constant test data
        temporal_matcher=new_matcher,
        metrics_calculators={
            (2, 2): PairwiseIntercomparisonMetrics().calc_metrics
        },
    )
    new_results = new_val.calc(
        0, 1, 1, rename_cols=False, only_with_reference=True
    )

    # old setup
    ds_names = list(datasets.keys())
    with pytest.warns(DeprecationWarning):
        metrics = IntercomparisonMetrics(
            dataset_names=ds_names,
            # passing the names here explicitly, see GH issue #220
            refname=refname,
            other_names=ds_names[1:],
            calc_tau=True,
        )
    old_val = Validation(
        datasets,
        refname,
        scaling=None,  # doesn't work with the constant test data
        temporal_matcher=old_matcher,
        metrics_calculators={(2, 2): metrics.calc_metrics},
    )
    old_results = old_val.calc(0, 1, 1, rename_cols=False)

    old_key = (("ASCAT", "sm"), ("ISMN", "soil_moisture"))
    new_key = (("ASCAT", "sm"), ("ISMN", "soil_moisture"))

    assert old_results[old_key]["n_obs"] == new_results[new_key]["n_obs"]


def test_TripleCollocationMetrics_failure():
    """
    Test if TripleCollocationMetrics returns the correct status attribute if
    there is not enough data for bootstrapping.
    """
    df = make_some_data()
    data = df.iloc[0:50][["ref", "k1", "k2"]]

    triplet_metrics_calculator = TripleCollocationMetrics(
        "ref", bootstrap_cis=True
    )
    res = triplet_metrics_calculator.calc_metrics(data, gpi_info=(0, 0, 0))
    assert res["status"] == eh.METRICS_CALCULATION_FAILED

# def test_sorting_issue():
# GH #220
# might be a good start for fixing the issue

# dr = pd.date_range("2000", "2020", freq="D")
# n = len(dr)
# x = np.ones(n) * 2
# df = pd.DataFrame(
#     {
#         "reference": x,
#         "zplus2": x + 2,
#         "plus1": x + 1,
#     },
#     index=dr
# )

# datasets = {
#     key + "_name": {"columns": [key], "class": DummyReader(df, key)}
#     for key in df
# }

# val = Validation(
#     datasets,
#     "reference_name",
#     scaling=None,  # doesn't work with the constant test data
#     # temporal_matcher=None,
#     temporal_matcher=make_combined_temporal_matcher(
#          pd.Timedelta(6, "h")
#     ),
#     metrics_calculators={
#         (3, 2): PairwiseIntercomparisonMetrics().calc_metrics
#     }
# )
# results = val.calc(1, 1, 1)

# assert 0
