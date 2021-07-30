import pytest

"""
Test for the adapters.
"""

from datetime import datetime
import numpy as np
import numpy.testing as nptest
import os
import pandas as pd
import warnings

from pytesmo.validation_framework.adapters import (
    MaskingAdapter,
    AdvancedMaskingAdapter,
    ColumnCombineAdapter
)
from pytesmo.validation_framework.adapters import SelfMaskingAdapter
from pytesmo.validation_framework.adapters import AnomalyAdapter
from pytesmo.validation_framework.adapters import AnomalyClimAdapter
from tests.test_validation_framework.test_datasets import TestDataset

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from ascat.read_native.cdr import AscatGriddedNcTs


def test_masking_adapter():
    for col in (None, "x"):
        ds = TestDataset("", n=20)
        ds_mask = MaskingAdapter(ds, "<", 10, col)
        data_masked = ds_mask.read_ts()
        data_masked2 = ds_mask.read()

        nptest.assert_almost_equal(
            data_masked["x"].values,
            np.concatenate(
                [np.ones((10), dtype=bool), np.zeros((10), dtype=bool)]
            ),
        )
        nptest.assert_almost_equal(
            data_masked2["x"].values,
            np.concatenate(
                [np.ones((10), dtype=bool), np.zeros((10), dtype=bool)]
            ),
        )

        if col is None:
            nptest.assert_almost_equal(
                data_masked["y"].values, np.ones((20), dtype=bool)
            )
            nptest.assert_almost_equal(
                data_masked2["y"].values, np.ones((20), dtype=bool)
            )


def test_self_masking_adapter():
    ref_x = np.arange(10)
    ref_y = np.arange(10) * 0.5
    ds = TestDataset("", n=20)

    ds_mask = SelfMaskingAdapter(ds, "<", 10, "x")
    data_masked = ds_mask.read_ts()
    data_masked2 = ds_mask.read()

    nptest.assert_almost_equal(data_masked["x"].values, ref_x)
    nptest.assert_almost_equal(data_masked2["x"].values, ref_x)
    nptest.assert_almost_equal(data_masked["y"].values, ref_y)
    nptest.assert_almost_equal(data_masked2["y"].values, ref_y)


def my_bitmasking(a, b):
    return a.astype(int) & b == b


def test_advanced_masking_adapter():
    ref_x = np.arange(5, 15, 2)
    ref_y = np.arange(5, 15, 2) * 0.5
    ds = TestDataset("", n=20)

    ds_mask = AdvancedMaskingAdapter(
        ds,
        [
            ("x", ">=", 5),
            ("x", "<", 15),
            ("x", my_bitmasking, 1),
        ],
    )
    data_masked = ds_mask.read_ts()
    data_masked2 = ds_mask.read()

    nptest.assert_almost_equal(data_masked["x"].values, ref_x)
    nptest.assert_almost_equal(data_masked2["x"].values, ref_x)
    nptest.assert_almost_equal(data_masked["y"].values, ref_y)
    nptest.assert_almost_equal(data_masked2["y"].values, ref_y)

    # 9 is not a valid operator, should raise an exception
    with pytest.raises(ValueError):
        ds_mask = AdvancedMaskingAdapter(
            ds,
            [
                ("x", 9, 5),
            ],
        )
        data_masked = ds_mask.read_ts()


def test_anomaly_adapter():
    ds = TestDataset("", n=20)
    ds_anom = AnomalyAdapter(ds)
    data_anom = ds_anom.read_ts()
    data_anom2 = ds_anom.read()
    nptest.assert_almost_equal(data_anom["x"].values[0], -8.5)
    nptest.assert_almost_equal(data_anom["y"].values[0], -4.25)
    nptest.assert_almost_equal(data_anom2["x"].values[0], -8.5)
    nptest.assert_almost_equal(data_anom2["y"].values[0], -4.25)


def test_anomaly_adapter_one_column():
    ds = TestDataset("", n=20)
    ds_anom = AnomalyAdapter(ds, columns=["x"])
    data_anom = ds_anom.read()
    nptest.assert_almost_equal(data_anom["x"].values[0], -8.5)
    nptest.assert_almost_equal(data_anom["y"].values[0], 0)


def test_anomaly_clim_adapter():
    ds = TestDataset("", n=20)
    ds_anom = AnomalyClimAdapter(ds)
    data_anom = ds_anom.read_ts()
    data_anom2 = ds_anom.read()
    nptest.assert_almost_equal(data_anom["x"].values[4], -5.5)
    nptest.assert_almost_equal(data_anom["y"].values[4], -2.75)
    nptest.assert_almost_equal(data_anom2["x"].values[4], -5.5)
    nptest.assert_almost_equal(data_anom2["y"].values[4], -2.75)


def test_anomaly_clim_adapter_one_column():
    ds = TestDataset("", n=20)
    ds_anom = AnomalyClimAdapter(ds, columns=["x"])
    data_anom = ds_anom.read_ts()
    nptest.assert_almost_equal(data_anom["x"].values[4], -5.5)
    nptest.assert_almost_equal(data_anom["y"].values[4], 2)


def test_adapters_custom_fct_name():
    def assert_all_read_fcts(reader):
        assert(np.all(reader.read() == reader.read_ts()))
        assert(np.all(reader.read() == reader.alias_read()))

    base = TestDataset("", n=20)
    assert_all_read_fcts(base)
    sma = SelfMaskingAdapter(base, '>=', 5, column_name='y',
                             read_name='alias_read')
    assert_all_read_fcts(sma)
    smanom = AnomalyAdapter(sma, read_name='alias_read')
    assert_all_read_fcts(smanom)


# the ascat reader gives back an ascat timeseries instead of a dataframe -
# make sure the adapters can deal with that
def test_adapters_with_ascat():
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

    ascat_reader = AscatGriddedNcTs(
        ascat_data_folder,
        "TUW_METOP_ASCAT_WARP55R22_{:04d}",
        grid_filename=grid_fname,
    )

    ascat_anom = AnomalyAdapter(ascat_reader, window_size=35, columns=["sm"])
    data = ascat_anom.read_ts(12.891455, 45.923004)
    assert data is not None
    assert np.any(data["sm"].values != 0)
    data = ascat_anom.read(12.891455, 45.923004)
    assert data is not None
    assert np.any(data["sm"].values != 0)

    ascat_self = SelfMaskingAdapter(ascat_reader, ">", 0, "sm")
    data2 = ascat_self.read_ts(12.891455, 45.923004)
    assert data2 is not None
    assert np.all(data2["sm"].values > 0)
    data2 = ascat_self.read(12.891455, 45.923004)
    assert data2 is not None
    assert np.all(data2["sm"].values > 0)

    ascat_mask = MaskingAdapter(ascat_reader, ">", 0, "sm")
    data3 = ascat_mask.read_ts(12.891455, 45.923004)
    assert data3 is not None
    assert np.any(data3["sm"].values)
    data3 = ascat_mask.read(12.891455, 45.923004)
    assert data3 is not None
    assert np.any(data3["sm"].values)

    ascat_clim = AnomalyClimAdapter(ascat_reader, columns=["sm"])
    data4 = ascat_clim.read_ts(12.891455, 45.923004)
    assert data4 is not None
    assert np.any(data["sm"].values != 0)
    data4 = ascat_clim.read(12.891455, 45.923004)
    assert data4 is not None
    assert np.any(data["sm"].values != 0)


class TestTimezoneReader(object):
    def read(self, *args, **kwargs):
        data = np.arange(5.0)
        data[3] = np.nan
        return pd.DataFrame(
            {"data": data},
            index=pd.date_range(
                datetime(2007, 1, 1, 0), "2007-01-05", freq="D", tz="UTC"
            ),
        )

    def read_ts(self, *args, **kwargs):
        return self.read(*args, **kwargs)


@pytest.mark.filterwarnings(
    "ignore:Dropping timezone information:UserWarning"
)
def test_timezone_removal():
    tz_reader = TestTimezoneReader()

    reader_anom = AnomalyAdapter(tz_reader, window_size=35, columns=["data"])
    assert reader_anom.read_ts(0) is not None

    reader_self = SelfMaskingAdapter(tz_reader, ">", 0, "data")
    assert reader_self.read_ts(0) is not None

    reader_mask = MaskingAdapter(tz_reader, ">", 0, "data")
    assert reader_mask.read_ts(0) is not None

    reader_clim = AnomalyClimAdapter(tz_reader, columns=["data"])
    assert reader_clim.read_ts(0) is not None


def test_column_comb_adapter():
    ds = TestDataset("", n=20)
    orig = ds.read()
    ds_adapted = ColumnCombineAdapter(
        ds, func=pd.DataFrame.mean, columns=["x", "y"],
        func_kwargs={'skipna': True}, new_name='xy_mean')
    ds_mean1 = ds_adapted.read_ts()
    ds_mean2 = ds_adapted.read()

    for ds_mean in [ds_mean1, ds_mean2]:
        nptest.assert_equal(ds_mean["x"].values, orig["x"].values)
        nptest.assert_equal(ds_mean["y"].values, orig["y"].values)
        nptest.assert_equal(ds_mean["xy_mean"].values,
                            (ds_mean["x"] + ds_mean["y"]) / 2.)
