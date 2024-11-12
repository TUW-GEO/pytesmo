import pytest

from pytesmo.validation_framework.adapters import TimestampAdapter

"""
Test for the adapters.
"""

from datetime import datetime
import numpy as np
import numpy.testing as nptest
import os
import pandas as pd
import warnings

from pytesmo.validation_framework.adapters import (MaskingAdapter,
                                                   AdvancedMaskingAdapter,
                                                   ColumnCombineAdapter)
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
        data_masked = ds_mask.read()

        nptest.assert_almost_equal(
            data_masked["x"].values,
            np.concatenate(
                [np.ones((10), dtype=bool),
                 np.zeros((10), dtype=bool)]),
        )

        if col is None:
            nptest.assert_almost_equal(data_masked["y"].values,
                                       np.ones((20), dtype=bool))


def test_self_masking_adapter():
    ref_x = np.arange(10)
    ref_y = np.arange(10) * 0.5
    ds = TestDataset("", n=20)

    ds_mask = SelfMaskingAdapter(ds, "<", 10, "x")
    data_masked = ds_mask.read()

    nptest.assert_almost_equal(data_masked["x"].values, ref_x)
    nptest.assert_almost_equal(data_masked["y"].values, ref_y)


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
    data_masked = ds_mask.read()

    nptest.assert_almost_equal(data_masked["x"].values, ref_x)
    nptest.assert_almost_equal(data_masked["y"].values, ref_y)

    # 9 is not a valid operator, should raise an exception
    with pytest.raises(ValueError):
        ds_mask = AdvancedMaskingAdapter(
            ds,
            [
                ("x", 9, 5),
            ],
        )
        _ = ds_mask.read()


def test_advanced_masking_adapter_nans_ignored():
    ds = TestDataset("", n=20)
    # introduce nan
    ts = ds.read()
    ts.iloc[7, ts.columns.get_loc('x')] = np.nan

    def _read():
        return ts

    setattr(ds, "read", _read)

    # the NaN in the flag field (x) is filtered out normally
    ds_mask = AdvancedMaskingAdapter(
        ds,
        [
            ("x", ">=", 5),
            ("x", "<", 15),
        ],
    )

    data_masked = ds_mask.read()

    ref_x = np.array([5., 6., 8., 9., 10., 11., 12., 13., 14.])
    ref_y = ref_x * 0.5

    nptest.assert_almost_equal(data_masked["x"].values, ref_x)
    nptest.assert_almost_equal(data_masked["y"].values, ref_y)

    # the NaN is now ignored
    ds_mask = AdvancedMaskingAdapter(
        ds,
        [
            ("x", ">=", 5),
            ("x", "<", 15),
        ],
        ignore_nans=True,
    )

    data_masked = ds_mask.read()

    ref_x = np.array([5., 6., np.nan, 8., 9., 10., 11., 12., 13., 14.])
    ref_y = np.arange(5, 15) * 0.5

    nptest.assert_almost_equal(data_masked["x"].values, ref_x)
    nptest.assert_almost_equal(data_masked["y"].values, ref_y)


def test_anomaly_adapter():
    ds = TestDataset("", n=20)
    ds_anom = AnomalyAdapter(ds)
    data_anom = ds_anom.read()
    nptest.assert_almost_equal(data_anom["x"].values[0], -8.5)
    nptest.assert_almost_equal(data_anom["y"].values[0], -4.25)


def test_anomaly_adapter_one_column():
    ds = TestDataset("", n=20)
    ds_anom = AnomalyAdapter(ds, columns=["x"])
    data_anom = ds_anom.read()
    nptest.assert_almost_equal(data_anom["x"].values[0], -8.5)
    nptest.assert_almost_equal(data_anom["y"].values[0], 0)


def test_anomaly_clim_adapter():
    ds = TestDataset("", n=20)
    ds_anom = AnomalyClimAdapter(ds)
    data_anom = ds_anom.read()
    nptest.assert_almost_equal(data_anom["x"].values[4], -5.5)
    nptest.assert_almost_equal(data_anom["y"].values[4], -2.75)


def test_anomaly_clim_adapter_one_column():
    ds = TestDataset("", n=20)
    ds_anom = AnomalyClimAdapter(ds, columns=["x"])
    data_anom = ds_anom.read()
    nptest.assert_almost_equal(data_anom["x"].values[4], -5.5)
    nptest.assert_almost_equal(data_anom["y"].values[4], 2)


def test_adapters_custom_fct_name():
    def assert_all_read_fcts(reader):
        assert (np.all(reader.read() == reader.read()))
        assert (np.all(reader.read() == reader.alias_read()))

    base = TestDataset("", n=20)
    assert_all_read_fcts(base)
    sma = SelfMaskingAdapter(
        base, '>=', 5, column_name='y', read_name='alias_read')
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
    data = ascat_anom.read(12.891455, 45.923004)
    assert data is not None
    assert np.any(data["sm"].values != 0)

    ascat_self = SelfMaskingAdapter(ascat_reader, ">", 0, "sm")
    data2 = ascat_self.read(12.891455, 45.923004)
    assert data2 is not None
    assert np.all(data2["sm"].values > 0)
    data2 = ascat_self.read(12.891455, 45.923004)
    assert data2 is not None
    assert np.all(data2["sm"].values > 0)

    ascat_mask = MaskingAdapter(ascat_reader, ">", 0, "sm")
    data3 = ascat_mask.read(12.891455, 45.923004)
    assert data3 is not None
    assert np.any(data3["sm"].values)
    data3 = ascat_mask.read(12.891455, 45.923004)
    assert data3 is not None
    assert np.any(data3["sm"].values)

    ascat_clim = AnomalyClimAdapter(ascat_reader, columns=["sm"])
    data4 = ascat_clim.read(12.891455, 45.923004)
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
                datetime(2007, 1, 1, 0), "2007-01-05", freq="D", tz="UTC"),
        )

    def read_ts(self, *args, **kwargs):
        return self.read(*args, **kwargs)


@pytest.mark.filterwarnings("ignore:Dropping timezone information:UserWarning")
def test_timezone_removal():
    tz_reader = TestTimezoneReader()

    reader_anom = AnomalyAdapter(tz_reader, window_size=35, columns=["data"])
    assert reader_anom.read(0) is not None

    reader_self = SelfMaskingAdapter(tz_reader, ">", 0, "data")
    assert reader_self.read(0) is not None

    reader_mask = MaskingAdapter(tz_reader, ">", 0, "data")
    assert reader_mask.read(0) is not None

    reader_clim = AnomalyClimAdapter(tz_reader, columns=["data"])
    assert reader_clim.read(0) is not None


def test_column_comb_adapter():
    ds = TestDataset("", n=20)
    orig = ds.read()
    ds_adapted = ColumnCombineAdapter(
        ds,
        func=pd.Series.mean,
        columns=["x", "y"],
        func_kwargs={'skipna': True},
        new_name='xy_mean')
    ds_mean = ds_adapted.read()

    nptest.assert_equal(ds_mean["x"].values, orig["x"].values)
    nptest.assert_equal(ds_mean["y"].values, orig["y"].values)
    nptest.assert_equal(ds_mean["xy_mean"].values,
                        (ds_mean["x"] + ds_mean["y"]) / 2.)

    # try an empty DataFrame
    def read_empty():
        return pd.DataFrame(columns=['x', 'y'])

    setattr(ds, "read", read_empty)
    ds_adapted = ColumnCombineAdapter(
        ds,
        func=pd.DataFrame.mean,
        columns=["x", "y"],
        func_kwargs={'skipna': True},
        new_name='xy_mean')

    pd.testing.assert_frame_equal(ds_adapted.read(),
                                  pd.DataFrame(columns=['x', 'y', 'xy_mean']))


@pytest.mark.filterwarnings(
    "ignore:The input DataFrame is either empty or has.*:UserWarning")
def test_timestamp_adapter():
    ds = TestDataset("", n=20)

    # Simple case
    # ================

    index = np.arange('2005-02', '2005-03', dtype='datetime64[D]')
    sm_var = np.random.randn(*index.shape)
    time_offset_field = np.random.normal(
        loc=1000.0, scale=1.0, size=index.shape).astype(int)

    def _read():
        return pd.DataFrame(
            data=np.array([sm_var, time_offset_field]).transpose(),
            columns=["sm", "offset"],
            index=index)

    setattr(ds, "read", _read)
    origin = ds.read().drop(columns="offset")

    adapted_ds = TimestampAdapter(
        ds, time_offset_fields="offset", time_units="s")
    adapted = adapted_ds.read()

    # Date should be unchanges as we are using a ~1000 sec offset
    assert (origin.index.date == adapted.index.date).all()
    # The offset is expressed in seconds
    assert origin.index[0] + np.timedelta64(time_offset_field[0],
                                            "s") == adapted.index[0]
    # The dataframe is integral
    assert (origin.columns == adapted.columns).all()

    adapted_ds = TimestampAdapter(
        ds, time_offset_fields="offset", time_units="m")
    adapted = adapted_ds.read()

    # The offset is expressed in minutes
    assert origin.index[0] + np.timedelta64(time_offset_field[0],
                                            "m") == adapted.index[0]

    # This time we do not drop the columns
    origin = ds.read()
    adapted_ds = TimestampAdapter(
        ds, time_offset_fields="offset", time_units="s", drop_original=False)
    adapted = adapted_ds.read()

    # The dataframe is integral
    assert (origin.columns == adapted.columns).all()

    # test use of new column
    # -----------------------
    adapted_ds = TimestampAdapter(
        ds,
        time_offset_fields="offset",
        time_units="s",
        replace_index=False,
        output_field="exact_timestamp")
    adapted = adapted_ds.read()
    assert (adapted.columns == ["sm", "exact_timestamp"]).all()

    # test NaNs in offset and generic time
    # -----------------------
    index = np.arange('2005-02', '2005-03', dtype='datetime64[D]')
    index[4] = np.datetime64("NaT")
    sm_var = np.random.randn(*index.shape)
    time_offset_field = np.random.normal(
        loc=1000.0, scale=1.0, size=index.shape)
    time_offset_field[7] = np.nan

    def _read_nans():
        return pd.DataFrame(
            data=np.array([sm_var, time_offset_field]).transpose(),
            columns=["sm", "offset"],
            index=index)

    setattr(ds, "read", _read_nans)
    origin = ds.read()

    adapted_ds = TimestampAdapter(
        ds, time_offset_fields="offset", time_units="s")
    adapted = adapted_ds.read()

    # One index (NaT) value should be dropped
    assert len(adapted.index) == len(origin.index) - 1
    # The Nan offset should be interpreted as 0
    assert adapted.index[6] == datetime(2005, 2, 8, 0, 0)

    # test all NaNs in dataframe
    # -----------------------
    index = np.arange('2005-02', '2005-03', dtype='datetime64[D]')
    index[:] = np.datetime64("NaT")
    sm_var = np.random.randn(*index.shape)
    time_offset_field = np.random.normal(
        loc=1000.0, scale=1.0, size=index.shape)
    time_offset_field[7] = np.nan

    def _read_all_nans():
        return pd.DataFrame(
            data=np.array([sm_var, time_offset_field]).transpose(),
            columns=["sm", "offset"],
            index=index)

    setattr(ds, "read", _read_all_nans)
    origin = ds.read()

    adapted_ds = TimestampAdapter(
        ds, time_offset_fields="offset", time_units="s")
    adapted = adapted_ds.read()

    # The original should be returned
    pd.testing.assert_frame_equal(origin, adapted)

    # test empty Dataframe
    # -----------------------

    def _read_empty():
        return pd.DataFrame(columns=["sm", "offset"], )

    setattr(ds, "read", _read_empty)
    origin = ds.read()

    adapted_ds = TimestampAdapter(
        ds, time_offset_fields="offset", time_units="s")
    adapted = adapted_ds.read()

    # The original should be returned
    assert adapted.empty
    assert (adapted.columns == ["sm", "offset"]).all()

    # Complex case
    # ================
    base_time = np.arange(100, 200)
    index = base_time.copy()
    sm_var = np.random.randn(*index.shape)
    time_offset_field_min = np.random.normal(
        loc=40.0, scale=1.0, size=index.shape).astype(int)
    time_offset_field_sec = np.random.normal(
        loc=1000.0, scale=1.0, size=index.shape).astype(int)

    def _read_complex():
        return pd.DataFrame(
            data=np.array([
                sm_var, base_time, time_offset_field_min, time_offset_field_sec
            ]).transpose(),
            columns=["sm", "base_time", "offset_min", "offset_sec"],
            index=index)

    setattr(ds, "read", _read_complex)
    origin = ds.read()

    adapted_ds = TimestampAdapter(
        ds,
        time_offset_fields=["offset_min", "offset_sec"],
        time_units=["m", "s"],
        base_time_field="base_time",
        base_time_reference="2005-02-01")

    adapted = adapted_ds.read()

    should_be = origin.apply(
        lambda row: np.datetime64("2005-02-01") + np.timedelta64(
            int(row["base_time"]), "D") + np.timedelta64(
            int(row["offset_min"]), "m") + np.timedelta64(
            int(row["offset_sec"]), "s"),
        axis=1).values

    assert (adapted.index.values == should_be).all()
