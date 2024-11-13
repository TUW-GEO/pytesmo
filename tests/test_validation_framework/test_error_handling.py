import pandas as pd
import numpy as np
import pytest

from pytesmo.validation_framework.validation import Validation
from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics,
)
from pytesmo.validation_framework.temporal_matchers import (
    make_combined_temporal_matcher,
)
import pytesmo.validation_framework.error_handling as eh
from pytesmo.validation_framework.data_manager import DataManager

from .utils import create_datasets


@pytest.mark.filterwarnings("ignore:Not enough observations.*:UserWarning")
def test_error_handling_empty_df():
    # This tests whether error handling works if one of the datasets consists
    # of an empty dataframe.
    # The options "ignore" and "deprecated" should both return an empty
    # dictionary in this case, while the others handle the error
    # this tests whether with the "deprecated" and "ignore" option of error
    # handling, an empty dictionary is returned if one of the datasets is just
    # an empty dataframe, and therefore temporal matching fails
    n_datasets = 5
    npoints = 5
    nsamples = 100

    datasets = create_datasets(n_datasets, npoints, nsamples, missing="empty")
    metric_calculator = PairwiseIntercomparisonMetrics()

    val = Validation(
        datasets,
        spatial_ref="0-ERA5",
        metrics_calculators={(n_datasets, 2): metric_calculator.calc_metrics},
        temporal_matcher=make_combined_temporal_matcher(
            pd.Timedelta(12, "h")),
    )
    gpis = list(range(npoints))
    args = (gpis, gpis, gpis)
    kwargs = dict(rename_cols=False, only_with_reference=True)

    # 'raise' should raise an error
    with pytest.raises(eh.NoTempMatchedDataError):
        with pytest.warns(UserWarning, match="No data for dataset 4-missing"):
            results = val.calc(*args, **kwargs, handle_errors="raise")

    # 'ignore' should modify the status code, but not raise an error
    with pytest.warns(UserWarning, match="No data for dataset 4-missing"):
        results = val.calc(*args, **kwargs, handle_errors="ignore")
    for key in results:
        for metric in results[key]:
            assert len(results[key][metric]) == npoints
            if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                assert np.all(np.isnan(results[key][metric]))
        assert np.all(results[key]["status"] == eh.NO_TEMP_MATCHED_DATA)


@pytest.mark.filterwarnings("ignore:Not enough observations.*:UserWarning")
def test_error_handling_nodata():
    # this tests if we get the NoGpiDataError if one dataset doesn't have any
    # values.  Here we use only 2 datasets, otherwise the third one will be
    # available for a comparison and we will again end up with a
    # NoTempMatchedDataError
    n_datasets = 2
    npoints = 5
    nsamples = 100

    datasets = create_datasets(n_datasets, npoints, nsamples, missing="nodata")
    metric_calculator = PairwiseIntercomparisonMetrics()

    val = Validation(
        datasets,
        spatial_ref="0-ERA5",
        metrics_calculators={(n_datasets, 2): metric_calculator.calc_metrics},
        temporal_matcher=make_combined_temporal_matcher(
            pd.Timedelta(12, "h")),
    )
    gpis = list(range(npoints))
    args = (gpis, gpis, gpis)
    kwargs = dict(rename_cols=False, only_with_reference=True)

    # 'raise' should raise an error
    with pytest.raises(eh.NoGpiDataError):
        with pytest.warns(UserWarning, match="No data for dataset 1-missing"):
            results = val.calc(*args, **kwargs, handle_errors="raise")

    # 'ignore' should modify the status code, but not raise an error
    with pytest.warns(UserWarning, match="No data for dataset 1-missing"):
        results = val.calc(*args, **kwargs, handle_errors="ignore")
    for key in results:
        for metric in results[key]:
            assert len(results[key][metric]) == npoints
            if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                assert np.all(np.isnan(results[key][metric]))
        assert np.all(results[key]["status"] == eh.NO_GPI_DATA)


@pytest.mark.filterwarnings("ignore:Not enough observations.*:UserWarning")
def test_error_handling_not_enough_data():
    # This tests if we get a proper warning if we have not enough data to
    # calculate correlations (nsamples = 5). In this case, the behaviour of all
    # options should be the same (setting the correct status, but nothing
    # otherwise)
    n_datasets = 5
    npoints = 5
    nsamples = 5

    datasets = create_datasets(n_datasets, npoints, nsamples)
    metric_calculator = PairwiseIntercomparisonMetrics()

    val = Validation(
        datasets,
        spatial_ref="0-ERA5",
        metrics_calculators={(n_datasets, 2): metric_calculator.calc_metrics},
        temporal_matcher=make_combined_temporal_matcher(
            pd.Timedelta(12, "h")),
    )
    gpis = list(range(npoints))
    args = (gpis, gpis, gpis)
    kwargs = dict(rename_cols=False, only_with_reference=True)

    for handle_errors in ["ignore", "raise"]:
        with pytest.warns(
                UserWarning,
                match="Not enough observations to calculate metrics."
        ):
            results = val.calc(*args, **kwargs,
                               handle_errors=handle_errors)
        for key in results:
            for metric in results[key]:
                assert len(results[key][metric]) == npoints
                if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                    assert np.all(np.isnan(results[key][metric]))
            assert np.all(results[key]["status"] == eh.INSUFFICIENT_DATA)


def test_error_handling_ok():
    # everything should work out nicely here
    n_datasets = 5
    npoints = 5
    nsamples = 100

    datasets = create_datasets(n_datasets, npoints, nsamples)
    metric_calculator = PairwiseIntercomparisonMetrics()

    val = Validation(
        datasets,
        spatial_ref="0-ERA5",
        metrics_calculators={(n_datasets, 2): metric_calculator.calc_metrics},
        temporal_matcher=make_combined_temporal_matcher(
            pd.Timedelta(12, "h")),
    )
    gpis = list(range(npoints))
    args = (gpis, gpis, gpis)
    kwargs = dict(rename_cols=False, only_with_reference=True)

    for handle_errors in ["ignore", "raise"]:
        results = val.calc(*args, **kwargs, handle_errors=handle_errors)
        for key in results:
            for metric in results[key]:
                assert len(results[key][metric]) == npoints
                if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                    assert np.all(np.isfinite(results[key][metric]))
            assert np.all(results[key]["status"] == eh.OK)


@pytest.mark.filterwarnings("ignore:Not enough observations.*:UserWarning")
def test_error_handling_scaling_failed():
    # This tests whether a scaling error is raised if the scaling fails due to
    # insufficient data.
    n_datasets = 5
    npoints = 5
    nsamples = 100

    class BadScaler:
        def scale(self, data, ref_idx, gpi_info):
            raise ValueError("This is a test.")

    datasets = create_datasets(n_datasets, npoints, nsamples)
    metric_calculator = PairwiseIntercomparisonMetrics()

    val = Validation(
        datasets,
        scaling=BadScaler(),
        spatial_ref="0-ERA5",
        metrics_calculators={(n_datasets, 2): metric_calculator.calc_metrics},
        temporal_matcher=make_combined_temporal_matcher(
            pd.Timedelta(12, "h")),
    )
    gpis = list(range(npoints))
    args = (gpis, gpis, gpis)
    kwargs = dict(rename_cols=False, only_with_reference=True)

    # test if raising raises
    with pytest.raises(eh.ScalingError, match="Scaling failed"):
        results = val.calc(*args, **kwargs, handle_errors="raise")

    # test if ignore returns code
    results = val.calc(*args, **kwargs, handle_errors="ignore")
    for key in results:
        for metric in results[key]:
            assert len(results[key][metric]) == npoints
            if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                assert np.all(np.isnan(results[key][metric]))
        assert np.all(results[key]["status"] == eh.SCALING_FAILED)


@pytest.mark.filterwarnings("ignore:Not enough observations.*:UserWarning")
def test_error_handling_datamanager_failed():
    # This tests whether a scaling error is raised if the scaling fails due to
    # insufficient data.
    n_datasets = 5
    npoints = 5
    nsamples = 100

    datasets = create_datasets(n_datasets, npoints, nsamples)

    spatial_ref = "0-ERA5"
    data_manager = DataManager(datasets, spatial_ref, None)

    def bad_get_data(*args):
        raise ValueError("This is a test.")

    data_manager.get_data = bad_get_data

    metric_calculator = PairwiseIntercomparisonMetrics()

    val = Validation(
        data_manager,
        spatial_ref=spatial_ref,
        metrics_calculators={(n_datasets, 2): metric_calculator.calc_metrics},
        temporal_matcher=make_combined_temporal_matcher(
            pd.Timedelta(12, "h")),
    )
    gpis = list(range(npoints))
    args = (gpis, gpis, gpis)
    kwargs = dict(rename_cols=False, only_with_reference=True)

    # test if raising raises
    with pytest.raises(eh.DataManagerError, match="Getting the data for gpi"):
        results = val.calc(*args, **kwargs, handle_errors="raise")

    # test if ignore returns code
    results = val.calc(*args, **kwargs, handle_errors="ignore")
    for key in results:
        for metric in results[key]:
            assert len(results[key][metric]) == npoints
            if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                assert np.all(np.isnan(results[key][metric]))
        assert np.all(results[key]["status"] == eh.DATA_MANAGER_FAILED)


@pytest.mark.filterwarnings("ignore:Not enough observations.*:UserWarning")
def test_error_handling_temp_matching_failed():
    # This tests whether a TemporalMatchingError is raised if the matching
    # fails
    n_datasets = 5
    npoints = 5
    nsamples = 100

    def bad_matching(*args, **kwargs):
        raise ValueError("This is a test.")

    datasets = create_datasets(n_datasets, npoints, nsamples)
    metric_calculator = PairwiseIntercomparisonMetrics()

    val = Validation(
        datasets,
        spatial_ref="0-ERA5",
        metrics_calculators={(n_datasets, 2): metric_calculator.calc_metrics},
        temporal_matcher=bad_matching,
    )
    gpis = list(range(npoints))
    args = (gpis, gpis, gpis)
    kwargs = dict(rename_cols=False, only_with_reference=True)

    # raise should raise a TemporalMatchingError
    with pytest.raises(eh.TemporalMatchingError,
                       match="Temporal matching failed"):
        results = val.calc(*args, **kwargs, handle_errors="raise")

    # ignore should just log the correct return code
    results = val.calc(*args, **kwargs, handle_errors="ignore")
    for key in results:
        for metric in results[key]:
            assert len(results[key][metric]) == npoints
            if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                assert np.all(np.isnan(results[key][metric]))
        assert np.all(results[key]["status"] == eh.TEMPORAL_MATCHING_FAILED)


@pytest.mark.filterwarnings("ignore:Not enough observations.*:UserWarning")
def test_error_handling_metrics_calculation_failed():
    # This tests whether a MetricsCalculationError is raised if metrics
    # calculation fails
    n_datasets = 5
    npoints = 5
    nsamples = 100

    def bad_metrics(data, gpi_info):
        if len(data) == 0:
            return {"status": np.int32([-1])}
        raise ValueError("This is a test.")

    datasets = create_datasets(n_datasets, npoints, nsamples)

    val = Validation(
        datasets,
        spatial_ref="0-ERA5",
        metrics_calculators={(n_datasets, 2): bad_metrics},
        temporal_matcher=make_combined_temporal_matcher(
            pd.Timedelta(12, "h")),
    )
    gpis = list(range(npoints))
    args = (gpis, gpis, gpis)
    kwargs = dict(rename_cols=False, only_with_reference=True)

    # raise should raise the MetricsCalculationError
    with pytest.raises(eh.MetricsCalculationError,
                       match="Metrics calculation failed"):
        results = val.calc(*args, **kwargs, handle_errors="raise")

    # ignore should just log the correct return code
    results = val.calc(*args, **kwargs, handle_errors="ignore")
    for key in results:
        for metric in results[key]:
            assert len(results[key][metric]) == npoints
        assert np.all(results[key]["status"] == eh.METRICS_CALCULATION_FAILED)
