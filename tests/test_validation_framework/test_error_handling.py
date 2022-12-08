import pytest

from pytesmo.validation_framework.validation import Validation
from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics)
from pytesmo.validation_framework.temporal_matchers import (
    make_combined_temporal_matcher)
import pytesmo.validation_framework.error_handling as eh

from .utils import *


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
        temporal_matcher=make_combined_temporal_matcher(pd.Timedelta(12, "H")),
    )
    gpis = list(range(npoints))

    for handle_errors in ["ignore", "deprecated"]:
        with pytest.warns(UserWarning, match="No data for dataset 4-missing"):
            results = val.calc(
                gpis,
                gpis,
                gpis,
                rename_cols=False,
                only_with_reference=True,
                handle_errors=handle_errors)
        assert results == {}

    # 'raise' should raise an error
    with pytest.raises(eh.NoTempMatchedDataError):
        with pytest.warns(UserWarning, match="No data for dataset 4-missing"):
            results = val.calc(
                gpis,
                gpis,
                gpis,
                rename_cols=False,
                only_with_reference=True,
                handle_errors="raise")

    # 'returncode' should modify the status code, but not raise an error
    with pytest.warns(UserWarning, match="No data for dataset 4-missing"):
        results = val.calc(
            gpis,
            gpis,
            gpis,
            rename_cols=False,
            only_with_reference=True,
            handle_errors="returncode")
    for key in results:
        for metric in results[key]:
            assert len(results[key][metric]) == npoints
            if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                assert np.all(np.isnan(results[key][metric]))
        assert np.all(results[key]["status"] == eh.NO_TEMP_MATCHED_DATA)


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
        temporal_matcher=make_combined_temporal_matcher(pd.Timedelta(12, "H")),
    )
    gpis = list(range(npoints))

    for handle_errors in ["ignore", "deprecated"]:
        with pytest.warns(UserWarning, match="No data for dataset 1-missing"):
            results = val.calc(
                gpis,
                gpis,
                gpis,
                rename_cols=False,
                only_with_reference=True,
                handle_errors=handle_errors)
        assert results == {}

    # 'raise' should raise an error
    with pytest.raises(eh.NoGpiDataError):
        with pytest.warns(UserWarning, match="No data for dataset 1-missing"):
            results = val.calc(
                gpis,
                gpis,
                gpis,
                rename_cols=False,
                only_with_reference=True,
                handle_errors="raise")

    # 'returncode' should modify the status code, but not raise an error
    with pytest.warns(UserWarning, match="No data for dataset 1-missing"):
        results = val.calc(
            gpis,
            gpis,
            gpis,
            rename_cols=False,
            only_with_reference=True,
            handle_errors="returncode")
    for key in results:
        for metric in results[key]:
            assert len(results[key][metric]) == npoints
            if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                assert np.all(np.isnan(results[key][metric]))
        assert np.all(results[key]["status"] == eh.NO_GPI_DATA)


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
        temporal_matcher=make_combined_temporal_matcher(pd.Timedelta(12, "H")),
    )
    gpis = list(range(npoints))

    for handle_errors in ["ignore", "deprecated", "raise", "returncode"]:
        with pytest.warns(
                UserWarning,
                match="Not enough observations to calculate metrics."):
            results = val.calc(
                gpis,
                gpis,
                gpis,
                rename_cols=False,
                only_with_reference=True,
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
        temporal_matcher=make_combined_temporal_matcher(pd.Timedelta(12, "H")),
    )
    gpis = list(range(npoints))

    for handle_errors in ["ignore", "deprecated", "raise", "returncode"]:
        results = val.calc(
            gpis,
            gpis,
            gpis,
            rename_cols=False,
            only_with_reference=True,
            handle_errors=handle_errors)
        for key in results:
            for metric in results[key]:
                assert len(results[key][metric]) == npoints
                if metric not in ["status", "gpi", "lat", "lon", "n_obs"]:
                    assert np.all(np.isfinite(results[key][metric]))
            assert np.all(results[key]["status"] == eh.OK)
