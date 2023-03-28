"""
Definition of exceptions and error return codes.

The error codes are integer values. They should normally be non-negative if
properly handled. An error code of -1 indicates failure in handling the error.
0 indicates success.
"""

# This is the default value, but it should normally not appear anywhere
# outside, because the more specific failures should set a new return code
UNCAUGHT = -1
# Everything worked as planned
OK = 0
# If the metrics calculation (metrics_calculator.calc_metrics) fails because
# there is not enough data (len(data) < minobs)
INSUFFICIENT_DATA = 1
# If the metrics calculation fails due to another (unforeseen) reason. Should
# not happen too often.
METRICS_CALCULATION_FAILED = 2
# Failure in temporal matching
TEMPORAL_MATCHING_FAILED = 3
# Temporal matching returned without error, but the data we need is not
# available. This can happen if there is no temporal overlap of the required
# datasets, or if there is no data at all for one of the required datasets.
NO_TEMP_MATCHED_DATA = 4
# the scaling failed
SCALING_FAILED = 5
# the call to perform_validation failed due to other unforeseen reasons
VALIDATION_FAILED = 6
# the call to self.data_manager.get_data returned no data
NO_GPI_DATA = 7
# The call to self.data_manager.get_data failed
DATA_MANAGER_FAILED = 8


class ValidationError(Exception):
    # base exception, should not be used directly, only to catch all the
    # different subclasses at once
    return_code = UNCAUGHT


class MetricsCalculationError(ValidationError):
    return_code = METRICS_CALCULATION_FAILED


class TemporalMatchingError(ValidationError):
    return_code = TEMPORAL_MATCHING_FAILED


class NoTempMatchedDataError(ValidationError):
    return_code = NO_TEMP_MATCHED_DATA


class ScalingError(ValidationError):
    return_code = SCALING_FAILED


class ValidationFailedError(ValidationError):
    return_code = VALIDATION_FAILED


class NoGpiDataError(ValidationError):
    return_code = NO_GPI_DATA


class DataManagerError(ValidationError):
    return_code = DATA_MANAGER_FAILED
