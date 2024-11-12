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

"""
Metric calculators implement combinations of metrics and structure the output.
"""

import copy
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import warnings

import pytesmo.metrics as metrics
import pytesmo.df_metrics as df_metrics
from pytesmo.scaling import scale
from pytesmo.validation_framework.data_manager import get_result_names
import pytesmo.validation_framework.error_handling as eh
from pytesmo.df_metrics import n_combinations
from pytesmo.metrics import (
    pairwise,
    tcol_metrics,
)
from pytesmo.metrics.confidence_intervals import (
    tcol_metrics_with_bootstrapped_ci,
    with_bootstrapped_ci,
)
from pytesmo.metrics._fast_pairwise import (
    _moments_welford,
    _mse_corr_from_moments,
    _mse_var_from_moments,
    _mse_bias_from_moments,
    _pearsonr_from_moments,
)
from pytesmo.metrics.pairwise import (
    _bias_ci_from_moments,
    ubrmsd_ci,
    pearson_r_ci,
    spearman_r_ci,
    kendall_tau_ci,
)


def _get_tc_metric_template(metr, ds_names):
    """return empty dict to fill TC results into"""
    if isinstance(metr, str):
        metr = [metr]

    # metrics that are computed between dataset triples
    met_thds_template = {
        "snr": np.float32([np.nan]),
        "err_std": np.float32([np.nan]),
        "beta": np.float32([np.nan]),
    }

    in_lut = np.isin(
        np.array(metr), np.array(list(met_thds_template.keys()))
    )
    if not all(in_lut):
        unknown = np.take(metr, np.where(in_lut is False)[0])
        raise ValueError("Unknown metric(s): {}".format(" ".join(unknown)))

    met_thds = {}
    for m in metr:
        for d in ds_names:
            met_thds[(m, d)] = copy.copy(met_thds_template[m])

    return met_thds


def _get_metric_template(metr):
    """return empty dict to fill results into"""
    if isinstance(metr, str):
        metr = [metr]

    lut = dict()
    # metrics that are equal for all data sets
    met_comm = {"n_obs": np.int32([0])}
    # metrics that are computed between dataset pairs
    met_tds = {
        "R": np.float32([np.nan]),
        "p_R": np.float32([np.nan]),
        "rho": np.float32([np.nan]),
        "p_rho": np.float32([np.nan]),
        "BIAS": np.float32([np.nan]),
        "tau": np.float32([np.nan]),
        "p_tau": np.float32([np.nan]),
        "RMSD": np.float32([np.nan]),
        "mse": np.float32([np.nan]),
        "RSS": np.float32([np.nan]),
        "mse_corr": np.float32([np.nan]),
        "mse_bias": np.float32([np.nan]),
        "mse_var": np.float32([np.nan]),
        "urmsd": np.float32([np.nan]),
        "BIAS_ci_lower": np.float32([np.nan]),
        "BIAS_ci_upper": np.float32([np.nan]),
        "RMSD_ci_lower": np.float32([np.nan]),
        "RMSD_ci_upper": np.float32([np.nan]),
        "RSS_ci_lower": np.float32([np.nan]),
        "RSS_ci_upper": np.float32([np.nan]),
        "mse_ci_lower": np.float32([np.nan]),
        "mse_ci_upper": np.float32([np.nan]),
        "mse_corr_ci_lower": np.float32([np.nan]),
        "mse_corr_ci_upper": np.float32([np.nan]),
        "mse_bias_ci_lower": np.float32([np.nan]),
        "mse_bias_ci_upper": np.float32([np.nan]),
        "mse_var_ci_lower": np.float32([np.nan]),
        "mse_var_ci_upper": np.float32([np.nan]),
        "urmsd_ci_lower": np.float32([np.nan]),
        "urmsd_ci_upper": np.float32([np.nan]),
        "R_ci_lower": np.float32([np.nan]),
        "R_ci_upper": np.float32([np.nan]),
        "rho_ci_lower": np.float32([np.nan]),
        "rho_ci_upper": np.float32([np.nan]),
        "tau_ci_lower": np.float32([np.nan]),
        "tau_ci_upper": np.float32([np.nan]),
    }

    lut.update(met_comm)
    lut.update(met_tds)

    in_lut = np.isin(np.array(metr), np.array(list(lut.keys())))
    if not all(in_lut):
        unknown = np.take(metr, np.where(in_lut is False)[0])
        raise ValueError("Unknown metric(s): {}".format(" ".join(unknown)))

    return {m: lut[m] for m in metr}


class MetadataMetrics(object):
    """
    This class sets up the gpi info and metadata (if used) in the results
    template.
    This is used as the basis for all other metric calculators.

    Parameters
    ----------
    other_name: string or tuple, optional
        Name of the column of the non-reference / other dataset in the
        pandas DataFrame
    metadata_template: dictionary, optional
        A dictionary containing additional fields (and types) of the form
        dict = {'field': np.float32([np.nan]}. Allows users to specify
        information in the job tuple, i.e. jobs.append((idx,
        metadata['longitude'], metadata['latitude'], metadata_dict)) which
        is then propagated to the end netCDF results file.
    min_obs : int, optional
        Minium number of observations required t calculate metrics. Default is
        10.
    """

    def __init__(self, other_name="k1", metadata_template=None, min_obs=10):

        self.result_template = {
            "gpi": np.int32([-1]),
            "lon": np.float64([np.nan]),
            "lat": np.float64([np.nan]),
            "status": np.int32([eh.UNCAUGHT]),
        }

        self.metadata_template = metadata_template
        if self.metadata_template is not None:
            self.result_template.update(metadata_template)

        self.other_name = other_name
        self.min_obs = min_obs

    def calc_metrics(self, data, gpi_info):
        """
        Adds the gpi info and metadata to the results.

        Parameters
        ----------
        data : pandas.DataFrame
            see individual calculators for more information. not directly used
            here.
        gpi_info : tuple
            of (gpi, lon, lat) or, optionally, (gpi, lon, lat, metadata) where
            metadata is a dictionary
        """

        dataset = copy.deepcopy(self.result_template)

        dataset["gpi"][0] = gpi_info[0]
        dataset["lon"][0] = gpi_info[1]
        dataset["lat"][0] = gpi_info[2]

        if self.metadata_template is not None:
            for key, value in self.metadata_template.items():
                try:
                    dataset[key][0] = gpi_info[3][key]
                except (IndexError):
                    raise Exception(
                        "No metadata has been provided to the job. "
                        "Should be of form {field: metadata_value} using "
                        "the metadata_template supplied to init function."
                    )

        return dataset


class BasicMetrics(MetadataMetrics):
    """
    This class just computes the basic metrics,
        - Pearson's R
        - Spearman's rho
        - RMSD
        - BIAS
        - optionally Kendall's tau

    it also stores information about gpi, lat, lon
    and number of observations

    Parameters
    ----------
    other_name: string or tuple, optional
        Name of the column of the non-reference / other dataset in the
        pandas DataFrame
    calc_tau: boolean, optional
        if True then also tau is calculated. This is set to False by default
        since the calculation of Kendalls tau is rather slow and can
        significantly impact performance of e.g. global validation studies
    metadata_template: dictionary, optional
        A dictionary containing additional fields (and types) of the form
        ``dict = {'field': np.float32([np.nan]}``. Allows users to specify
        information in the job tuple, i.e.::

            jobs.append((idx, metadata['longitude'], metadata['latitude'],
                         metadata_dict))``

        which is then propagated to the end netCDF results file.
    """

    def __init__(
            self, other_name="k1", calc_tau=False, metadata_template=None
    ):
        super(BasicMetrics, self).__init__(
            other_name=other_name, metadata_template=metadata_template
        )

        self.basic_metrics = [
            "n_obs",
            "R",
            "p_R",
            "rho",
            "p_rho",
            "RMSD",
            "BIAS",
            "tau",
            "p_tau",
        ]

        self.result_template.update(_get_metric_template(self.basic_metrics))

        self.calc_tau = calc_tau

    def calc_metrics(self, data, gpi_info):
        """
        calculates the desired statistics

        Parameters
        ----------
        data : pandas.DataFrame
            with 2 columns, the first column is the reference dataset
            named 'ref'
            the second column the dataset to compare against named 'other'
        gpi_info : tuple
            of (gpi, lon, lat)

        Notes
        -----
        Kendall tau is calculation is optional at the moment
        because the scipy implementation is very slow which is problematic for
        global comparisons

        """
        dataset = super(BasicMetrics, self).calc_metrics(data, gpi_info)

        if len(data) < self.min_obs:
            dataset["status"][0] = eh.INSUFFICIENT_DATA
            return dataset

        x, y = data["ref"].values, data[self.other_name].values
        R, p_R = stats.pearsonr(x, y)
        rho, p_rho = stats.spearmanr(x, y)
        RMSD = metrics.rmsd(x, y)
        BIAS = metrics.bias(x, y)

        dataset["R"][0], dataset["p_R"][0] = R, p_R
        dataset["rho"][0], dataset["p_rho"][0] = rho, p_rho
        dataset["RMSD"][0] = RMSD
        dataset["BIAS"][0] = BIAS
        dataset["n_obs"][0] = len(data)

        if self.calc_tau:
            tau, p_tau = metrics.kendalltau(x, y)
            dataset["tau"][0], dataset["p_tau"][0] = tau, p_tau

        dataset["status"][0] = eh.OK
        return dataset


class BasicMetricsPlusMSE(BasicMetrics):
    """
    Basic Metrics plus Mean squared Error and the decomposition of the MSE
    into correlation, bias and variance parts.
    """

    def __init__(self, other_name="k1", metadata_template=None):
        super(BasicMetricsPlusMSE, self).__init__(
            other_name=other_name, metadata_template=metadata_template
        )

        self.mse_metrics = ["mse", "mse_corr", "mse_bias", "mse_var"]
        self.result_template.update(_get_metric_template(self.mse_metrics))

    def calc_metrics(self, data, gpi_info):
        dataset = super(BasicMetricsPlusMSE, self).calc_metrics(
            data, gpi_info
        )
        # setting back to uncaught in case something goes wrong
        dataset["status"][0] = eh.UNCAUGHT
        if len(data) < self.min_obs:
            dataset["status"][0] = eh.INSUFFICIENT_DATA
            return dataset
        x, y = data["ref"].values, data[self.other_name].values
        mse, mse_corr, mse_bias, mse_var = metrics.mse_decomposition(x, y)
        dataset["mse"][0] = mse
        dataset["mse_corr"][0] = mse_corr
        dataset["mse_bias"][0] = mse_bias
        dataset["mse_var"][0] = mse_var
        dataset["status"][0] = eh.OK
        return dataset


class FTMetrics(MetadataMetrics):
    """
    This class computes Freeze/Thaw Metrics
    Calculated metrics are:

    - SSF frozen/temp unfrozen
    - SSF unfrozen/temp frozen
    - SSF unfrozen/temp unfrozen
    - SSF frozen/temp frozen

    it also stores information about gpi, lat, lon
    and number of total observations

    """

    def __init__(
            self, frozen_flag=2, other_name="k1", metadata_template=None
    ):
        super(FTMetrics, self).__init__(
            other_name=other_name, metadata_template=metadata_template
        )

        self.frozen_flag_value = frozen_flag
        self.result_template.update(
            {
                "ssf_fr_temp_un": np.float32([np.nan]),
                "ssf_fr_temp_fr": np.float32([np.nan]),
                "ssf_un_temp_fr": np.float32([np.nan]),
                "ssf_un_temp_un": np.float32([np.nan]),
                "n_obs": np.int32([0]),
            }
        )

    def calc_metrics(self, data, gpi_info):
        """
        calculates the desired statistics

        Parameters
        ----------
        data : pandas.DataFrame
            with 2 columns, the first column is the reference dataset
            named 'ref'
            the second column the dataset to compare against named 'other'
        gpi_info : tuple
            of (gpi, lon, lat)

        Notes
        -----
        Kendall tau is not calculated at the moment
        because the scipy implementation is very slow which is problematic for
        global comparisons

        """
        dataset = super(FTMetrics, self).calc_metrics(data, gpi_info)

        if len(data) < self.min_obs:
            dataset["status"][0] = eh.INSUFFICIENT_DATA
            return dataset

        ssf, temp = data["ref"].values, data[self.other_name].values
        # SSF <= 1 unfrozen
        # SSF >= 2 frozen

        ssf_frozen = np.where(ssf == self.frozen_flag_value)[0]
        ssf_unfrozen = np.where(ssf != self.frozen_flag_value)[0]

        temp_ssf_frozen = temp[ssf_frozen]
        temp_ssf_unfrozen = temp[ssf_unfrozen]

        # correct classifications
        ssf_temp_frozen = np.where(temp_ssf_frozen < 0)[0]
        ssf_temp_unfrozen = np.where(temp_ssf_unfrozen >= 0)[0]

        # incorrect classifications
        ssf_fr_temp_unfrozen = np.where(temp_ssf_frozen >= 0)[0]
        ssf_un_temp_frozen = np.where(temp_ssf_unfrozen < 0)[0]

        dataset["ssf_fr_temp_un"][0] = len(ssf_fr_temp_unfrozen)
        dataset["ssf_fr_temp_fr"][0] = len(ssf_temp_frozen)

        dataset["ssf_un_temp_fr"][0] = len(ssf_un_temp_frozen)
        dataset["ssf_un_temp_un"][0] = len(ssf_temp_unfrozen)

        dataset["n_obs"][0] = len(data)

        dataset["status"][0] = eh.OK
        return dataset


class HSAF_Metrics(MetadataMetrics):
    """
    This class computes metrics as defined by the H-SAF consortium in
    order to prove the operational readiness of a product. It also stores
    information about gpi, lat, lon and number of observations.
    """

    def __init__(
            self,
            other_name1="k1",
            other_name2="k2",
            dataset_names=None,
            metadata_template=None,
    ):

        super(HSAF_Metrics, self).__init__(
            other_name=other_name1, metadata_template=metadata_template
        )

        # prepare validation dataset names as provided
        self.other_name1 = other_name1
        self.other_name2 = other_name2
        self.df_columns = ["ref", self.other_name1, self.other_name2]
        if dataset_names is None:
            self.ds_names = self.df_columns
        else:
            self.ds_names = dataset_names

        # create lut between df columns and dataset names
        self.ds_names_lut = {}
        for name, col in zip(self.ds_names, self.df_columns):
            self.ds_names_lut[col] = name

        self.tds_names = []
        for combi in itertools.combinations(self.df_columns, 2):
            self.tds_names.append("{:}_and_{:}".format(*combi))

        metrics_common = {"n_obs": np.int32([0])}

        metrics_sds = {
            "snr": np.float32([np.nan]),
            "err_var": np.float32([np.nan]),
            "beta": np.float32([np.nan]),
        }

        metrics_tds = {
            "R": np.float32([np.nan]),
            "p_R": np.float32([np.nan]),
            "rho": np.float32([np.nan]),
            "p_rho": np.float32([np.nan]),
            "bias": np.float32([np.nan]),
            "ubrmsd": np.float32([np.nan]),
        }

        self.seasons = ["ALL", "DJF", "MAM", "JJA", "SON"]

        for season in self.seasons:
            # get template for common metric
            for metric in metrics_common.keys():
                key = "{:}_{:}".format(season, metric)
                self.result_template[key] = metrics_common[metric].copy()

            # get template for single-dataset metric
            for name in self.ds_names:
                for metric in metrics_sds.keys():
                    key = "{:}_{:}_{:}".format(name, season, metric)
                    self.result_template[key] = metrics_sds[metric].copy()

            # get template for two-dataset metric
            for tds_name in self.tds_names:
                split_tds_name = tds_name.split("_and_")
                tds_name_key = "{:}_{:}".format(
                    self.ds_names_lut[split_tds_name[0]],
                    self.ds_names_lut[split_tds_name[1]],
                )
                for metric in metrics_tds.keys():
                    key = "{:}_{:}_{:}".format(tds_name_key, season, metric)
                    self.result_template[key] = metrics_tds[metric].copy()

        self.month_to_season = np.array(
            [
                "",
                "DJF",
                "DJF",
                "MAM",
                "MAM",
                "MAM",
                "JJA",
                "JJA",
                "JJA",
                "SON",
                "SON",
                "SON",
                "DJF",
            ]
        )

    def calc_metrics(self, data, gpi_info):
        """
        calculates the desired statistics

        Parameters
        ----------
        data : pandas.DataFrame
            with 3 columns, the first column is the reference dataset
            named 'ref'
            the second and third column are the datasets to compare against
            named 'k1 and k2'
        gpi_info : tuple
            Grid point info (i.e. gpi, lon, lat)
        """
        dataset = super(HSAF_Metrics, self).calc_metrics(data, gpi_info)

        for season in self.seasons:

            if season != "ALL":
                subset = self.month_to_season[data.index.month] == season
            else:
                subset = np.ones(len(data), dtype=bool)

            # number of observations
            n_obs = subset.sum()
            if n_obs < self.min_obs:
                continue
            dataset["{:}_n_obs".format(season)][0] = n_obs

            # get single dataset metrics
            # calculate SNR
            x = data[self.df_columns[0]].values[subset]
            y = data[self.df_columns[1]].values[subset]
            z = data[self.df_columns[2]].values[subset]

            snr, err, beta = metrics.tcol_metrics(x, y, z)

            for i, name in enumerate(self.ds_names):
                dataset["{:}_{:}_snr".format(name, season)][0] = snr[i]
                dataset["{:}_{:}_err_var".format(name, season)][0] = err[i]
                dataset["{:}_{:}_beta".format(name, season)][0] = beta[i]

            # calculate Pearson correlation
            pearson_R, pearson_p = df_metrics.pearsonr(data)
            pearson_R = pearson_R._asdict()
            pearson_p = pearson_p._asdict()

            # calculate Spearman correlation
            spea_rho, spea_p = df_metrics.spearmanr(data)
            spea_rho = spea_rho._asdict()
            spea_p = spea_p._asdict()

            # scale data to reference in order to calculate absolute metrics
            data_scaled = scale(data, method="min_max")

            # calculate bias
            bias_nT = df_metrics.bias(data_scaled)
            bias_dict = bias_nT._asdict()

            # calculate ubRMSD
            ubRMSD_nT = df_metrics.ubrmsd(data_scaled)
            ubRMSD_dict = ubRMSD_nT._asdict()

            for tds_name in self.tds_names:
                R = pearson_R[tds_name]
                p_R = pearson_p[tds_name]
                rho = spea_rho[tds_name]
                p_rho = spea_p[tds_name]
                bias = bias_dict[tds_name]
                ubRMSD = ubRMSD_dict[tds_name]

                split_tds_name = tds_name.split("_and_")
                tds_name_key = "{:}_{:}".format(
                    self.ds_names_lut[split_tds_name[0]],
                    self.ds_names_lut[split_tds_name[1]],
                )

                dataset["{:}_{:}_R".format(tds_name_key, season)][0] = R
                dataset["{:}_{:}_p_R".format(tds_name_key, season)][0] = p_R
                dataset["{:}_{:}_rho".format(tds_name_key, season)][0] = rho
                dataset["{:}_{:}_p_rho".format(tds_name_key, season)][
                    0
                ] = p_rho
                dataset["{:}_{:}_bias".format(tds_name_key, season)][
                    0
                ] = bias
                dataset["{:}_{:}_ubrmsd".format(tds_name_key, season)][
                    0
                ] = ubRMSD

        dataset["status"][0] = eh.OK
        return dataset


class IntercomparisonMetrics(MetadataMetrics):
    """
    Compare Basic Metrics of multiple satellite data sets to one reference
    data set via:

    - Pearson's R and p
    - Spearman's rho and p
    - RMSD
    - BIAS
    - ubRMSD
    - mse
    - RSS
    - optionally Kendall's tau

    Parameters
    ----------
    refname : str, optional
        Name of the reference column in the DataFrame.
    other_names: tuple, optional (default: ('k1', 'k2', 'k3))
        Name of the column of the non-reference / other datasets in the
        DataFrame.
    calc_rho: boolean, optional
        If True then also Spearman's rho is calculated. This is set to True by
        default.
    calc_tau: boolean, optional
        if True then also tau is calculated. This is set to False by default
        since the calculation of Kendalls tau is rather slow and can
        significantly impact performance of e.g. global validation studies
    metrics_between_nonref : bool, optional (default: False)
        Allow 2-dataset combinations where the ref is not included.
        Warning: can lead to many combinations.
    dataset_names : list, optional (default: None)
        Names of the original datasets, that are used to find the lookup table
        for the df cols.
    metadata_template: dict, optional (default: None)
        See MetadataMetrics

    """

    def __init__(
            self,
            refname="ref",
            other_names=("k1", "k2", "k3"),
            calc_tau=False,
            metrics_between_nonref=False,
            calc_rho=True,
            dataset_names=None,
            metadata_template=None,
    ):
        warnings.warn(
            "pytesmo IntercomparisonMetrics calculator "
            "is deprecated and will be removed in a future "
            "release. Use the PairwiseIntercomparisonMetrics "
            "class instead.", DeprecationWarning
        )
        other_names = list(other_names)
        super(IntercomparisonMetrics, self).__init__(
            other_name=other_names, metadata_template=metadata_template
        )

        # string that splits the dataset names and metric names in the output
        # e.g. 'metric_between_dataset1_and_dataset2'
        self.ds_names_split, self.metric_ds_split = "_and_", "_between_"

        self.df_columns = [refname] + self.other_name

        self.calc_rho = calc_rho
        self.calc_tau = calc_tau

        if dataset_names is None:
            self.ds_names = self.df_columns
        else:
            self.ds_names = dataset_names

        self.ds_names_lut = {}
        for name, col in zip(self.ds_names, self.df_columns):
            self.ds_names_lut[col] = name

        combis = n_combinations(
            self.df_columns,
            2,
            must_include=refname if not metrics_between_nonref else None,
        )

        self.tds_names = []
        for combi in combis:
            self.tds_names.append(
                "{1}{0}{2}".format(self.ds_names_split, *combi)
            )

        # metrics that are equal for all datasets
        metrics_common = ["n_obs"]
        # metrics that are calculated between dataset pairs
        metrics_tds = [
            "R",
            "p_R",
            "rho",
            "p_rho",
            "BIAS",
            "RMSD",
            "mse",
            "RSS",
            "mse_corr",
            "mse_bias",
            "urmsd",
            "mse_var",
            "tau",
            "p_tau",
        ]

        metrics_common = _get_metric_template(metrics_common)
        metrics_tds = _get_metric_template(metrics_tds)

        for metric in metrics_common.keys():
            self.result_template[metric] = metrics_common[metric].copy()

        for tds_name in self.tds_names:
            split_tds_name = tds_name.split(self.ds_names_split)
            tds_name_key = self.ds_names_split.join(
                [
                    self.ds_names_lut[split_tds_name[0]],
                    self.ds_names_lut[split_tds_name[1]],
                ]
            )
            for metric in metrics_tds.keys():
                key = self.metric_ds_split.join([metric, tds_name_key])
                self.result_template[key] = metrics_tds[metric].copy()

        if not calc_rho:
            self.result_template.pop("rho", None)
            self.result_template.pop("p_rho", None)
        if not calc_tau:
            self.result_template.pop("tau", None)
            self.result_template.pop("p_tau", None)

    def calc_metrics(self, data, gpi_info):
        """
        calculates the desired statistics

        Parameters
        ----------
        data : pd.DataFrame
            with >2 columns, the first column is the reference dataset
            named 'ref' other columns are the datasets to compare against
            named 'other_i'
        gpi_info : tuple
            of (gpi, lon, lat)

        Notes
        -----
        Kendall tau is calculation is optional at the moment
        because the scipy implementation is very slow which is problematic for
        global comparisons
        """

        dataset = super(IntercomparisonMetrics, self).calc_metrics(
            data, gpi_info
        )

        subset = np.ones(len(data), dtype=bool)

        n_obs = subset.sum()
        dataset["n_obs"][0] = n_obs
        if n_obs < self.min_obs:
            dataset["status"][0] = eh.INSUFFICIENT_DATA
            return dataset

        # make sure we have the correct order
        data = data[self.df_columns]

        # calculate Pearson correlation
        pearson_R, pearson_p = df_metrics.pearsonr(data)
        pearson_R, pearson_p = pearson_R._asdict(), pearson_p._asdict()

        # calculate bias
        bias_nT = df_metrics.bias(data)
        bias_dict = bias_nT._asdict()

        # calculate RMSD
        rmsd = df_metrics.rmsd(data)
        rmsd_dict = rmsd._asdict()

        # calculate MSE
        mse, mse_corr, mse_bias, mse_var = df_metrics.mse_decomposition(data)
        mse_dict, mse_corr_dict, mse_bias_dict, mse_var_dict = (
            mse._asdict(),
            mse_corr._asdict(),
            mse_bias._asdict(),
            mse_var._asdict(),
        )

        # calculate RSS
        rss = df_metrics.RSS(data)
        rss_dict = rss._asdict()

        # calculate Spearman correlation
        if self.calc_rho:
            rho, p_rho = df_metrics.spearmanr(data)
            rho_dict, p_rho_dict = rho._asdict(), p_rho._asdict()
        else:
            rho = p_rho = p_rho_dict = rho_dict = None

        # calulcate tau
        if self.calc_tau:
            tau, p_tau = df_metrics.kendalltau(data)
            tau_dict, p_tau_dict = tau._asdict(), p_tau._asdict()
        else:
            tau = p_tau = p_tau_dict = tau_dict = None

        # No extra scaling is performed here.
        # always scale for ubRMSD with mean std
        # calculate ubRMSD
        data_scaled = scale(data, method="mean_std")
        ubRMSD_nT = df_metrics.ubrmsd(data_scaled)
        ubRMSD_dict = ubRMSD_nT._asdict()

        for tds_name in self.tds_names:
            R, p_R = pearson_R[tds_name], pearson_p[tds_name]
            bias = bias_dict[tds_name]
            mse = mse_dict[tds_name]
            mse_corr = mse_corr_dict[tds_name]
            mse_bias = mse_bias_dict[tds_name]
            mse_var = mse_var_dict[tds_name]
            rmsd = rmsd_dict[tds_name]
            ubRMSD = ubRMSD_dict[tds_name]
            rss = rss_dict[tds_name]

            if rho_dict and p_rho_dict:
                rho = rho_dict[tds_name]
                p_rho = p_rho_dict[tds_name]
            if tau_dict and p_tau_dict:
                tau = tau_dict[tds_name]
                p_tau = p_tau_dict[tds_name]

            split_tds_name = tds_name.split(self.ds_names_split)
            tds_name_key = self.ds_names_split.join(
                [
                    self.ds_names_lut[split_tds_name[0]],
                    self.ds_names_lut[split_tds_name[1]],
                ]
            )

            dataset[self.metric_ds_split.join(["R", tds_name_key])][0] = R
            dataset[self.metric_ds_split.join(["p_R", tds_name_key])][
                0
            ] = p_R
            dataset[self.metric_ds_split.join(["BIAS", tds_name_key])][
                0
            ] = bias
            dataset[self.metric_ds_split.join(["mse", tds_name_key])][
                0
            ] = mse
            dataset[self.metric_ds_split.join(["mse_corr", tds_name_key])][
                0
            ] = mse_corr
            dataset[self.metric_ds_split.join(["mse_bias", tds_name_key])][
                0
            ] = mse_bias
            dataset[self.metric_ds_split.join(["mse_var", tds_name_key])][
                0
            ] = mse_var
            dataset[self.metric_ds_split.join(["RMSD", tds_name_key])][
                0
            ] = rmsd
            dataset[self.metric_ds_split.join(["urmsd", tds_name_key])][
                0
            ] = ubRMSD
            dataset[self.metric_ds_split.join(["RSS", tds_name_key])][
                0
            ] = rss

            if self.calc_rho:
                dataset[self.metric_ds_split.join(["rho", tds_name_key])][
                    0
                ] = rho
                dataset[self.metric_ds_split.join(["p_rho", tds_name_key])][
                    0
                ] = p_rho
            if self.calc_tau:
                dataset[self.metric_ds_split.join(["tau", tds_name_key])][
                    0
                ] = tau
                dataset[self.metric_ds_split.join(["p_tau", tds_name_key])][
                    0
                ] = p_tau

        dataset["status"][0] = eh.OK
        return dataset


class TCMetrics(MetadataMetrics):
    """
    This class computes triple collocation metrics as defined in the QA4SM
    project. It uses 2 satellite and 1 reference data sets as inputs only.
    It can be extended to perform intercomparison between possible triples
    of more than 3 datasets.
    """

    def __init__(
            self,
            other_names=("k1", "k2"),
            calc_tau=False,
            dataset_names=None,
            tc_metrics_for_ref=True,
            metrics_between_nonref=False,
            metadata_template=None,
    ):
        """
        Triple Collocation metrics as implemented in the QA4SM project.

        Parameters
        ----------
        other_names : tuple, optional (default: ('k1', 'k2'))
            Names of the data sets that are not the reference in the
            data frame.
        calc_tau : bool, optional (default: False)
            Calculate Kendall's Tau (slow)
        dataset_names : tuple, optional (default: None)
            List that maps the names of the satellite dataset columns to their
            real name that will be used in the results file.
        tc_metrics_for_ref : bool, optional (default: False)
            Store TC metrics for the reference data set as well.
        metrics_between_nonref : bool, optional (default: False)
            Allow 2-dataset combinations where the ref is not included.
            Warning: can lead to many combinations.
        metadata_template: dictionary, optional
            A dictionary containing additional fields (and types) of the form
            dict = {'field': np.float32([np.nan]}. Allows users to specify
            information in the job tuple,
            i.e. jobs.append(
                (idx,
                 metadata['longitude'],
                 metadata['latitude'],
                 metadata_dict))
            which is then propagated to the end netCDF results file.

        """
        warnings.warn(
            "pytesmo TCMetrics calculator "
            "is deprecated and will be removed in a future "
            "release. Use the TripleCollocationMetrics "
            "class instead.", DeprecationWarning
        )

        self.ref_name = "ref"
        other_names = list(other_names)
        super(TCMetrics, self).__init__(
            other_name=other_names, metadata_template=metadata_template
        )

        # string that splits the dataset names and metric names in the output
        # e.g. 'metric_between_dataset1_and_dataset2'
        self.ds_names_split, self.metric_ds_split = "_and_", "_between_"

        self.calc_tau = calc_tau
        self.df_columns = [self.ref_name] + self.other_name

        if dataset_names is None:
            self.ds_names = self.df_columns
        else:
            self.ds_names = dataset_names

        self.ds_names_lut = {}
        for name, col in zip(self.ds_names, self.df_columns):
            self.ds_names_lut[col] = name

        self.metrics_between_nonref = metrics_between_nonref
        self.tds_names, self.thds_names = self._make_names()

        # metrics that are equal for all datasets
        metrics_common = ["n_obs"]
        # metrics that are calculated between dataset pairs
        metrics_tds = [
            "R",
            "p_R",
            "rho",
            "p_rho",
            "BIAS",
            "RMSD",
            "mse",
            "RSS",
            "mse_corr",
            "mse_bias",
            "urmsd",
            "mse_var",
            "tau",
            "p_tau",
        ]
        # metrics that are calculated between dataset triples
        metrics_thds = ["snr", "err_std", "beta"]

        metrics_common = _get_metric_template(metrics_common)
        metrics_tds = _get_metric_template(metrics_tds)

        ignore_ds = [self.ref_name] if not tc_metrics_for_ref else ()
        metrics_thds = _get_tc_metric_template(
            metrics_thds,
            [
                self.ds_names_lut[n]
                for n in self.df_columns
                if n not in ignore_ds
            ],
        )

        for metric in metrics_common.keys():
            self.result_template[metric] = metrics_common[metric].copy()

        for tds_name in self.tds_names:
            split_tds_name = tds_name.split(self.ds_names_split)
            tds_name_key = self.ds_names_split.join(
                [
                    self.ds_names_lut[split_tds_name[0]],
                    self.ds_names_lut[split_tds_name[1]],
                ]
            )
            for metric in metrics_tds.keys():
                key = self.metric_ds_split.join([metric, tds_name_key])
                self.result_template[key] = metrics_tds[metric].copy()

        for thds_name in self.thds_names:
            split_tds_name = thds_name.split(self.ds_names_split)
            thds_name_key = self.ds_names_split.join(
                [
                    self.ds_names_lut[split_tds_name[0]],
                    self.ds_names_lut[split_tds_name[1]],
                    self.ds_names_lut[split_tds_name[2]],
                ]
            )
            for metric, ds in metrics_thds.keys():
                if not any(
                        [
                            self.ds_names_lut[other_ds] == ds
                            for other_ds in thds_name.split(
                                self.ds_names_split)
                        ]
                ):
                    continue
                full_name = "_".join([metric, ds])
                key = self.metric_ds_split.join([full_name, thds_name_key])
                self.result_template[key] = metrics_thds[(metric, ds)].copy()

        if not calc_tau:
            self.result_template.pop("tau", None)
            self.result_template.pop("p_tau", None)

    def _make_names(self):
        tds_names, thds_names = [], []
        combis_2 = n_combinations(
            self.df_columns,
            2,
            must_include=[self.ref_name]
            if not self.metrics_between_nonref
            else None,
        )
        combis_3 = n_combinations(
            self.df_columns, 3, must_include=[self.ref_name]
        )

        for combi in combis_2:
            tds_names.append(self.ds_names_split.join(combi))

        for combi in combis_3:
            thds_names.append(
                "{1}{0}{2}{0}{3}".format(self.ds_names_split, *combi)
            )

        return tds_names, thds_names

    def _tc_res_dict(self, res):
        """name is the TC metric name and res the according named tuple"""
        res_dict = {}

        metric = np.array([type(r).__name__ for r in res])
        assert all(metric == np.repeat(metric[0], metric.size))

        for r in res:
            r_d = r._asdict()
            ds = self.ds_names_split.join(list(r_d.keys()))
            res_dict[ds] = dict(zip(list(r_d.keys()), list(r_d.values())))

        return res_dict

    def calc_metrics(self, data, gpi_info):
        """
        Calculate Triple Collocation metrics

        Parameters
        ----------
        data : pd.DataFrame
            with >2 columns, the first column is the reference dataset
            named 'ref' other columns are the data sets to compare against
            named 'other_i'
        gpi_info : tuple
            of (gpi, lon, lat)

        Notes
        -----
        Kendall tau is calculation is optional at the moment
        because the scipy implementation is very slow which is problematic for
        global comparisons
        """

        dataset = copy.deepcopy(self.result_template)

        dataset["gpi"][0] = gpi_info[0]
        dataset["lon"][0] = gpi_info[1]
        dataset["lat"][0] = gpi_info[2]

        if self.metadata_template is not None:
            for key, value in self.metadata_template.items():
                dataset[key][0] = gpi_info[3][key]

        # number of observations
        subset = np.ones(len(data), dtype=bool)

        n_obs = subset.sum()
        dataset["n_obs"][0] = n_obs
        if n_obs < self.min_obs:
            dataset["status"][0] = eh.INSUFFICIENT_DATA
            return dataset

        # calculate Pearson correlation
        pearson_R, pearson_p = df_metrics.pearsonr(data)
        pearson_R, pearson_p = pearson_R._asdict(), pearson_p._asdict()
        # calculate Spearman correlation
        spea_rho, spea_p = df_metrics.spearmanr(data)
        spea_rho, spea_p = spea_rho._asdict(), spea_p._asdict()
        # calculate bias
        bias_nT = df_metrics.bias(data)
        bias_dict = bias_nT._asdict()
        # calculate RMSD
        rmsd = df_metrics.rmsd(data)
        rmsd_dict = rmsd._asdict()
        # calculate MSE
        mse, mse_corr, mse_bias, mse_var = df_metrics.mse_decomposition(data)
        mse_dict = mse._asdict()
        mse_corr_dict = mse_corr._asdict()
        mse_bias_dict = mse_bias._asdict()
        mse_var_dict = mse_var._asdict()
        # calculate RSS
        rss = df_metrics.RSS(data)
        rss_dict = rss._asdict()
        # calculate ubRMSD
        # todo: we could use the TC derived scaling parameters here?
        data_scaled = scale(data, method="mean_std")
        ubRMSD_nT = df_metrics.ubrmsd(data_scaled)
        ubRMSD_dict = ubRMSD_nT._asdict()
        # calulcate tau
        if self.calc_tau:
            tau, p_tau = df_metrics.kendalltau(data)
            tau_dict, p_tau_dict = tau._asdict(), p_tau._asdict()
        else:
            tau = p_tau = p_tau_dict = tau_dict = None
        # calculate TC metrics
        ref_ind = np.where(np.array(data.columns) == self.ref_name)[0][0]
        snrs, err_stds, betas = df_metrics.tcol_metrics(
            data, ref_ind=ref_ind
        )
        snr_dict = self._tc_res_dict(snrs)
        err_std_dict = self._tc_res_dict(err_stds)
        beta_dict = self._tc_res_dict(betas)

        # store TC results
        for thds_name in self.thds_names:
            snr = snr_dict[thds_name]
            err_std = err_std_dict[thds_name]
            beta = beta_dict[thds_name]

            split_thds_name = thds_name.split(self.ds_names_split)
            thds_name_key = self.ds_names_split.join(
                [
                    self.ds_names_lut[split_thds_name[0]],
                    self.ds_names_lut[split_thds_name[1]],
                    self.ds_names_lut[split_thds_name[2]],
                ]
            )

            for metr, res in dict(
                    snr=snr, err_std=err_std, beta=beta
            ).items():
                for ds, ds_res in res.items():
                    m_ds = "{}_{}".format(metr, self.ds_names_lut[ds])
                    n = "{}{}{}".format(
                        m_ds, self.metric_ds_split, thds_name_key
                    )
                    if n in dataset.keys():
                        dataset[n][0] = ds_res

        # Store basic metrics results
        for tds_name in self.tds_names:
            R, p_R = pearson_R[tds_name], pearson_p[tds_name]
            rho, p_rho = spea_rho[tds_name], spea_p[tds_name]
            bias = bias_dict[tds_name]
            mse = mse_dict[tds_name]
            mse_corr = mse_corr_dict[tds_name]
            mse_bias = mse_bias_dict[tds_name]
            mse_var = mse_var_dict[tds_name]
            rmsd = rmsd_dict[tds_name]
            ubRMSD = ubRMSD_dict[tds_name]
            rss = rss_dict[tds_name]

            if tau_dict and p_tau_dict:
                tau = tau_dict[tds_name]
                p_tau = p_tau_dict[tds_name]

            split_tds_name = tds_name.split(self.ds_names_split)
            tds_name_key = self.ds_names_split.join(
                [
                    self.ds_names_lut[split_tds_name[0]],
                    self.ds_names_lut[split_tds_name[1]],
                ]
            )

            dataset[self.metric_ds_split.join(["R", tds_name_key])][0] = R
            dataset[self.metric_ds_split.join(["p_R", tds_name_key])][
                0
            ] = p_R
            dataset[self.metric_ds_split.join(["rho", tds_name_key])][
                0
            ] = rho
            dataset[self.metric_ds_split.join(["p_rho", tds_name_key])][
                0
            ] = p_rho
            dataset[self.metric_ds_split.join(["BIAS", tds_name_key])][
                0
            ] = bias
            dataset[self.metric_ds_split.join(["mse", tds_name_key])][
                0
            ] = mse
            dataset[self.metric_ds_split.join(["mse_corr", tds_name_key])][
                0
            ] = mse_corr
            dataset[self.metric_ds_split.join(["mse_bias", tds_name_key])][
                0
            ] = mse_bias
            dataset[self.metric_ds_split.join(["mse_var", tds_name_key])][
                0
            ] = mse_var
            dataset[self.metric_ds_split.join(["RMSD", tds_name_key])][
                0
            ] = rmsd
            dataset[self.metric_ds_split.join(["urmsd", tds_name_key])][
                0
            ] = ubRMSD
            dataset[self.metric_ds_split.join(["RSS", tds_name_key])][
                0
            ] = rss

            if self.calc_tau:
                dataset[self.metric_ds_split.join(["tau", tds_name_key])][
                    0
                ] = tau
                dataset[self.metric_ds_split.join(["p_tau", tds_name_key])][
                    0
                ] = p_tau

        dataset["status"][0] = eh.OK
        return dataset


class RollingMetrics(MetadataMetrics):
    """
    This class computes rolling metrics for Pearson R and RMSD. It also stores
    information about gpi, lat, lon and number of observations.

    Parameters
    ----------
    other_name: string or tuple, optional
        Name of the column of the non-reference / other dataset in the
        pandas DataFrame
    metadata_template: dictionary, optional
        A dictionary containing additional fields (and types) of the form
        ``dict = {'field': np.float32([np.nan]}``. Allows users to specify
        information in the job tuple, i.e.::

            jobs.append((idx, metadata['longitude'], metadata['latitude'],
                         metadata_dict))``

        which is then propagated to the end netCDF results file.
    """

    def __init__(self, other_name="k1", metadata_template=None):
        super(RollingMetrics, self).__init__(
            other_name=other_name, metadata_template=metadata_template
        )

        self.basic_metrics = ["R", "p_R", "RMSD"]
        self.result_template.update(_get_metric_template(self.basic_metrics))

    def calc_metrics(
            self, data, gpi_info, window_size="30d", center=True, min_periods=2
    ):
        """
        Calculate the desired statistics.

        Parameters
        ----------
        data : pandas.DataFrame
            with 2 columns, the first column is the reference dataset
            named 'ref' the second column the dataset to compare
            against named 'other'
        gpi_info : tuple
            of (gpi, lon, lat)
        window_size : string
            Window size defined as string.
        center : bool, optional
            Set window at the center.
        min_periods : int, optional
            Minimum number of observations in window required for computation.
        """
        dataset = super(RollingMetrics, self).calc_metrics(data, gpi_info)

        if len(data) < self.min_obs:
            dataset["status"][0] = eh.INSUFFICIENT_DATA
            return dataset

        xy = data.to_numpy()
        timestamps = data.index.to_julian_date().values
        window_size_jd = pd.Timedelta(
            window_size
        ).to_numpy() / np.timedelta64(1, "D")
        pr_arr, rmsd_arr = metrics.rolling_pr_rmsd(
            timestamps,
            xy[:, 0],
            xy[:, 1],
            window_size_jd,
            center,
            min_periods,
        )

        dataset["time"] = np.array([data.index])
        dataset["R"] = np.array([pr_arr[:, 0]])
        dataset["p_R"] = np.array([pr_arr[:, 1]])
        dataset["RMSD"] = np.array([rmsd_arr[:]])

        dataset["status"][0] = eh.OK
        return dataset


def get_dataset_names(ref_key, datasets, n=3):
    """
    Get dataset names in correct order as used in the validation framework

    - reference dataset = ref
    - first other dataset = k1
    - second other dataset = k2

    This is important to correctly iterate through the H-SAF metrics and to
    save each metric with the name of the used datasets

    Parameters
    ----------
    ref_key: basestring
        Name of the reference dataset
    datasets: dict
        Dictionary of dictionaries as provided to the validation framework
        in order to perform the validation process.

    Returns
    -------
    dataset_names: list
        List of the dataset names in correct order

    """
    ds_dict = {}
    for ds in datasets.keys():
        ds_dict[ds] = datasets[ds]["columns"]
    ds_names = get_result_names(ds_dict, ref_key, n)
    dataset_names = []
    for name in ds_names[0]:
        dataset_names.append(name[0])

    return dataset_names


class PairwiseMetricsMixin:
    def _pairwise_metric_names(self):
        """
        Returns a list of metric names to be calculated between pairs.
        """
        metrics = [
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
        ]
        if self.calc_spearman:
            metrics += ["rho", "p_rho"]
        if self.calc_kendall:
            metrics += ["tau", "p_tau"]
        if self.analytical_cis:
            metrics += [
                "BIAS_ci_lower", "BIAS_ci_upper",
                "urmsd_ci_lower", "urmsd_ci_upper",
                "R_ci_lower", "R_ci_upper",
            ]
            if self.calc_spearman:
                metrics += ["rho_ci_lower", "rho_ci_upper"]
            if self.calc_kendall:
                metrics += ["tau_ci_lower", "tau_ci_upper"]
        if self.bootstrap_cis:
            metrics += ["mse_var_ci_lower", "mse_var_ci_upper",
                        "mse_corr_ci_lower", "mse_corr_ci_upper",
                        "mse_bias_ci_lower", "mse_bias_ci_upper",
                        "RMSD_ci_lower", "RMSD_ci_upper",
                        "mse_ci_lower", "mse_ci_upper",
                        "RSS_ci_lower", "RSS_ci_upper",
                        ]
        return metrics

    def _calc_pairwise_metrics(
            self, x, y, mx, my, varx, vary, cov, result, suffix=""
    ):
        """
        Calculates pairwise metrics, making use of pre-computed moments.

        Parameters
        ----------
        x, y : np.ndarray
            Data as numpy matrix
        mx, my : float
            Means
        varx, vary : float
            Variances
        cov : float
            Covariance
        result : dict
            Result template dictionary. This is where the results are written
            to.
        suffix : str, optional
            Suffix to be used for storing the metric calculation result. E.g.,
            with ``suffix=_between_k1_and_k2``, bias will be stored as
            ``BIAS_between_k1_and_k2``.
        """
        n_obs = len(x)
        result["BIAS" + suffix][0] = mx - my

        mse_corr = _mse_corr_from_moments(mx, my, varx, vary, cov)
        mse_var = _mse_var_from_moments(mx, my, varx, vary, cov)
        mse_bias = _mse_bias_from_moments(mx, my, varx, vary, cov)
        mse = max(mse_corr + mse_var + mse_bias, 0)
        result["mse_corr" + suffix][0] = mse_corr
        result["mse_var" + suffix][0] = mse_var
        result["mse_bias" + suffix][0] = mse_bias
        result["mse" + suffix][0] = mse

        result["RSS" + suffix][0] = mse * n_obs
        result["RMSD" + suffix][0] = np.sqrt(mse)
        result["urmsd" + suffix][0] = np.sqrt(mse - mse_bias)

        R, p_R = _pearsonr_from_moments(varx, vary, cov, n_obs)
        result["R" + suffix][0] = R
        result["p_R" + suffix][0] = p_R

        if self.calc_spearman:
            (
                result["rho" + suffix][0],
                result["p_rho" + suffix][0],
            ) = stats.spearmanr(x, y)
        if self.calc_kendall:
            (
                result["tau" + suffix][0],
                result["p_tau" + suffix][0],
            ) = stats.kendalltau(x, y)

        if self.analytical_cis:
            (
                result["BIAS_ci_lower" + suffix][0],
                result["BIAS_ci_upper" + suffix][0],
            ) = _bias_ci_from_moments(0.05, mx, my, varx, vary, cov, n_obs)

            (
                result["urmsd_ci_lower" + suffix][0],
                result["urmsd_ci_upper" + suffix][0],
            ) = ubrmsd_ci(x, y, result["urmsd" + suffix][0])

            (
                result["R_ci_lower" + suffix][0],
                result["R_ci_upper" + suffix][0],
            ) = pearson_r_ci(x, y, result["R" + suffix][0])

            if self.calc_spearman:
                (
                    result["rho_ci_lower" + suffix][0],
                    result["rho_ci_upper" + suffix][0],
                ) = spearman_r_ci(x, y, result["rho" + suffix][0])
            if self.calc_kendall:
                (
                    result["tau_ci_lower" + suffix][0],
                    result["tau_ci_upper" + suffix][0],
                ) = kendall_tau_ci(x, y, result["tau" + suffix][0])

        if self.bootstrap_cis:
            for m in ["mse_var", "mse_corr", "mse_bias", "mse", "RMSD"]:
                if m == "mse":
                    metric_func = pairwise.msd
                elif m == "RMSD":
                    metric_func = pairwise.rmsd
                else:
                    metric_func = getattr(pairwise, m)
                kwargs = {}
                if hasattr(self, 'bootstrap_alpha'):
                    kwargs['alpha'] = getattr(self, 'bootstrap_alpha')
                if hasattr(self, 'bootstrap_min_obs'):
                    kwargs['minimum_data_length'] = getattr(
                        self, 'bootstrap_min_obs')
                _, lb, ub = with_bootstrapped_ci(metric_func, x, y, **kwargs)
                result[f"{m}_ci_lower" + suffix][0] = lb
                result[f"{m}_ci_upper" + suffix][0] = ub


class PairwiseIntercomparisonMetrics(MetadataMetrics, PairwiseMetricsMixin):
    """
    Basic metrics for comparison of two datasets:

    - RMSD
    - BIAS
    - ubRMSD
    - mse and decomposition (mse_var, mse_corr, mse_bias)
    - RSS
    - Pearson's R and p
    - Spearman's rho and p (optional)
    - Kendall's tau and p (optional)

    Additionally, confidence intervals for these metrics can be calculated
    (optional).

    **NOTE**: When using this within a
      ``pytesmo.validation_framework.validation.Validation``, use
      ``temporal_matcher=make_combined_temporal_matcher(<window>)`` as keyword
      argument. ``make_combined_temporal_matcher`` can be imported from
      ``pytesmo.validation_framework.temporal_matchers``.

    Parameters
    ----------
    min_obs : int, optional
        Minimum number of observations required to calculate metrics. Default
        is 10.
    calc_spearman : bool, optional
        Whether to calculate Spearman's rank correlation coefficient. Default
        is True.
    calc_kendall : bool, optional
        Whether to calculate Kendall's rank correlation coefficient. Default is
        True.
    analytical_cis : bool, optional (default: True)
        Whether to calculate analytical confidence intervals for the following
        metrics:
            - BIAS
            - mse_bias
            - RMSD
            - urmsd
            - mse
            - R
            - rho (only if ``calc_spearman=True``)
            - tau (only if ``calc_kendall=True``)
    bootstrap_cis: bool, optional (default: False)
        Whether to calculate bootstrap confidence intervals for the following
        metrics:
            - mse_corr
            - mse_var
        The default is `False`. This might be a lot of computational effort.
    bootstrap_min_obs: int, optional (default: 100)
        Minimum number of observations to draw from the time series for boot-
        strapping.
    bootstrap_alpha: float, optional (default: 0.05)
        Confidence level.
    """

    def __init__(
            self,
            min_obs=10,
            calc_spearman=True,
            calc_kendall=True,
            analytical_cis=True,
            bootstrap_cis=False,
            bootstrap_min_obs=100,
            bootstrap_alpha=0.05,
            metadata_template=None,
    ):
        super().__init__(min_obs=min_obs, metadata_template=metadata_template)

        self.calc_spearman = calc_spearman
        self.calc_kendall = calc_kendall
        self.analytical_cis = analytical_cis
        self.bootstrap_cis = bootstrap_cis
        self.bootstrap_min_obs = bootstrap_min_obs
        self.bootstrap_alpha = bootstrap_alpha

        metrics = self._pairwise_metric_names()
        metrics.append("n_obs")
        self.result_template.update(_get_metric_template(metrics))

    def calc_metrics(self, data, gpi_info):
        """
        Calculates pairwise metrics.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 2 columns between which metrics should be
            calculated.
        gpi_info : tuple
            (gpi, lon, lat)
        """
        result = super().calc_metrics(data, gpi_info)

        n_obs = len(data)
        result["n_obs"][0] = n_obs
        if n_obs < self.min_obs:
            result["status"][0] = eh.INSUFFICIENT_DATA
            warnings.warn(
                "Not enough observations to calculate metrics.", UserWarning
            )
            return result

        data_matrix = data.values
        x = data_matrix[:, 0]
        y = data_matrix[:, 1]

        # we can calculate almost all metrics from moments
        mx, my, varx, vary, cov = _moments_welford(x, y)
        self._calc_pairwise_metrics(x, y, mx, my, varx, vary, cov, result)
        result["status"][0] = eh.OK
        return result


class TripleCollocationMetrics(MetadataMetrics, PairwiseMetricsMixin):
    """
    Computes triple collocation metrics

    The triple collocation metrics calculated are:

    - SNR
    - error standard deviation
    - linear scaling/multiplicative (first-order) bias

    **NOTE**: When using this within a
      ``pytesmo.validation_framework.validation.Validation``, use
      ``temporal_matcher=make_combined_temporal_matcher(<window>)`` as keyword
      argument. ``make_combined_temporal_matcher`` can be imported from
      ``pytesmo.validation_framework.temporal_matchers``.

    Parameters
    ----------
    refname : str
        Name of the reference column that is passed to ``calc_metrics``. This
        will also be used to name the results. **Make sure that you set
        ``rename_cols=False`` in the call to ``Validation.calc``, otherwise the
        names will be wrong.**
    min_obs : int, optional
        Minimum number of observations required to calculate metrics. Default
        is 10.
    bootstrap_cis
        Whether to calculate bootstrap confidence intervals for triple
        collocation metrics.
        The default is `False`. This might be a lot of computational effort.
    bootstrap_min_obs: int, optional (default: 100)
        Minimum number of observations to draw from the time series for boot-
        strapping.
    bootstrap_alpha: float, optional (default: 0.05)
        Confidence level.
    metadata_template: dict, optional (default: None)
        A dictionary containing additional fields (and types) of the form
        dict = {'field': np.float32([np.nan]}. Allows users to specify
        information in the job tuple, i.e. jobs.append((idx,
        metadata['longitude'], metadata['latitude'], metadata_dict)) which
        is then propagated to the end netCDF results file.
    """

    def __init__(
            self,
            refname,
            min_obs=10,
            bootstrap_cis=False,
            bootstrap_min_obs=100,
            bootstrap_alpha=0.05,
            metadata_template=None,
    ):

        super().__init__(min_obs=min_obs, metadata_template=metadata_template)

        self.bootstrap_cis = bootstrap_cis
        self.bootstrap_min_obs = bootstrap_min_obs
        self.bootstrap_alpha = bootstrap_alpha
        self.refname = refname
        self.result_template.update(_get_metric_template(["n_obs"]))

    def _get_metric_template(self, refname, othernames):
        # othernames must have length 2 here!
        result_template = {}
        tcol_metrics = ["snr", "err_std", "beta"]
        tcol_template = _get_tc_metric_template(
            tcol_metrics, (refname, *othernames)
        )
        result_template.update(copy.copy(tcol_template))
        if self.bootstrap_cis:
            for metric in tcol_metrics:
                for ds in [refname] + othernames:
                    result_template[(metric + "_ci_lower", ds)] = np.array(
                        [np.nan], dtype=np.float32
                    )
                    result_template[(metric + "_ci_upper", ds)] = np.array(
                        [np.nan], dtype=np.float32
                    )
        return result_template

    def calc_metrics(self, data, gpi_info):
        """
        Calculates triple collocation metrics.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with one reference column and two other columns between
            which the metrics are calculated. The name of the reference column
            must be the same as used in the constructor.
            Make sure to use ``rename_cols`` for ``Validation.calc``, so that
            the names are correct.
        gpi_info : tuple
            (gpi, lon, lat)
        """
        result = super().calc_metrics(data, gpi_info)
        n_obs = len(data)
        result["n_obs"][0] = n_obs

        # get the remaining metrics template for this specific combination
        othernames = list(data.columns)
        othernames.remove(self.refname)
        result.update(self._get_metric_template(self.refname, othernames))

        if n_obs < self.min_obs:
            result["status"][0] = eh.INSUFFICIENT_DATA
            warnings.warn(
                "Not enough observations to calculate metrics.", UserWarning
            )
            return result

        # calculate triple collocation metrics
        ds_names = (self.refname, *othernames)
        arrays = (data[name].values for name in ds_names)
        if not self.bootstrap_cis:
            try:
                res = tcol_metrics(*arrays)
                for i, name in enumerate(ds_names):
                    for j, metric in enumerate(["snr", "err_std", "beta"]):
                        result[(metric, name)][0] = res[j][i]
                result["status"][0] = eh.OK
            except ValueError:
                result["status"] = eh.METRICS_CALCULATION_FAILED
        else:
            try:
                # handle failing bootstrapping because e.g.
                # too small sample size
                res = tcol_metrics_with_bootstrapped_ci(
                    *arrays, minimum_data_length=self.bootstrap_min_obs,
                    alpha=self.bootstrap_alpha)
                for i, name in enumerate(ds_names):
                    for j, metric in enumerate(["snr", "err_std", "beta"]):
                        result[(metric, name)][0] = res[j][0][i]
                        result[(metric + "_ci_lower", name)][0] = res[j][1][i]
                        result[(metric + "_ci_upper", name)][0] = res[j][2][i]
                result["status"][0] = eh.OK
            except ValueError:
                # if the calculation fails, the template results (np.nan) are used
                result["status"] = eh.METRICS_CALCULATION_FAILED
        return result
