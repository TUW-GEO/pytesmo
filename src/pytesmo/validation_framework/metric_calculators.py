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

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Metric calculators implement combinations of metrics and structure the output.
"""

import copy
import itertools

from scipy.special import betainc
import pandas as pd
import numpy as np
from numba import jit

import pytesmo.metrics as metrics
import pytesmo.df_metrics as df_metrics
from pytesmo.scaling import scale
from pytesmo.validation_framework.data_manager import get_result_names
from pytesmo.df_metrics import n_combinations


def _get_tc_metric_template(metr, ds_names):
    """ return empty dict to fill TC results into """
    if isinstance(metr, str):
        metr = [metr]

    # metrics that are computed between dataset triples
    met_thds_template = {'snr': np.float32([np.nan]),
                         'err_std': np.float32([np.nan]),
                         'beta': np.float32([np.nan])}

    in_lut = np.isin(np.array(metr), np.array(list(met_thds_template.keys())))
    if not all(in_lut):
        unknown = np.take(metr, np.where(in_lut == False)[0])
        raise ValueError('Unknown metric(s): {}'.format(' '.join(unknown)))

    met_thds = {}
    for m in metr:
        for d in ds_names:
            met_thds[(m, d)] = met_thds_template[m]

    return met_thds


def _get_metric_template(metr):
    """ return empty dict to fill results into """
    if isinstance(metr, str):
        metr = [metr]

    lut = dict()
    # metrics that are equal for all data sets
    met_comm = {'n_obs': np.int32([0])}
    # metrics that are computed between dataset pairs
    met_tds = {'R': np.float32([np.nan]),
               'p_R': np.float32([np.nan]),
               'rho': np.float32([np.nan]),
               'p_rho': np.float32([np.nan]),
               'BIAS': np.float32([np.nan]),
               'tau': np.float32([np.nan]),
               'p_tau': np.float32([np.nan]),
               'RMSD': np.float32([np.nan]),
               'mse': np.float32([np.nan]),
               'RSS': np.float32([np.nan]),
               'mse_corr': np.float32([np.nan]),
               'mse_bias': np.float32([np.nan]),
               'urmsd': np.float32([np.nan]),
               'mse_var': np.float32([np.nan])}

    lut.update(met_comm)
    lut.update(met_tds)

    in_lut = np.isin(np.array(metr), np.array(list(lut.keys())))
    if not all(in_lut):
        unknown = np.take(metr, np.where(in_lut == False)[0])
        raise ValueError('Unknown metric(s): {}'.format(' '.join(unknown)))

    return {m: lut[m] for m in metr}

class MonthsMetricsAdapter(object):
    """ Adapt MetricCalculators to calculate metrics for groups across months """
    def __init__(self, calculator, sets=None):
        """
        Add functionality to a metric calculator to calculate validation metrics
        for subsets of certain months in a time series (e.g. seasonal).
        Parameters
        ----------
        calculator : MetadataMetrics or any child of it
        sets : dict, optional (default: None)
            A dictionary consisting of a set name (which is added to the metric
            name as a suffix) and the list of months that belong to that set.
            If None is passed, we use 4 (seasonal) sets named after the fist
            letter of each month used.
        """
        self.cls = calculator
        if sets is None:
            sets = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
                    'JJA': [6, 7, 8], 'SON': [9, 10, 11],
                    'ALL': list(range(1,13))}

        self.sets = sets

        # metadata metrics and lon, lat, gpi are excluded from applying seasonally
        self.non_seas_metrics = ['gpi', 'lon', 'lat']
        if self.cls.metadata_template is not None:
            self.non_seas_metrics += list(self.cls.metadata_template.keys())

        all_metrics = calculator.result_template
        subset_metrics = {}

        # for each subset create a copy of the metric template
        for name in sets.keys():
            for k, v in all_metrics.items():
                if k in self.non_seas_metrics:
                    subset_metrics[f"{k}"] = v
                else:
                    subset_metrics[f"{name}_{k}"] = v

        self.result_template = subset_metrics

    @staticmethod
    def filter_months(df, months, dropna=False):
        """
        Select only entries of a time series that are within certain month(s)

        Parameters
        ----------
        df : pd.DataFrame
            Time series (index.month must exist) that is filtered
        months : list
            Months for which data is kept, e.g. [12,1,2] to keep data for winter
        dropna : bool, optional (default: False)
            Drop lines for months that are not to be kept, if this is false, the
            original index is not changed, but filtered values are replaced with nan.

        Returns
        -------
        df_filtered : pd.DataFrame
            The filtered series
        """
        dat = df.copy(True)
        dat['__index_month'] = dat.index.month
        cond = ['__index_month == {}'.format(m) for m in months]
        selection = dat.query(' | '.join(cond)).index
        dat.drop('__index_month', axis=1, inplace=True)

        if dropna:
            return dat.loc[selection]
        else:
            dat.loc[dat.index.difference(selection)] = np.nan
            return dat

    def calc_metrics(self, data, gpi_info):
        """
        Calculates the desired statistics, for each set that was defined.

        Parameters
        ----------
        data : pandas.DataFrame
            with 2 columns, the first column is the reference dataset
            named 'ref'
            the second column the dataset to compare against named 'other'
        gpi_info : tuple
            Grid point info (i.e. gpi, lon, lat)
        """
        dataset = self.result_template.copy()

        for setname, months in self.sets.items():
            df = self.filter_months(data, months=months, dropna=True)
            ds = self.cls.calc_metrics(df, gpi_info=gpi_info)
            for metric, res in ds.items():
                if metric in self.non_seas_metrics:
                    k = f"{metric}"
                else:
                    k = f"{setname}_{metric}"
                dataset[k] = res

        return dataset
                

class MetadataMetrics(object):
    """
    This class sets up the gpi info and metadata (if used) in the results template.
    This is used as the basis for all other metric calculators.

    Parameters
    ----------
    other_name: string or tuple, optional
        Name of the column of the non-reference / other dataset in the
        pandas DataFrame
    metadata_template: dictionary, optional
        A dictionary containing additional fields (and types) of the form
        dict = {'field': np.float32([np.nan]}. Allows users to specify information in the job tuple,
        i.e. jobs.append((idx, metadata['longitude'], metadata['latitude'], metadata_dict)) which
        is then propagated to the end netCDF results file.
    """

    def __init__(self, other_name='k1', metadata_template=None):

        self.result_template = {'gpi': np.int32([-1]),
                                'lon': np.float64([np.nan]),
                                'lat': np.float64([np.nan])}

        self.metadata_template = metadata_template
        if self.metadata_template != None:
            self.result_template.update(metadata_template)

        self.other_name = other_name

    def calc_metrics(self, data, gpi_info):
        """
        Adds the gpi info and metadata to the results.

        Parameters
        ----------
        data : pandas.DataFrame
            see individual calculators for more information. not directly used here.
        gpi_info : tuple
            of (gpi, lon, lat)
            or, optionally, (gpi, lon, lat, metadata) where metadata is a dictionary
        """

        dataset = copy.deepcopy(self.result_template)

        dataset['gpi'][0] = gpi_info[0]
        dataset['lon'][0] = gpi_info[1]
        dataset['lat'][0] = gpi_info[2]

        if self.metadata_template != None:
            for key, value in self.metadata_template.items():
                try:
                    dataset[key][0] = gpi_info[3][key]
                except(IndexError):
                    raise Exception('No metadata has been provided to the job. '
                                    'Should be of form {field: metadata_value} using the metadata_template '
                                    'supplied to init function.')

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
        since the calculation of Kendalls tau is rather slow and can significantly
        impact performance of e.g. global validation studies
    metadata_template: dictionary, optional
        A dictionary containing additional fields (and types) of the form
        dict = {'field': np.float32([np.nan]}. Allows users to specify
        information in the job tuple,
        i.e. jobs.append(
            (idx, metadata['longitude'], metadata['latitude'], metadata_dict))
        which is then propagated to the end netCDF results file.
    """

    def __init__(self, other_name='k1', calc_tau=False, metadata_template=None):
        super(BasicMetrics, self).__init__(other_name=other_name,
                                           metadata_template=metadata_template)

        self.basic_metrics = ['n_obs', 'R', 'p_R', 'rho', 'p_rho', 'RMSD',
                              'BIAS', 'tau', 'p_tau']

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

        if len(data) < 10:
            return dataset

        x, y = data['ref'].values, data[self.other_name].values
        R, p_R = metrics.pearsonr(x, y)
        rho, p_rho = metrics.spearmanr(x, y)
        RMSD = metrics.rmsd(x, y)
        BIAS = metrics.bias(x, y)

        dataset['R'][0], dataset['p_R'][0] = R, p_R
        dataset['rho'][0], dataset['p_rho'][0] = rho, p_rho
        dataset['RMSD'][0] = RMSD
        dataset['BIAS'][0] = BIAS
        dataset['n_obs'][0] = len(data)

        if self.calc_tau:
            tau, p_tau = metrics.kendalltau(x, y)
            dataset['tau'][0], dataset['p_tau'][0] = tau, p_tau

        return dataset


class BasicMetricsPlusMSE(BasicMetrics):
    """
    Basic Metrics plus Mean squared Error and the decomposition of the MSE
    into correlation, bias and variance parts.
    """

    def __init__(self, other_name='k1',
                 metadata_template=None):
        super(BasicMetricsPlusMSE, self).__init__(other_name=other_name,
                                                  metadata_template=metadata_template)

        self.mse_metrics = ['mse', 'mse_corr', 'mse_bias', 'mse_var']
        self.result_template.update(_get_metric_template(self.mse_metrics))

    def calc_metrics(self, data, gpi_info):
        dataset = super(BasicMetricsPlusMSE, self).calc_metrics(data, gpi_info)
        if len(data) < 10:
            return dataset
        x, y = data['ref'].values, data[self.other_name].values
        mse, mse_corr, mse_bias, mse_var = metrics.mse(x, y)
        dataset['mse'][0] = mse
        dataset['mse_corr'][0] = mse_corr
        dataset['mse_bias'][0] = mse_bias
        dataset['mse_var'][0] = mse_var

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

    def __init__(self, frozen_flag=2,
                 other_name='k1',
                 metadata_template=None):
        super(FTMetrics, self).__init__(other_name=other_name,
                                        metadata_template=metadata_template)

        self.frozen_flag_value = frozen_flag
        self.result_template.update({'ssf_fr_temp_un': np.float32([np.nan]),
                                     'ssf_fr_temp_fr': np.float32([np.nan]),
                                     'ssf_un_temp_fr': np.float32([np.nan]),
                                     'ssf_un_temp_un': np.float32([np.nan]),
                                     'n_obs': np.int32([0])})

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

        # if len(data) < 10: return dataset

        ssf, temp = data['ref'].values, data[self.other_name].values
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

        dataset['ssf_fr_temp_un'][0] = len(ssf_fr_temp_unfrozen)
        dataset['ssf_fr_temp_fr'][0] = len(ssf_temp_frozen)

        dataset['ssf_un_temp_fr'][0] = len(ssf_un_temp_frozen)
        dataset['ssf_un_temp_un'][0] = len(ssf_temp_unfrozen)

        dataset['n_obs'][0] = len(data)

        return dataset

class HSAF_Metrics(MetadataMetrics):
    """
    This class computes metrics as defined by the H-SAF consortium in
    order to prove the operational readiness of a product. It also stores
    information about gpi, lat, lon and number of observations.
    """

    def __init__(self,
                 other_name1='k1',
                 other_name2='k2',
                 dataset_names=None,
                 metadata_template=None):

        super(HSAF_Metrics, self).__init__(other_name=other_name1,
                                           metadata_template=metadata_template)

        # prepare validation dataset names as provided
        self.other_name1 = other_name1
        self.other_name2 = other_name2
        self.df_columns = ['ref', self.other_name1, self.other_name2]
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

        metrics_common = {'n_obs': np.int32([0])}

        metrics_sds = {'snr': np.float32([np.nan]),
                       'err_var': np.float32([np.nan]),
                       'beta': np.float32([np.nan])}

        metrics_tds = {'R': np.float32([np.nan]),
                       'p_R': np.float32([np.nan]),
                       'rho': np.float32([np.nan]),
                       'p_rho': np.float32([np.nan]),
                       'bias': np.float32([np.nan]),
                       'ubrmsd': np.float32([np.nan])}

        self.seasons = ['ALL', 'DJF', 'MAM', 'JJA', 'SON']

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
                split_tds_name = tds_name.split('_and_')
                tds_name_key = "{:}_{:}".format(self.ds_names_lut[
                    split_tds_name[0]],
                    self.ds_names_lut[
                    split_tds_name[1]])
                for metric in metrics_tds.keys():
                    key = "{:}_{:}_{:}".format(tds_name_key, season, metric)
                    self.result_template[key] = metrics_tds[metric].copy()

        self.month_to_season = np.array(['', 'DJF', 'DJF', 'MAM', 'MAM',
                                         'MAM', 'JJA', 'JJA', 'JJA', 'SON',
                                         'SON', 'SON', 'DJF'])

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

            if season != 'ALL':
                subset = self.month_to_season[data.index.month] == season
            else:
                subset = np.ones(len(data), dtype=bool)

            # number of observations
            n_obs = subset.sum()
            if n_obs < 10:
                continue
            dataset['{:}_n_obs'.format(season)][0] = n_obs

            # get single dataset metrics
            # calculate SNR
            x = data[self.df_columns[0]].values[subset]
            y = data[self.df_columns[1]].values[subset]
            z = data[self.df_columns[2]].values[subset]

            snr, err, beta = metrics.tcol_snr(x, y, z)

            for i, name in enumerate(self.ds_names):
                dataset['{:}_{:}_snr'.format(name, season)][0] = snr[i]
                dataset['{:}_{:}_err_var'.format(name, season)][0] = err[i]
                dataset['{:}_{:}_beta'.format(name, season)][0] = beta[i]

            # calculate Pearson correlation
            pearson_R, pearson_p = df_metrics.pearsonr(data)
            pearson_R = pearson_R._asdict()
            pearson_p = pearson_p._asdict()

            # calculate Spearman correlation
            spea_rho, spea_p = df_metrics.spearmanr(data)
            spea_rho = spea_rho._asdict()
            spea_p = spea_p._asdict()

            # scale data to reference in order to calculate absolute metrics
            data_scaled = scale(data, method='min_max')

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

                split_tds_name = tds_name.split('_and_')
                tds_name_key = "{:}_{:}".format(self.ds_names_lut[
                    split_tds_name[0]],
                    self.ds_names_lut[
                    split_tds_name[1]])

                dataset['{:}_{:}_R'.format(tds_name_key, season)][0] = R
                dataset['{:}_{:}_p_R'.format(tds_name_key, season)][0] = p_R
                dataset['{:}_{:}_rho'.format(tds_name_key, season)][0] = rho
                dataset['{:}_{:}_p_rho'.format(tds_name_key, season)][0] = \
                    p_rho
                dataset['{:}_{:}_bias'.format(tds_name_key, season)][0] = bias
                dataset['{:}_{:}_ubrmsd'.format(tds_name_key, season)][0] = \
                    ubRMSD

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
    other_names: tuple, optional (default: ('k1', 'k2', 'k3))
        Name of the column of the non-reference / other datasets in the
        DataFrame.
    calc_tau: boolean, optional
        if True then also tau is calculated. This is set to False by default
        since the calculation of Kendalls tau is rather slow and can significantly
        impact performance of e.g. global validation studies
    dataset_names : list, optional (default: None)
        Names of the original datasets, that are used to find the lookup table
        for the df cols.
    metadata_template: dict, optional (default: None)
        See MetadataMetrics

    """

    def __init__(self, other_names=('k1', 'k2', 'k3'), calc_tau=False,
                 dataset_names=None, metadata_template=None):

        other_names = list(other_names)
        super(IntercomparisonMetrics, self).__init__(
            other_name=other_names, metadata_template=metadata_template)

        # string that splits the dataset names and metric names in the output
        # e.g. 'metric_between_dataset1_and_dataset2'
        self.ds_names_split, self.metric_ds_split = '_and_', '_between_'

        self.df_columns = ['ref'] + self.other_name

        self.calc_tau = calc_tau

        if dataset_names is None:
            self.ds_names = self.df_columns
        else:
            self.ds_names = dataset_names

        self.ds_names_lut = {}
        for name, col in zip(self.ds_names, self.df_columns):
            self.ds_names_lut[col] = name

        combis = n_combinations(self.df_columns, 2, must_include='ref')
        self.tds_names = []
        for combi in combis:
            self.tds_names.append("{1}{0}{2}".format(
                self.ds_names_split, *combi))

        # metrics that are equal for all datasets
        metrics_common = ['n_obs']
        # metrics that are calculated between dataset pairs
        metrics_tds = ['R', 'p_R', 'rho', 'p_rho', 'BIAS', 'RMSD', 'mse', 'RSS',
                       'mse_corr', 'mse_bias', 'urmsd', 'mse_var', 'tau', 'p_tau']

        metrics_common = _get_metric_template(metrics_common)
        metrics_tds = _get_metric_template(metrics_tds)

        for metric in metrics_common.keys():
            self.result_template[metric] = metrics_common[metric].copy()

        for tds_name in self.tds_names:
            split_tds_name = tds_name.split(self.ds_names_split)
            tds_name_key = \
                self.ds_names_split.join([self.ds_names_lut[split_tds_name[0]],
                                          self.ds_names_lut[split_tds_name[1]]])
            for metric in metrics_tds.keys():
                key = self.metric_ds_split.join([metric, tds_name_key])
                self.result_template[key] = metrics_tds[metric].copy()

        if not calc_tau:
            self.result_template.pop('tau', None)
            self.result_template.pop('p_tau', None)

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

        dataset = super(IntercomparisonMetrics,
                        self).calc_metrics(data, gpi_info)

        subset = np.ones(len(data), dtype=bool)

        n_obs = subset.sum()
        if n_obs < 10:
            return dataset

        dataset['n_obs'][0] = n_obs

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
        mse, mse_corr, mse_bias, mse_var = df_metrics.mse(data)
        mse_dict, mse_corr_dict, mse_bias_dict, mse_var_dict = \
            mse._asdict(), mse_corr._asdict(), mse_bias._asdict(), mse_var._asdict()

        # calculate RSS
        rss = df_metrics.RSS(data)
        rss_dict = rss._asdict()

        # calulcate tau
        if self.calc_tau:
            tau, p_tau = df_metrics.kendalltau(data)
            tau_dict, p_tau_dict = tau._asdict(), p_tau._asdict()
        else:
            tau = p_tau = p_tau_dict = tau_dict = None

        # No extra scaling is performed here.
        # always scale for ubRMSD with mean std
        # calculate ubRMSD
        data_scaled = scale(data, method='mean_std')
        ubRMSD_nT = df_metrics.ubrmsd(data_scaled)
        ubRMSD_dict = ubRMSD_nT._asdict()

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
                [self.ds_names_lut[split_tds_name[0]], self.ds_names_lut[split_tds_name[1]]])

            dataset[self.metric_ds_split.join(['R', tds_name_key])][0] = R
            dataset[self.metric_ds_split.join(['p_R', tds_name_key])][0] = p_R
            dataset[self.metric_ds_split.join(['rho', tds_name_key])][0] = rho
            dataset[self.metric_ds_split.join(
                ['p_rho', tds_name_key])][0] = p_rho
            dataset[self.metric_ds_split.join(
                ['BIAS', tds_name_key])][0] = bias
            dataset[self.metric_ds_split.join(['mse', tds_name_key])][0] = mse
            dataset[self.metric_ds_split.join(
                ['mse_corr', tds_name_key])][0] = mse_corr
            dataset[self.metric_ds_split.join(
                ['mse_bias', tds_name_key])][0] = mse_bias
            dataset[self.metric_ds_split.join(
                ['mse_var', tds_name_key])][0] = mse_var
            dataset[self.metric_ds_split.join(
                ['RMSD', tds_name_key])][0] = rmsd
            dataset[self.metric_ds_split.join(
                ['urmsd', tds_name_key])][0] = ubRMSD
            dataset[self.metric_ds_split.join(['RSS', tds_name_key])][0] = rss

            if self.calc_tau:
                dataset[self.metric_ds_split.join(
                    ['tau', tds_name_key])][0] = tau
                dataset[self.metric_ds_split.join(
                    ['p_tau', tds_name_key])][0] = p_tau

        return dataset


class TCMetrics(MetadataMetrics):
    """
    This class computes triple collocation metrics as defined in the QA4SM
    project. It uses 2 satellite and 1 reference data sets as inputs only.
    It can be extended to perform intercomparison between possible triples
    of more than 3 datasets.
    """

    def __init__(self, other_names=('k1', 'k2'), calc_tau=False, dataset_names=None,
                 metadata_template=None):
        """
        Triple Collocation metrics as implemented in the QA4SM project.

        Parameters
        ----------
        other_names : tuple, optional (default: ('k1', 'k2'))
            Names of the data sets that are not the reference in the data frame.
        calc_tau : bool, optional (default: False)
            Calculate Kendall's Tau (slow)
        dataset_names : tuple, optional (default: None)
            List that maps the names of the satellite dataset columns to their
            real name that will be used in the results file.
        metadata_template: dictionary, optional
            A dictionary containing additional fields (and types) of the form
            dict = {'field': np.float32([np.nan]}. Allows users to specify
            information in the job tuple,
            i.e. jobs.append(
                (idx, metadata['longitude'], metadata['latitude'], metadata_dict))
            which is then propagated to the end netCDF results file.

        """
        self.ref_name = 'ref'
        other_names = list(other_names)
        super(TCMetrics, self).__init__(
            other_name=other_names, metadata_template=metadata_template)

        # string that splits the dataset names and metric names in the output
        # e.g. 'metric_between_dataset1_and_dataset2'
        self.ds_names_split, self.metric_ds_split = '_and_', '_between_'

        self.calc_tau = calc_tau
        self.df_columns = [self.ref_name] + self.other_name

        if dataset_names is None:
            self.ds_names = self.df_columns
        else:
            self.ds_names = dataset_names

        self.ds_names_lut = {}
        for name, col in zip(self.ds_names, self.df_columns):
            self.ds_names_lut[col] = name

        self.tds_names, self.thds_names = self._make_names()

        # metrics that are equal for all datasets
        metrics_common = ['n_obs']
        # metrics that are calculated between dataset pairs
        metrics_tds = ['R', 'p_R', 'rho', 'p_rho', 'BIAS', 'RMSD', 'mse', 'RSS',
                       'mse_corr', 'mse_bias', 'urmsd', 'mse_var', 'tau', 'p_tau']
        # metrics that are calculated between dataset triples
        metrics_thds = ['snr', 'err_std', 'beta']

        metrics_common = _get_metric_template(metrics_common)
        metrics_tds = _get_metric_template(metrics_tds)
        metrics_thds = _get_tc_metric_template(metrics_thds,
                                               [self.ds_names_lut[n] for n in self.df_columns if n != self.ref_name])

        for metric in metrics_common.keys():
            self.result_template[metric] = metrics_common[metric].copy()

        for tds_name in self.tds_names:
            split_tds_name = tds_name.split(self.ds_names_split)
            tds_name_key = \
                self.ds_names_split.join([self.ds_names_lut[split_tds_name[0]],
                                          self.ds_names_lut[split_tds_name[1]]])
            for metric in metrics_tds.keys():
                key = self.metric_ds_split.join([metric, tds_name_key])
                self.result_template[key] = metrics_tds[metric].copy()

        for thds_name in self.thds_names:
            split_tds_name = thds_name.split(self.ds_names_split)
            thds_name_key = \
                self.ds_names_split.join([self.ds_names_lut[split_tds_name[0]],
                                          self.ds_names_lut[split_tds_name[1]],
                                          self.ds_names_lut[split_tds_name[2]]])
            for metric, ds in metrics_thds.keys():
                if not any([self.ds_names_lut[other_ds] == ds
                            for other_ds in thds_name.split(self.ds_names_split)]):
                    continue
                full_name = '_'.join([metric, ds])
                key = self.metric_ds_split.join([full_name, thds_name_key])
                self.result_template[key] = metrics_thds[(metric, ds)].copy()

        if not calc_tau:
            self.result_template.pop('tau', None)
            self.result_template.pop('p_tau', None)

    def _make_names(self):
        tds_names, thds_names = [], []
        combis_2 = n_combinations(
            self.df_columns, 2, must_include=[self.ref_name])
        combis_3 = n_combinations(
            self.df_columns, 3, must_include=[self.ref_name])

        for combi in combis_2:
            tds_names.append(self.ds_names_split.join(combi))

        for combi in combis_3:
            thds_names.append("{1}{0}{2}{0}{3}".format(
                self.ds_names_split, *combi))

        return tds_names, thds_names

    def _tc_res_dict(self, res):
        """name is the TC metric name and res the according named tuple """
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
            with >2 columns, the first column is the reference dataset named 'ref'
            other columns are the data sets to compare against named 'other_i'
        gpi_info : tuple
            of (gpi, lon, lat)

        Notes
        -----
        Kendall tau is calculation is optional at the moment
        because the scipy implementation is very slow which is problematic for
        global comparisons
        """

        dataset = copy.deepcopy(self.result_template)

        dataset['gpi'][0] = gpi_info[0]
        dataset['lon'][0] = gpi_info[1]
        dataset['lat'][0] = gpi_info[2]

        if self.metadata_template != None:
            for key, value in self.metadata_template.items():
                dataset[key][0] = gpi_info[3][key]

        # number of observations
        subset = np.ones(len(data), dtype=bool)

        n_obs = subset.sum()
        if n_obs < 10:
            return dataset

        dataset['n_obs'][0] = n_obs

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
        mse, mse_corr, mse_bias, mse_var = df_metrics.mse(data)
        mse_dict = mse._asdict()
        mse_corr_dict = mse_corr._asdict()
        mse_bias_dict = mse_bias._asdict()
        mse_var_dict = mse_var._asdict()
        # calculate RSS
        rss = df_metrics.RSS(data)
        rss_dict = rss._asdict()
        # calculate ubRMSD
        # todo: we could use the TC derived scaling parameters here?
        data_scaled = scale(data, method='mean_std')
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
        snrs, err_stds, betas = df_metrics.tcol_snr(data, ref_ind=ref_ind)
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
                [self.ds_names_lut[split_thds_name[0]],
                 self.ds_names_lut[split_thds_name[1]],
                 self.ds_names_lut[split_thds_name[2]]])

            for metr, res in dict(snr=snr, err_std=err_std, beta=beta).items():
                for ds, ds_res in res.items():
                    m_ds = "{}_{}".format(metr, self.ds_names_lut[ds])
                    n = '{}{}{}'.format(
                        m_ds, self.metric_ds_split, thds_name_key)
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
                [self.ds_names_lut[split_tds_name[0]], self.ds_names_lut[split_tds_name[1]]])

            dataset[self.metric_ds_split.join(['R', tds_name_key])][0] = R
            dataset[self.metric_ds_split.join(['p_R', tds_name_key])][0] = p_R
            dataset[self.metric_ds_split.join(['rho', tds_name_key])][0] = rho
            dataset[self.metric_ds_split.join(
                ['p_rho', tds_name_key])][0] = p_rho
            dataset[self.metric_ds_split.join(
                ['BIAS', tds_name_key])][0] = bias
            dataset[self.metric_ds_split.join(['mse', tds_name_key])][0] = mse
            dataset[self.metric_ds_split.join(
                ['mse_corr', tds_name_key])][0] = mse_corr
            dataset[self.metric_ds_split.join(
                ['mse_bias', tds_name_key])][0] = mse_bias
            dataset[self.metric_ds_split.join(
                ['mse_var', tds_name_key])][0] = mse_var
            dataset[self.metric_ds_split.join(
                ['RMSD', tds_name_key])][0] = rmsd
            dataset[self.metric_ds_split.join(
                ['urmsd', tds_name_key])][0] = ubRMSD
            dataset[self.metric_ds_split.join(['RSS', tds_name_key])][0] = rss

            if self.calc_tau:
                dataset[self.metric_ds_split.join(
                    ['tau', tds_name_key])][0] = tau
                dataset[self.metric_ds_split.join(
                    ['p_tau', tds_name_key])][0] = p_tau

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
        dict = {'field': np.float32([np.nan]}. Allows users to specify
        information in the job tuple,
        i.e. jobs.append(
            (idx, metadata['longitude'], metadata['latitude'], metadata_dict))
        which is then propagated to the end netCDF results file.

    """

    def __init__(self, other_name='k1', metadata_template=None):

        super(RollingMetrics, self).__init__(
            other_name=other_name, metadata_template=metadata_template)

        self.basic_metrics = ['R', 'p_R', 'RMSD']
        self.result_template.update(_get_metric_template(self.basic_metrics))

    def calc_metrics(self, data, gpi_info, window_size='30d', center=True,
                     min_periods=2):
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

        xy = data.to_numpy()
        timestamps = data.index.to_julian_date().values
        window_size_jd = pd.Timedelta(
            window_size).to_numpy()/np.timedelta64(1, 'D')
        pr_arr, rmsd_arr = rolling_pr_rmsd(
            timestamps, xy, window_size_jd, center, min_periods)

        dataset['time'] = np.array([data.index])
        dataset['R'] = np.array([pr_arr[:, 0]])
        dataset['p_R'] = np.array([pr_arr[:, 1]])
        dataset['RMSD'] = np.array([rmsd_arr[:]])

        return dataset


@jit
def rolling_pr_rmsd(timestamps, data, window_size, center, min_periods):
    """
    Computation of rolling Pearson R.

    Parameters
    ----------
    timestamps : float64
        Time stamps as julian dates.
    data : numpy.ndarray
        Time series data in 2d array.
    window_size : float
        Window size in fraction of days.
    center : bool
        Set window at the center.
    min_periods : int
        Minimum number of observations in window required for computation.

    Results
    -------
    pr_arr : numpy.array
        Pearson R and p-value.
    """
    pr_arr = np.empty((timestamps.size, 2), dtype=np.float32)
    rmsd_arr = np.empty(timestamps.size, dtype=np.float32)
    ddof = 0

    for i in range(timestamps.size):
        time_diff = timestamps - timestamps[i]

        if center:
            inside_window = np.abs(time_diff) <= window_size
        else:
            inside_window = (time_diff <= 0) & (time_diff > -window_size)

        idx = np.nonzero(inside_window)[0]
        n_obs = inside_window.sum()

        if n_obs == 0 or n_obs < min_periods:
            pr_arr[i, :] = np.nan
        else:
            sub1 = data[idx[0]:idx[-1]+1, 0]
            sub2 = data[idx[0]:idx[-1]+1, 1]

            # pearson r
            pr_arr[i, 0] = np.corrcoef(sub1, sub2)[0, 1]

            # p-value
            if np.abs(pr_arr[i, 0]) == 1.0:
                pr_arr[i, 1] = 0.0
            else:
                df = n_obs - 2.
                t_squared = pr_arr[i, 0]*pr_arr[i, 0] * \
                    (df / ((1.0 - pr_arr[i, 0]) * (1.0 + pr_arr[i, 0])))
                x = df / (df + t_squared)
                x = np.ma.where(x < 1.0, x, 1.0)
                pr_arr[i, 1] = betainc(0.5*df, 0.5, x)

            # rmsd
            rmsd_arr[i] = np.sqrt(
                np.sum((sub1 - sub2) ** 2) / (sub1.size - ddof))

    return pr_arr, rmsd_arr


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
        ds_dict[ds] = datasets[ds]['columns']
    ds_names = get_result_names(ds_dict, ref_key, n)
    dataset_names = []
    for name in ds_names[0]:
        dataset_names.append(name[0])

    return dataset_names

if __name__ == '__main__':
    calc = IntercomparisonMetrics(other_names=('k1', 'k2', 'k3'),
                                  calc_tau=False,
                                  metadata_template=dict(meta1=np.array(['TBD']),
                                                         meta2=np.float32([np.nan])))

    adapted = MonthsMetricsAdapter(calc)

    idx = pd.date_range('2000-01-01', '2010-07-21', freq='D')
    df = pd.DataFrame(index=idx,
                      data={'ref': np.random.rand(idx.size),
                            'k1': np.random.rand(idx.size),
                            'k2': np.random.rand(idx.size),
                            'k3': np.random.rand(idx.size)})

    adapted.calc_metrics(df, (0,1,2,{'meta1':'meta', 'meta2':12}))