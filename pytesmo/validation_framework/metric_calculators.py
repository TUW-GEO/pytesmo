# Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

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

'''
Created on Sep 24, 2013

Metric calculators useable in together with core

@author: Christoph.Paulik@geo.tuwien.ac.at
'''

import pytesmo.metrics as metrics
import pytesmo.df_metrics as df_metrics
from pytesmo.scaling import scale
from pytesmo.validation_framework.data_manager import get_result_names

import copy
import itertools
import numpy as np


class BasicMetrics(object):
    """
    This class just computes the basic metrics,
    Pearson's R
    Spearman's rho
    optionally Kendall's tau
    RMSD
    BIAS

    it also stores information about gpi, lat, lon
    and number of observations

    Parameters
    ----------
    other_name: string, optional
        Name of the column of the non-reference / other dataset in the
        pandas DataFrame
    calc_tau: boolean, optional
        if True then also tau is calculated. This is set to False by default
        since the calculation of Kendalls tau is rather slow and can significantly
        impact performance of e.g. global validation studies
    """

    def __init__(self, other_name='k1',
                 calc_tau=False):

        self.result_template = {'R': np.float32([np.nan]),
                                'p_R': np.float32([np.nan]),
                                'rho': np.float32([np.nan]),
                                'p_rho': np.float32([np.nan]),
                                'tau': np.float32([np.nan]),
                                'p_tau': np.float32([np.nan]),
                                'RMSD': np.float32([np.nan]),
                                'BIAS': np.float32([np.nan]),
                                'n_obs': np.int32([0]),
                                'gpi': np.int32([-1]),
                                'lon': np.float64([np.nan]),
                                'lat': np.float64([np.nan])}

        self.other_name = other_name
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
        dataset = copy.deepcopy(self.result_template)

        dataset['n_obs'][0] = len(data)
        dataset['gpi'][0] = gpi_info[0]
        dataset['lon'][0] = gpi_info[1]
        dataset['lat'][0] = gpi_info[2]

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
                 calc_tau=False):

        super(BasicMetricsPlusMSE, self).__init__(other_name=other_name,
                                                  calc_tau=calc_tau)
        self.result_template.update({'mse': np.float32([np.nan]),
                                     'mse_corr': np.float32([np.nan]),
                                     'mse_bias': np.float32([np.nan]),
                                     'mse_var': np.float32([np.nan])})

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


class FTMetrics(object):
    """
    This class computes Freeze/Thaw Metrics
    Calculated metrics are:
        SSF frozen/temp unfrozen
        SSF unfrozen/temp frozen
        SSF unfrozen/temp unfrozen
        SSF frozen/temp frozen
    it also stores information about gpi, lat, lon
    and number of total observations
    """

    def __init__(self, frozen_flag=2,
                 other_name='k1'):

        self.frozen_flag_value = frozen_flag
        self.result_template = {'ssf_fr_temp_un': np.float32([np.nan]),
                                'ssf_fr_temp_fr': np.float32([np.nan]),
                                'ssf_un_temp_fr': np.float32([np.nan]),
                                'ssf_un_temp_un': np.float32([np.nan]),
                                'n_obs': np.int32([0]),
                                'gpi': np.int32([-1]),
                                'lon': np.float64([np.nan]),
                                'lat': np.float64([np.nan])}
        self.other_name = other_name

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
        dataset = copy.deepcopy(self.result_template)

        dataset['n_obs'][0] = len(data)
        dataset['gpi'][0] = gpi_info[0]
        dataset['lon'][0] = gpi_info[1]
        dataset['lat'][0] = gpi_info[2]

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

        return dataset


class BasicSeasonalMetrics(object):
    """
    This class just computes basic metrics on a seasonal basis. It also stores information about
    gpi, lat, lon and number of observations.
    """

    def __init__(self, result_path=None, other_name='k1'):

        self.result_path = result_path
        self.other_name = other_name

        metrics = {'R': np.float32([np.nan]),
                   'p_R': np.float32([np.nan]),
                   'rho': np.float32([np.nan]),
                   'p_rho': np.float32([np.nan]),
                   'n_obs': np.int32([0])}

        self.result_template = {'gpi': np.int32([-1]),
                                'lon': np.float32([np.nan]),
                                'lat': np.float32([np.nan])}

        self.seasons = ['ALL', 'DJF', 'MAM', 'JJA', 'SON']

        for season in self.seasons:
            for metric in metrics.keys():
                key = "{:}_{:}".format(season, metric)
                self.result_template[key] = metrics[metric].copy()

        self.month_to_season = np.array(['', 'DJF', 'DJF', 'MAM', 'MAM',
                                         'MAM', 'JJA', 'JJA', 'JJA', 'SON',
                                         'SON', 'SON', 'DJF'])

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
            Grid point info (i.e. gpi, lon, lat)
        """
        dataset = copy.deepcopy(self.result_template)

        dataset['gpi'][0] = gpi_info[0]
        dataset['lon'][0] = gpi_info[1]
        dataset['lat'][0] = gpi_info[2]

        for season in self.seasons:

            if season != 'ALL':
                subset = self.month_to_season[data.index.month] == season
            else:
                subset = np.ones(len(data), dtype=bool)

            if subset.sum() < 10:
                continue

            x = data['ref'].values[subset]
            y = data[self.other_name].values[subset]
            R, p_R = metrics.pearsonr(x, y)
            rho, p_rho = metrics.spearmanr(x, y)

            dataset['{:}_n_obs'.format(season)][0] = subset.sum()
            dataset['{:}_R'.format(season)][0] = R
            dataset['{:}_p_R'.format(season)][0] = p_R
            dataset['{:}_rho'.format(season)][0] = rho
            dataset['{:}_p_rho'.format(season)][0] = p_rho

        return dataset


class HSAF_Metrics(object):
    """
    This class computes metrics as defined by the H-SAF consortium in
    order to prove the operational readiness of a product. It also stores
    information about gpi, lat, lon and number of observations.
    """

    def __init__(self,
                 other_name1='k1',
                 other_name2='k2',
                 dataset_names=None):

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

        self.result_template = {'gpi': np.int32([-1]),
                                'lon': np.float32([np.nan]),
                                'lat': np.float32([np.nan])}

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
        dataset = copy.deepcopy(self.result_template)

        dataset['gpi'][0] = gpi_info[0]
        dataset['lon'][0] = gpi_info[1]
        dataset['lat'][0] = gpi_info[2]

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


class IntercomparisonMetrics(BasicMetrics):
    """
    Compare Basic Metrics of multiple satellite data sets to one reference data set.
    Pearson's R and p
    Spearman's rho and p
    optionally Kendall's tau
    RMSD
    BIAS
    ubRMSD
    mse

    Parameters
    ----------
    other_names: iterable, optional (default: ['k1', 'k2', 'k3])
        Name of the column of the non-reference / other datasets in the
        pandas DataFrame
    calc_tau: boolean, optional
        if True then also tau is calculated. This is set to False by default
        since the calculation of Kendalls tau is rather slow and can significantly
        impact performance of e.g. global validation studies
    dataset_names : list
        Names of the original datasets, that are used to find the lookup table
        for the df cols.

    """
    def __init__(self, other_names=['k1', 'k2', 'k3'],
                 calc_tau=False, dataset_names=None):

        super(IntercomparisonMetrics, self).__init__(other_name=other_names,
                                                     calc_tau=calc_tau)

        self.df_columns = ['ref'] + self.other_name


        if dataset_names is None:
            self.ds_names = self.df_columns
        else:
            self.ds_names = dataset_names

        self.ds_names_lut = {}
        for name, col in zip(self.ds_names, self.df_columns):
            self.ds_names_lut[col] = name

        self.tds_names = []
        for combi in itertools.combinations(self.df_columns, 2):
            if combi[0] != 'ref': continue # does not validate between non-reference data sets.
            self.tds_names.append("{:}_and_{:}".format(*combi))

        metrics_common = {'n_obs': np.int32([0])}

        metrics_tds = {'R': np.float32([np.nan]),
                       'p_R': np.float32([np.nan]),
                       'rho': np.float32([np.nan]),
                       'p_rho': np.float32([np.nan]),
                       'bias': np.float32([np.nan]),
                       'tau': np.float32([np.nan]),
                       'p_tau': np.float32([np.nan]),
                       'rmsd': np.float32([np.nan]),
                       'mse': np.float32([np.nan]),
                       'mse_corr': np.float32([np.nan]),
                       'mse_bias': np.float32([np.nan]),
                       'ubRMSD': np.float32([np.nan]),
                       'mse_var': np.float32([np.nan])}


        self.result_template = {'gpi': np.int32([-1]),
                                'lon': np.float32([np.nan]),
                                'lat': np.float32([np.nan])}

        for metric in metrics_common.keys():
            key = "{:}".format(metric)
            self.result_template[key] = metrics_common[metric].copy()

        for tds_name in self.tds_names:
            split_tds_name = tds_name.split('_and_')
            tds_name_key = "{:}_{:}".format(self.ds_names_lut[
                                                split_tds_name[0]],
                                            self.ds_names_lut[
                                                split_tds_name[1]])
            for metric in metrics_tds.keys():
                key = "{:}_between_{:}".format(metric, tds_name_key)
                self.result_template[key] = metrics_tds[metric].copy()

    def calc_metrics(self, data, gpi_info):
        """
        calculates the desired statistics

        Parameters
        ----------
        data : pandas.DataFrame
            with >2 columns, the first column is the reference dataset
            named 'ref'
            other columns are the datasets to compare against named 'other_i'
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

        # number of observations
        subset = np.ones(len(data), dtype=bool)

        n_obs = subset.sum()
        if n_obs < 10:
            return dataset

        dataset['n_obs'][0] = n_obs


        # calculate Pearson correlation
        pearson_R, pearson_p = df_metrics.pearsonr(data)
        pearson_R = pearson_R._asdict()
        pearson_p = pearson_p._asdict()

        # calculate Spearman correlation
        spea_rho, spea_p = df_metrics.spearmanr(data)
        spea_rho = spea_rho._asdict()
        spea_p = spea_p._asdict()

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

        # calulcate tau
        if self.calc_tau:
            tau, p_tau = df_metrics.kendalltau(data)
            tau_dict = tau._asdict()
            p_tau_dict = p_tau._asdict()
        else:
            tau = p_tau = p_tau_dict = tau_dict = None

        # No extra scaling is performed here.
        # always scale for ubRMSD with mean std
        #data_scaled = scale(data, method='mean_std')
        # calculate ubRMSD
        ubRMSD_nT = df_metrics.ubrmsd(data)
        ubRMSD_dict = ubRMSD_nT._asdict()

        for tds_name in self.tds_names:
            R = pearson_R[tds_name]
            p_R = pearson_p[tds_name]
            rho = spea_rho[tds_name]
            p_rho = spea_p[tds_name]
            bias = bias_dict[tds_name]
            mse = mse_dict[tds_name]
            mse_corr = mse_corr_dict[tds_name]
            mse_bias = mse_bias_dict[tds_name]
            mse_var = mse_var_dict[tds_name]
            rmsd = rmsd_dict[tds_name]
            ubRMSD = ubRMSD_dict[tds_name]
            if tau_dict and p_tau_dict:
                tau = tau_dict[tds_name]
                p_tau = p_tau_dict[tds_name]


            split_tds_name = tds_name.split('_and_')
            tds_name_key = "{:}_{:}".format(self.ds_names_lut[split_tds_name[0]],
                                            self.ds_names_lut[split_tds_name[1]])

            dataset['R_between_{:}'.format(tds_name_key)][0] = R
            dataset['p_R_between_{:}'.format(tds_name_key)][0] = p_R
            dataset['rho_between_{:}'.format(tds_name_key)][0] = rho
            dataset['p_rho_between_{:}'.format(tds_name_key)][0] = p_rho
            dataset['bias_between_{:}'.format(tds_name_key)][0] = bias
            dataset['mse_between_{:}'.format(tds_name_key)][0] = mse
            dataset['mse_corr_between_{:}'.format(tds_name_key)][0] = mse_corr
            dataset['mse_bias_between_{:}'.format(tds_name_key)][0] = mse_bias
            dataset['mse_var_between_{:}'.format(tds_name_key)][0] = mse_var
            dataset['rmsd_between_{:}'.format(tds_name_key)][0] = rmsd
            dataset['ubRMSD_between_{:}'.format(tds_name_key)][0] = ubRMSD

            if self.calc_tau:
                dataset['tau_between_{:}'.format(tds_name_key)][0] = tau
                dataset['p_tau_between_{:}'.format(tds_name_key)][0] = p_tau

        return dataset



class TCMetrics(object):
    """
    This class computes triple collocation metrics as defined in the QA4SM
    project. It uses 2 satellite and 1 reference data sets as inputs only.
    It can be extended to perform intercomparison between possible triples
    of more than 3 datasets.
    """

    def __init__(self, other_name1='k1', other_name2='k2',
                 calc_tau=False, dataset_names=None):
        '''
        Initialize the QA4SM metrics

        Parameters
        ----------
        other_name1 : str, optional (default: 'k1')
            Name that the first satellite dataset will have as a placeholder in
            the data frame
        other_name2 : str, optional (default: 'k2')
            Name that the second satellite dataset will have as a placeholder in
            the data frame
        calc_tau : bool, optional (default: False)
            Calculate Kendalls Tau (slow)
        dataset_names : list, optional (default: None)
            List that maps the names of the satellite dataset columns to their
            real name that will be used in the results file.
        '''

        self.result_template = {'R': np.float32([np.nan]),
                                'p_R': np.float32([np.nan]),
                                'rho': np.float32([np.nan]),
                                'p_rho': np.float32([np.nan]),
                                'tau': np.float32([np.nan]),
                                'p_tau': np.float32([np.nan]),
                                'RMSD': np.float32([np.nan]),
                                'BIAS': np.float32([np.nan]),
                                'n_obs': np.int32([0]),
                                'gpi': np.int32([-1]),
                                'lon': np.float64([np.nan]),
                                'lat': np.float64([np.nan])}

        self.other_name1, self.other_name2 = other_name1, other_name2
        self.calc_tau = calc_tau
        self.df_columns = ['ref', self.other_name1, self.other_name2]


        if dataset_names is None:
            self.ds_names = self.df_columns
        else:
            self.ds_names = dataset_names

        self.ds_names_lut = {}
        for name, col in zip(self.ds_names, self.df_columns):
            self.ds_names_lut[col] = name

        self.tds_names = []
        for combi in itertools.combinations(self.df_columns, 2):
            if combi[0] != 'ref': continue # only between ref and sat
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
                       'tau': np.float32([np.nan]),
                       'p_tau': np.float32([np.nan]),
                       'rmsd': np.float32([np.nan]),
                       'mse': np.float32([np.nan]),
                       'mse_corr': np.float32([np.nan]),
                       'mse_bias': np.float32([np.nan]),
                       'ubRMSD': np.float32([np.nan]),
                       'mse_var': np.float32([np.nan])}


        self.result_template = {'gpi': np.int32([-1]),
                                'lon': np.float32([np.nan]),
                                'lat': np.float32([np.nan])}

        for metric in metrics_common.keys():
            key = "{:}".format(metric)
            self.result_template[key] = metrics_common[metric].copy()


        # get template for single-dataset metric
        for name in self.ds_names:
            for metric in metrics_sds.keys():
                key = "{:}_{:}".format(name, metric)
                self.result_template[key] = metrics_sds[metric].copy()


        for tds_name in self.tds_names:
            split_tds_name = tds_name.split('_and_')
            tds_name_key = "{:}_{:}".format(self.ds_names_lut[
                                                split_tds_name[0]],
                                            self.ds_names_lut[
                                                split_tds_name[1]])
            for metric in metrics_tds.keys():
                key = "{:}_between_{:}".format(metric, tds_name_key)
                self.result_template[key] = metrics_tds[metric].copy()


    def calc_metrics(self, data, gpi_info):
        """
        calculates the desired statistics

        Parameters
        ----------
        data : pandas.DataFrame
            with >2 columns, the first column is the reference dataset
            named 'ref'
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

        # number of observations
        subset = np.ones(len(data), dtype=bool)

        n_obs = subset.sum()
        if n_obs < 10:
            return dataset

        dataset['n_obs'][0] = n_obs


        # calculate Pearson correlation
        pearson_R, pearson_p = df_metrics.pearsonr(data)
        pearson_R = pearson_R._asdict()
        pearson_p = pearson_p._asdict()

        # calculate Spearman correlation
        spea_rho, spea_p = df_metrics.spearmanr(data)
        spea_rho = spea_rho._asdict()
        spea_p = spea_p._asdict()

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

        # calulcate tau
        if self.calc_tau:
            tau, p_tau = df_metrics.kendalltau(data)
            tau_dict = tau._asdict()
            p_tau_dict = p_tau._asdict()
        else:
            tau = p_tau = p_tau_dict = tau_dict = None

        #data_scaled = scale(data, method='mean_std')
        # calculate ubRMSD
        ubRMSD_nT = df_metrics.ubrmsd(data)
        ubRMSD_dict = ubRMSD_nT._asdict()

        # get single dataset metrics
        # calculate SNR
        x = data[self.df_columns[0]].values[subset]
        y = data[self.df_columns[1]].values[subset]
        z = data[self.df_columns[2]].values[subset]

        snr, err, beta = metrics.tcol_snr(x, y, z)

        for i, name in enumerate(self.ds_names):
            dataset['{:}_snr'.format(name)][0] = snr[i]
            dataset['{:}_err_var'.format(name)][0] = err[i]
            dataset['{:}_beta'.format(name)][0] = beta[i]


        for tds_name in self.tds_names:
            R = pearson_R[tds_name]
            p_R = pearson_p[tds_name]
            rho = spea_rho[tds_name]
            p_rho = spea_p[tds_name]
            bias = bias_dict[tds_name]
            mse = mse_dict[tds_name]
            mse_corr = mse_corr_dict[tds_name]
            mse_bias = mse_bias_dict[tds_name]
            mse_var = mse_var_dict[tds_name]
            rmsd = rmsd_dict[tds_name]
            ubRMSD = ubRMSD_dict[tds_name]
            if tau_dict and p_tau_dict:
                tau = tau_dict[tds_name]
                p_tau = p_tau_dict[tds_name]


            split_tds_name = tds_name.split('_and_')
            tds_name_key = "{:}_{:}".format(self.ds_names_lut[
                split_tds_name[0]],
                self.ds_names_lut[
                split_tds_name[1]])

            dataset['R_between_{:}'.format(tds_name_key)][0] = R
            dataset['p_R_between_{:}'.format(tds_name_key)][0] = p_R
            dataset['rho_between_{:}'.format(tds_name_key)][0] = rho
            dataset['p_rho_between_{:}'.format(tds_name_key)][0] = p_rho
            dataset['bias_between_{:}'.format(tds_name_key)][0] = bias
            dataset['mse_between_{:}'.format(tds_name_key)][0] = mse
            dataset['mse_corr_between_{:}'.format(tds_name_key)][0] = mse_corr
            dataset['mse_bias_between_{:}'.format(tds_name_key)][0] = mse_bias
            dataset['mse_var_between_{:}'.format(tds_name_key)][0] = mse_var
            dataset['rmsd_between_{:}'.format(tds_name_key)][0] = rmsd
            dataset['ubRMSD_between_{:}'.format(tds_name_key)][0] = ubRMSD

            if self.calc_tau:
                dataset['tau_between_{:}'.format(tds_name_key)][0] = tau
                dataset['p_tau_between_{:}'.format(tds_name_key)][0] = p_tau

        return dataset


def get_dataset_names(ref_key, datasets, n=3):
    """
    Get dataset names in correct order as used in the validation framework
        -) reference dataset = ref
        -) first other dataset = k1
        -) second other dataset = k2
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