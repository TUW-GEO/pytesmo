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

import copy
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

    def __init__(self, frozen_flag=2):

        self.frozen_flag_value = frozen_flag
        self.result_template = {'ssf_fr_temp_un': np.float32([np.nan]),
                                'ssf_fr_temp_fr': np.float32([np.nan]),
                                'ssf_un_temp_fr': np.float32([np.nan]),
                                'ssf_un_temp_un': np.float32([np.nan]),
                                'n_obs': np.int32([0]),
                                'gpi': np.int32([-1]),
                                'lon': np.float64([np.nan]),
                                'lat': np.float64([np.nan])}

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

        ssf, temp = data['ref'].values, data['other'].values
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
