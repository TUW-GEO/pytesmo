'''
Created on Apr 17, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

from scipy import stats
import numpy as np
import pandas as pd
import scipy.interpolate as sc_int
import statsmodels.api as sm


def add_scaled(df, method='linreg', label_in=None, label_scale=None):
    """
    takes a dataframe and appends a scaled time series to it. If no labels are
    given the first column will be scaled to the second column of the DataFrame
	
    Parameters
    ----------
    df : pandas.DataFrame
        input dataframe
    method : string
        scaling method
    label_in: string, optional
        the column of the dataframe that should be scaled to that with label_scale
        default is the first column
    label_scale : string, optional
        the column of the dataframe the label_in column should be scaled to
        default is the second column
        
    Returns
    -------
    df : pandas.DataFrame
        input dataframe with new column labeled label_in+'_scaled_'+method
    """
    
    if label_in == None:
        label_in = df.columns.values[0]
    if label_scale == None:
        label_scale = df.columns.values[1]
        
    if method == 'linreg':    
        scaled = linreg(df[label_in].values, df[label_scale].values)
    elif method == 'mean_std':
        scaled = mean_std(df[label_in].values, df[label_scale].values)    
    elif method == 'min_max':
        scaled = min_max(df[label_in].values, df[label_scale].values)
    elif method == 'lin_cdf_match':
        scaled = lin_cdf_match(df[label_in].values, df[label_scale].values)
    elif method == 'cdf_match':
        scaled = cdf_match(df[label_in].values, df[label_scale].values) 
    
    else:
        raise ValueError("method not found")
                
    new_label = label_in + '_scaled_' + method
    
    df[new_label] = pd.Series(scaled, index=df.index)
    
    return df    

def scale(df,method='linreg',reference_index=0):
    """
    takes pandas.DataFrame and scales all columns to the column specified
    by reference_index with the chosen method

    Parameters
    ----------
    df : pandas.DataFrame
        containing matched time series that should be scaled
    method : string, optional
        method definition, has to be a function in globals() that takes 2 numpy.array
        as input and returns one numpy.array of same length
    reference_index : int, optional
        default 0, column index of reference dataset in dataframe
    
    Returns
    -------
    scaled data : pandas.DataFrame
        all time series of the input DataFrame scaled to the one specified by
        reference_index
    """
    dicton = globals()
    try:
        scaling_func = dicton[method]
    except KeyError:
        print 'scaling method not found'
        return None    
    
    reference = df[df.columns.values[reference_index]]
    df = df.drop([df.columns.values[reference_index]],axis=1)
    #new_df = pd.DataFrame
    for series in df:
        df[series] = pd.Series(scaling_func(df[series].values, reference.values),index=df.index)
    
    df.insert(reference_index,reference.name,reference)
    
    return df


def min_max(in_data, scale_to):
    """
    scales the input datasets so that they have the same minimum
    and maximum afterwards
    
    Parameters
    ----------
    in_data : numpy.array
        input dataset which will be scaled
    scale_to : numpy.array
        in_data will be scaled to this dataset
    
    Returns
    -------
    scaled dataset : numpy.array
        dataset in_data with same maximum and minimum as scale_to
    """
    return ((in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data)) * (np.max(scale_to) - np.min(scale_to)) + np.min(scale_to))

def linreg(in_data, scale_to):
    """
    scales the input datasets using linear regression
    
    Parameters
    ----------
    in_data : numpy.array
        input dataset which will be scaled
    scale_to : numpy.array
        in_data will be scaled to this dataset
    
    Returns
    -------
    scaled dataset : numpy.array
        dataset scaled using linear regression
    """
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(in_data, scale_to)
 
    return np.abs(slope) * in_data + intercept
    
def mean_std(in_data, scale_to):
    """
    scales the input datasets so that they have the same mean
    and standard deviation afterwards
    
    Parameters
    ----------
    in_data : numpy.array
        input dataset which will be scaled
    scale_to : numpy.array
        in_data will be scaled to this dataset
    
    Returns
    -------
    scaled dataset : numpy.array
        dataset in_data with same mean and standard deviation as scale_to
    """
    return ((in_data - np.mean(in_data)) / np.std(in_data)) * np.std(scale_to) + np.mean(scale_to) 
      
def lin_cdf_match(in_data, scale_to):
    '''
    computes cumulative density functions of in_data and scale_to at their respective bin-edges 
    by linear interpolation; then matches CDF of in_data to CDF of scale_to  
    
    Parameters
    ----------
    in_data: numpy.array
        input dataset which will be scaled
    scale_to: numpy.array
        in_data will be scaled to this dataset
        
    Returns
    -------
    CDF matched values: numpy.array
        dataset in_data with CDF as scale_to
    '''
      
    percentiles = [0, 5, 10, 30, 50, 70, 90, 95, 100]

    in_data_pctl = np.array(np.percentile(in_data, percentiles))
    scale_to_pctl = np.array(np.percentile(scale_to, percentiles))
    
    uniq_ind = np.unique(in_data_pctl, return_index=True)[1]
    in_data_pctl = in_data_pctl[uniq_ind]
    scale_to_pctl = scale_to_pctl[uniq_ind]

    uniq_ind = np.unique(scale_to_pctl, return_index=True)[1]
    in_data_pctl = in_data_pctl[uniq_ind]
    scale_to_pctl = scale_to_pctl[uniq_ind]

    f = sc_int.interp1d(in_data_pctl, scale_to_pctl)
    
    return f(in_data)     

def cdf_match(in_data, scale_to):
    '''
    1. computes discrete cumulative density functions of in_data- and scale_to at their respective bin_edges; 
    2. computes continuous CDFs by 6th order polynomial fitting; 
    3. CDF of in_data is matched to CDF of scale_to 
     
    Parameters
    ----------
    in_data: numpy.array
        input dataset which will be scaled
    scale_to: numpy.array
        in_data will be scaled to this dataset
        
    Returns
    -------
    CDF matched values: numpy.array
        dataset in_data with CDF as scale_to 
    '''  
    n_bins = 100
    
    in_data_bin_edges = np.linspace(min(in_data), max(in_data), n_bins)
    in_data_ecdf_func = sm.distributions.ECDF(in_data)
    in_data_ecdf = in_data_ecdf_func(in_data_bin_edges)
    
    scale_to_bin_edges = np.linspace(min(scale_to), max(scale_to), n_bins)
    scale_to_ecdf_func = sm.distributions.ECDF(scale_to)
    scale_to_ecdf = scale_to_ecdf_func(scale_to_bin_edges)
    
    ind_in_data = np.where((in_data_ecdf > 0.001) & (in_data_ecdf < 0.999))[0]
    ind_scale_to = np.where((scale_to_ecdf > 0.001) & (scale_to_ecdf < 0.999))[0]
    
    # compute discrete operator
    disc_op = []
    
    for i, value in np.ndenumerate(in_data_ecdf[ind_in_data]):
        
        diff = value - scale_to_ecdf[ind_scale_to]
        minabsdiff = min(np.abs(diff))
        minpos = np.where(np.abs(diff) == minabsdiff)[0][0]
        
        disc_op.append((in_data_bin_edges[ind_in_data[i]]) - (scale_to_bin_edges[ind_scale_to[minpos]]))
   
    # compute continuous operator
    cont_op = np.polyfit(in_data_bin_edges[ind_in_data], disc_op, 6, full=True)
    
    in_data_matched = in_data - np.polyval(cont_op[0], in_data)
    
    return in_data_matched
    




     
