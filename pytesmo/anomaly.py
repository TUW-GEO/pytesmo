'''
Created on June 20, 2013

@author: Alexander Gruber Alexander.Gruber@geo.tuwien.ac.at
'''

import numpy as np
import pandas as pd
import datetime
from pytesmo.timedate.julian import doy

def calc_anomaly(Ser,
                 window_size=35,
                 climatology=None):
    '''
    Calculates the anomaly of a time series (Pandas series).
    Both, climatology based, or moving-average based anomalies can be 
    calculated
	
    Parameters
    ----------
    Ser : pandas.Series (index must be a DateTimeIndex)
    
    window_size : float, optional
        The window-size [days] of the moving-average window to calculate the
        anomaly reference (only used if climatology is not provided)
        Default: 35 (days)
    
    climatology : pandas.Series (index: 1-366), optional
        if provided, anomalies will be based on the climatology
    
    timespann : [timespan_from, timespan_to], datetime.datetime(y,m,d), optional
        If set, only a subset
        
    Returns
    -------
    anomaly : pandas.Series
        Series containing the calculated anomalies
    '''
    
    if climatology is not None:
        
        Ser = pd.DataFrame(Ser,columns=['absolute'])
        Ser['doy'] = doy(Ser.index.month, Ser.index.day)
        
        clim = pd.DataFrame(climatology,columns=['climatology'])
        
        Ser = Ser.join(clim,on='doy',how='left')
        
        anomaly = Ser['absolute']-Ser['climatology']
        anomaly.index = Ser.index
        

    else:
        reference = moving_average(Ser, window_size=window_size,fast=True)
        anomaly =  Ser - reference
    
    return anomaly
        
        
def calc_climatology(Ser, 
                     moving_avg_orig=5, 
                     moving_avg_clim=30,
                     median=False,
                     timespan=None):
    '''
    Calculates the climatology of a data set
    
    Parameters
    ----------
    Ser : pandas.Series (index must be a DateTimeIndex)
    
    moving_avg_orig : float, optional
        The size of the moving_average window [days] that will be applied on the 
        input Series (gap filling, short-term rainfall correction)
        Default: 5
    
    moving_avg_clim : float, optional
        The size of the moving_average window [days] that will be applied on the 
        calculated climatology (long-term event correction)
        Default: 35
        
    median : boolean, optional
        if set to True, the climatology will be based on the median conditions
    
    timespan : [timespan_from, timespan_to], datetime.datetime(y,m,d), optional
        Set this to calculate the climatology based on a subset of the input
        Series
        
    Returns
    -------
    climatology : pandas.Series
        Series containing the calculated climatology
    '''
                     
    if timespan is not None:
        Ser = Ser.truncate(before=timespan[0], after=timespan[1])
    
    Ser = moving_average(Ser, window_size=moving_avg_orig, sample_to_days=True,fast=True)
    
    Ser = pd.DataFrame(Ser)
    
    Ser['doy'] = doy(Ser.index.month, Ser.index.day)
       
        
    if median:
        clim = Ser.groupby('doy').median()
    else:
        clim = Ser.groupby('doy').mean()
        
    return moving_average(pd.Series(clim.values.flatten(),index=clim.index.values), window_size=moving_avg_clim, no_date=True)
    
def moving_average(Ser, 
                   window_size=1, 
                   no_date=False,
                   sample_to_days=False,fast=False):
    '''
    Applies a moving average (box) filter on an input time series
	
    Parameters
    ----------
    Ser : pandas.Series (index must be a DateTimeIndex)
    
    window_size : float, optional
        The size of the moving_average window [days] that will be applied on the 
        input Series
        Default: 1
    
    no_date : boolean, optional
        Set this if the index is no DateTimeIndex. The window_size will then
        refer to array elements instead of days.
        
    sample_to_days : boolean, optional
        If set the series will be sampled to full days (gaps are filled)
        
    fast: boolean, optional
        uses the pandas implementation which is faster but does fill
        the timeseries end-window/2 with NaN values
        
    Returns
    -------
    Ser : pandas.Series
        moving-average filtered time series
    '''
    
    if not no_date:
        if sample_to_days:
            index = pd.date_range(start=min(Ser.index),end=max(Ser.index),freq='D')
        else:
            index = Ser.index
    else:
        tmp_index = Ser.index.values
        index = pd.date_range('1/1/2000',periods=len(Ser))
        Ser = pd.Series(Ser.tolist(),index=index)
    
    if fast:
        Ser2 = pd.DataFrame(Ser.astype(float))
        
        Ser2['orig_pos']=np.arange(Ser2.index.size)
        
        hourly = Ser2.resample('H',fill_method=None,closed='right')
        avg = pd.rolling_mean(hourly,window_size*24,center=True,min_periods=1)
        
        uniq,uniq_index = np.unique(hourly['orig_pos'].values,return_index=True)
        
        notnan=np.where(~ np.isnan(uniq))
        uniq_index = uniq_index[notnan]
        uniq = uniq[notnan].astype(int)
        avg_values = avg[avg.columns.values[0]].take(uniq_index).values
        
        
        if no_date:
            return pd.Series(avg_values.flatten(),index=tmp_index[uniq])
        
        result2 = pd.Series(avg_values.flatten(),index=Ser2.take(uniq).index)
        
        return result2
    
    result = pd.Series(index=index, dtype='float')
    win = datetime.timedelta(hours=window_size*24/2.)

    for i in result.index:
        result[i] = Ser[i-win:i+win].mean()

    if no_date:
        result = pd.Series(result.tolist(),index=tmp_index)   
        
    return result
    
    
    