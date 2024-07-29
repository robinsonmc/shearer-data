# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:08:43 2019

@author: mrobinson2
"""
import STFT as stft
import pandas as pd
import time_domain_processing as tdp
import numpy as np

#Both equivalent..........

def muscle_mean_freq(f,t,spectogram):
    
    #pd.set_option('plotting.backend', 'pandas_bokeh')
    sum_if = 0
    sum_i = 0
    for i,v in spectogram.iterrows():
        if f[i] < 5 or f[i] > 500:
            pass
        else:
            sum_if += v*f[i]
            sum_i += v

    mean_f = sum_if/sum_i
    
    
    ABC = tdp.window_rms(mean_f,250)
    
    #Return the mean freq and RMS df
    temp_df_mean_f = pd.DataFrame([t,mean_f])
    df_mean_f = temp_df_mean_f.T
    df_mean_f.columns = ['time','mean_f']
    del temp_df_mean_f
    
    temp_df_RMS = pd.DataFrame([t[249:], ABC])
    df_RMS = temp_df_RMS.T
    df_RMS.columns = ['time','RMS']
    del temp_df_RMS
    
    #df_mean_f.plot_bokeh('time','mean_f',kind='line')
    
    
    return df_mean_f, df_RMS

def muscle_spec_moment_ratio(f,t,spectogram, ind_1, ind_2):
    #Tested and working
    #pd.set_option('plotting.backend', 'pandas_bokeh')
    sum_if = 0
    sum_i = 0
    for i,v in spectogram.iterrows():
        if f[i] < 5 or f[i] > 500:
            pass
        else:
            sum_if += v*f[i]**ind_1
            sum_i += v*f[i]**ind_2

    mean_f = sum_if/sum_i
    
    #Return the mean freq and RMS df
    temp_df_mean_f = pd.DataFrame([t,mean_f])
    df_mean_f = temp_df_mean_f.T
    df_mean_f.columns = ['time','spec_moment_p']
    del temp_df_mean_f 
    
    return df_mean_f


def muscle_spec_moment(f,t,Zxx,p):
    #SUSPECT THIS IS INCORRECT
    
    '''
    f,t,Zxx - output of spectogram function
    p       - desired spectral moment (1 for mean frequency)
    
    Calculated as - 
    For column i in Zxx:
        \sum_{j = 0}^{end}  (f[j]*Zxx[i][j]^p)
        --------------------------------------
        \sum_{j = 0}^{end}    (Zxx[i][j]^p)
           
    
    Output is a dataframe containing the mean frequency for each time t
    Columns: 'time', 'spec_moment_p'
    '''
    instant_spec_moment = np.zeros(len(Zxx.columns)-1)
    for i in range(len(Zxx.columns)-1):
        sum_freq_times_amp = 0
        sum_amp            = 0
        for j in range(len(Zxx[i])):
            sum_amp += Zxx[i][j]**p
            sum_freq_times_amp += f[j]*Zxx[i][j]**p
    
        instant_spec_moment[i] = sum_freq_times_amp/sum_amp
        
    
    df = pd.DataFrame()
    df['time']          = t[:-1]
    df['spec_moment_p'] = instant_spec_moment
    #df_rms = pd.DataFrame()
    #df_rms['spec_moment_rms'] = tdp.window_rms(instant_spec_moment,250)
    #df_rms['time'] = t[249:-1]
    return df

if __name__ == '__main__':
    pass