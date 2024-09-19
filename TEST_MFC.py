# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:07:54 2020

@author: mrobinson2
"""

#TEST mean freq

import scipy.signal as ss
import numpy as np
import mean_freq_calc as mfc
from scipy import signal
import pandas as pd

T = np.arange(0,10+0.001,0.001)

Y = ss.chirp(T,1,10,10)


def shearer_spectrogram(shearer_data):
    fs = 2148.148148
    f,t,Zxx = signal.spectrogram(shearer_data,fs,noverlap=1024,nperseg=2048)    
    Spec = pd.DataFrame(abs(Zxx)**2)
    return f,t,Spec

def muscle_spec_moment_ratio(f,t,spectogram, ind_1, ind_2):
    #Tested and NOPE
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

def muscle_spec_moment_ratio_from_psd(f,psd, ind_1, ind_2):
    #Tested and NOPE
    #pd.set_option('plotting.backend', 'pandas_bokeh')
    sum_if = 0
    sum_i = 0
    #import matplotlib.pyplot as plt
    #plt.plot(f)
    for i in range(len(psd)):
        if f[i] < 5: 
            pass
        elif f[i] > 500:
            break
        else:
            sum_if += psd[i]*f[i]**ind_1
            sum_i += psd[i]*f[i]**ind_2

    mean_f = sum_if/sum_i
    
    #Return the mean freq and RMS df
    #temp_df_mean_f = pd.DataFrame([t,mean_f])
    #df_mean_f = temp_df_mean_f.T
    #df_mean_f.columns = ['time','spec_moment_p']
    #del temp_df_mean_f 
    
    return mean_f

def psd(data):
    #from matplotlib.mlab import psd
    import matplotlib.pyplot as plt
    
    Pxx,freq = plt.psd(data, NFFT=2048, Fs=2148.148148, \
                           noverlap=1024, scale_by_freq=True, return_line=False)
    
    #Pxx,freq = psd(data, NFFT=2048, Fs=2148.148148, \
    #                       noverlap=1024, scale_by_freq=True, return_line=False)
    
    return Pxx,freq
    
if __name__ == '__main__':
    f,t,Spec = shearer_spectrogram(A['fake'])
    
    
    RESULT = muscle_spec_moment_ratio(f,t,Spec, 1, 0)

