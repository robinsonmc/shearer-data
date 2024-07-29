# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:28:55 2019

Calculate time-domain features of EMG signal from dataframe input
dataframe output.


@author: mrobinson2
"""

import scipy.signal as ss
import read_delsys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#Set the default backend for df plotting to bokeh
#pd.set_option('plotting.backend', 'pandas_bokeh')

def normalise(envelope_df,mvc_dict):
    # DO NOT NORMALISE TIME COLUMN
    for colname, coldata in envelope_df.iteritems():
        print(colname)
        if colname != 'time':
            envelope_df[colname] = coldata/mvc_dict[colname]
    
    return envelope_df

def extract_mvc(path_to_folder):
    '''
    #1 - Get the structure of the folder and fill list with the 
    necessary files
    
    #2 - Take the maximum envelope value from the MVC contractions from each
    sensor -> store these as a set of values (dict?) and pass the dict to
    the function that creates the envelope.'''
    
    #Each day should have a directory - pass the directory for the day
    #Find each file that says MVC
    mvc_filelist = []

    for subdir, dirs, files in os.walk(path_to_folder):
        for file in files: 
            filename = subdir + os.sep + file
            if filename.endswith(".csv") and 'mvc' in filename:
                file_path_to_read = os.path.join(path_to_folder,filename)
                mvc_filelist.append(file_path_to_read)
                continue
            
    #For each file - extract the envelope and save the max value for each
    #sensor
    max_dict_list = []
    env_freq = 3
    for item in mvc_filelist:
        max_dict_list.append(get_maximums(\
                        get_envelope_from_file(item,env_freq)))
    
    
    #Overwrite if larger value found
    mvc_dict = {}
    for max_dict in max_dict_list:
        for key in max_dict:
            if mvc_dict.get(key) is None\
            or max_dict.get(key) > mvc_dict.get(key):
                mvc_dict[key] = max_dict[key]


    #Load all of these to dataframe, perform the rectification on these
    #Once rectified and low pass filtered, take the maximum for each sensor
    return mvc_dict

def get_envelope_from_file(filepath,freq):
    dataframe = read_delsys.read_delsys_csv(filepath)
    env = envelope(dataframe,freq)
    
    #Remove dataframe
    del dataframe
    #Return envelope
    return env

def get_maximums(dataframe):
    max_dict = {}
    for colname, coldata in dataframe.iteritems():
        max_dict[colname] = coldata.max()
        
    return max_dict

def filter_EMG(dataframe):
    '''Use the standard band-pass filtering for EMG signals
    
    6th order Butterworth @ 20-450Hz'''
    label_flag = 0
    if 'labels' in dataframe.columns:
        labels = dataframe['labels']
        dataframe = dataframe.drop('labels',1)
        label_flag = 1
    
    cols = dataframe.columns
    df_index = np.array(dataframe.index,dtype='object')
    
    fs = 2148.148
    b, a = ss.butter(6,[20/(fs/2), 450/(fs/2)], btype='bandpass')
    output = ss.lfilter(b,a,dataframe)
    
    output_df = pd.DataFrame(data = output[:,:],
                             columns = cols)
    output_df.index = df_index
    
    if label_flag == 1:
        output_df['labels'] = labels
    
    return output_df

def bandpass_dff(dataframe,fs,high):
    '''Input the dataframe, sample rate, low and high frequencies for 
    filtering
    
    
    '''
    b, a = ss.butter(6,high/(fs/2), btype='highpass')
    output = ss.lfilter(b,a,dataframe)
    
    
    return output
    

def envelope(dataframe,freq):
    '''
    Take the dataframe and rectify it as per EMG processing towards envelope
    
    input a dataframe and frequency of LP filter for envelop
    output a dataframe of signal envelopes
    '''
    
    cols = dataframe.columns
    
    #index
    #new_index_df = dataframe.set_index(pd.TimedeltaIndex(start ='0 hours',freq='465517 ns'))
    
    #subtract mean
    df_zm = dataframe.loc[:, dataframe.columns != 'time']\
                    - dataframe.loc[:, dataframe.columns != 'time'].mean()
    
    #Rectify
    rect_df = df_zm.abs()
    
    #Low pass filter
    fsq = 2148.148
    b,a = ss.butter(2,freq,btype='lowpass',fs=fsq)
    
    output = ss.lfilter(b,a,rect_df,axis=0)
    
    output_df = pd.DataFrame(data = output[:,:],
                             columns = cols[:-1])
    
    output_df['time'] = dataframe['time']
    #downsampled = output_df.resample('166667 ns')
    
    #Remove the junk
    del rect_df
    del cols
    del df_zm
    del output
    
    
    return output_df
    
def window_rms(a, window_size):
      a2 = np.power(a,2)
      window = np.ones(window_size)/float(window_size)
      return np.sqrt(np.convolve(a2, window, 'valid'))   
    
if __name__ == '__main__':
    pass
