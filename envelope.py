# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:27:18 2019

envelope class for emg data

@author: mrobinson2
"""

import scipy.signal as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import read_delsys
from pathlib import Path
from config import GBL_DEBUG


class MVC:
    def __init__(self, dir_path, freq):
        '''
        When creating the MVC class - this will be the dir containing
        the mvc data. This will be the directory above the collected data,
        as the mvc data corresponds to the entire day
        '''
        self.dir_path = Path(dir_path)
        self.freq = freq
        self.mvc_df = self.extract_mvc(self.dir_path,self.freq)
    
    
    def extract_mvc(self,path_to_folder,freq):
        '''
        #1 - Get the structure of the folder and fill list with the 
        necessary files
        
        #2 - Take the maximum envelope value from the MVC contractions from 
        each sensor -> store these as a set of values (dataframe?) store 
        this as an instance variable.'''
        #Does a pickle exist?? Check and just load from that
        mvc_pickle_list = list(Path(path_to_folder).glob('*mvc*.pickle'))
        if len(mvc_pickle_list) > 0:
            mvc_df = pd.read_pickle(path_to_folder / 'mvc.pickle')
            return mvc_df
        
        #Find each file that is a csv with mvc in title
        mvc_filelist = list(Path(path_to_folder).glob('*mvc*.csv'))
        
        ###HERE if the filelist is empty use the first run for all muscles
        #Check the get maximums
        if not mvc_filelist:
            try:
                mvc_filelist.append(path_to_folder / 's2_week2_monday_run1_part1'/ 's2_week2_monday_run1_part1.csv')
            finally:
                pass
        
        #For each file - extract the envelope and save the max value for each
        #sensor
        max_dict_list = []
        env_freq = freq
        for item in mvc_filelist:
            max_dict_list.append(self.get_maximums(\
                            self.get_mvc_from_file(item,env_freq,False)))
        
        
        #Overwrite if larger value found
        mvc_dict = {}
        for max_dict in max_dict_list:
            for key in max_dict:
                if mvc_dict.get(key) is None\
                or max_dict.get(key) > mvc_dict.get(key):
                    mvc_dict[key] = max_dict[key]
    
        if GBL_DEBUG == 1: print(mvc_dict)
        #Load all of these to dataframe, perform the rectification on these
        #Once rectified and low pass filtered, take the maximum for each sensor
        #WRITE PICKLE FILE THEN RETURN
        mvc_df = pd.DataFrame.from_dict(mvc_dict,orient='index')
        mvc_df = mvc_df.T
        mvc_df['freq'] = self.freq
        mvc_df.to_pickle(path_to_folder / 'mvc.pickle')
        return mvc_df
    
    #THIS CALLS DATA IN THE FORM THAT IS CONFUSING
    def get_mvc_from_file(self,filepath,freq,downsample):
        if GBL_DEBUG == 1: print(filepath)
        dataframe = read_delsys.read_delsys_csv(filepath)
        env = Envelope.get_envelope(dataframe,freq,downsample)
        
        #Remove dataframe
        del dataframe
        #Return envelope
        return env

    def get_maximums(self,dataframe):
        max_dict = {}
        for colname, coldata in dataframe.iteritems():
            max_dict[colname] = coldata.max()
            
        return max_dict

class Envelope:
    def __init__(self, dir_path, delsys_data, freq):
        self.dir_path = dir_path
        self.freq = freq
        self.mvc = MVC(self.dir_path, self.freq)
        self.env_df = self.normalise(Envelope.get_envelope(delsys_data,self.freq,True),self.mvc.mvc_df)
    
    @staticmethod
    def get_envelope(dataframe, freq, downsample):
        # First need to rectify
        #   then -subtract mean
        #   then filter
        #   then downsample to 10Hz
        label_flag = 0
        HMM_label_flag = 0
        
        if 'labels' in dataframe.columns:
            labels = dataframe['labels']
            dataframe = dataframe.drop('labels',1)
            label_flag = 1
            
        if 'HMM_labels' in dataframe.columns:
            HMM_labels = dataframe['HMM_labels']
            dataframe = dataframe.drop('HMM_labels',1)
            HMM_label_flag = 1
        
        dataframe = dataframe[dataframe['time'].notnull()]
        index = dataframe.index
        columns = dataframe.columns
        mean = dataframe.mean()
        mean['time'] = 0
        
        if GBL_DEBUG == 1:
            #print('Mean is {}'.format(mean))
            print('Datafame to get envelope for is {}'.format(dataframe))
            print('Dataframe columns are {}'.format(columns))
            #print('Dataframe labels column is {}'.format(dataframe['labels']))
        
        dataframe = dataframe.sub(mean)
        dataframe = dataframe.abs()
        
        #Low pass filter
        fsq = 2148.148
        b,a = ss.butter(2,freq,btype='lowpass',fs=fsq)
        
        output = ss.lfilter(b,a,dataframe,axis=0)
        output = pd.DataFrame(output,index=index,columns=columns)
        
        
        if label_flag == 1: 
            output['labels'] = labels
        if HMM_label_flag == 1: 
            output['HMM_labels'] = HMM_labels
        del dataframe
        
        if downsample == True:
            print('Downsampling...')
            #This downsampling requires a datetime index
            #Why does this need downsampling? This is just for full length data
            downsampled_output = output.resample('100ms').first()
            print('Done...')
            return downsampled_output
        
        return output
        
    
    @staticmethod
    def normalise(envelope_df,mvc_dict):
        if GBL_DEBUG == 1: 
            print('At the start of normalise: {}'.format(envelope_df))
            print('At start the numsamples is {}'.format(len(envelope_df)))
        # DO NOT NORMALISE TIME COLUMN
        for colname, coldata in envelope_df.iteritems():
            if colname != 'time' and colname != 'labels' and colname != 'HMM_labels':
                if GBL_DEBUG == 1: print(colname)
                envelope_df[colname] = coldata/mvc_dict[colname][0]
        
        if GBL_DEBUG == 1:
            print('At the end of normalise return: {}'.format(envelope_df))
            print('Samples now at: {}'.format(len(envelope_df)))
        return envelope_df

if __name__ == '__main__':
    pass
    #myEnv = Envelope('D:\Data\Shearer 1 - Test',3)
    #A = myEnv.extract_mvc(myEnv.dir_path)