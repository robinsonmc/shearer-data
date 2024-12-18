# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:58:28 2019

This module is designed to read the delsys CSV files and return the data
as a dask dataframe indexed by datetime

Use function read_delsys_csv(filepath) to return a dask dataframe with data

@author: mrobinson2
"""

import csv
#import dask.dataframe as dd
import pandas as pd
from config import GBL_DEBUG


def get_delsys_metadata(file_path):
    '''Parse the first part and last part of delsys csv file and retrieve 
    the number of header rows, number of data rows, a list containing the
    header row, the sampling frequency, and number of tail rows.
    
    Input: Path to file
    Output: Tuple (num_header_rows, num_data_rows, header_row, sampling_freq)
    '''
    
    counter = 0
    with open(file_path, newline='') as csvfile:
        emgreader = csv.reader(csvfile, delimiter=',')
        #Parse firt part to get header and number of rows to skip
        for row in emgreader:
            if counter == 0:
                first_row = row
            if row[0] == 'X[s]':
                header_row = row
                num_header_rows = counter
                break
            counter += 1
        
    
    first_row_words = first_row[0].split()
    #sampling_frequency = (float(
    #        first_row_words[first_row_words.index('frequency:') + 1]))   
    
    #Test if the sampling frequency is correct 
    #if debug == 1: 
        #print(sampling_frequency)
    #    assert(float(sampling_frequency) == 2148.148)
    
    return (num_header_rows, header_row, 2148.148148)

#Dask builtin readCSV, does not index correctly
    #Try pandas
def read_delsys_csv(filepath):
    '''Read a delsys type csv file generated by Trigno wireless sensors,
    custom function to retrieve the file metadata and then pandas read csv
    to generate dataframe
    
    Input: path to file
    Output: pandas dataframe of the values indexed in order with custom
    time column generated from sample number / sample rate
    '''
    num_header_rows, header_row, sampling_freq\
            = get_delsys_metadata(filepath)
    
    
    #Pandas dataframe
    #columns=header_row[1:32:2]
    columns=header_row[1:len(header_row):2]
    EMG_dataframe = pd.read_csv(filepath, skipinitialspace=True\
                        ,skiprows=num_header_rows,\
                            usecols=columns)
    
    #Add the time column by index / sample frequency
    EMG_dataframe['time'] = EMG_dataframe.index/sampling_freq
    
    return EMG_dataframe

if __name__ == '__main__':
    from time import time
    start = time()
    #Roughly 3 minutes for 5GB files
    shearer_data_test= read_delsys_csv("D:\Data_for_up\Week 4\s11_week4_friday\s11_week4_friday_run1\\s11_week4_friday_run1.csv")
    shearer_data_monday= read_delsys_csv("D:\Data_for_up\Week 1\Tuesday\s1_week1_tuesday_run1_part1\\s1_week1_tuesday_run1_part1.csv")
    end = time()
    print('The code ran in {0:0.3f} seconds'.format(end-start))