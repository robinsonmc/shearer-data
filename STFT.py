# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:36:25 2019

@author: mrobinson2
"""
import scipy
from scipy import signal
#import matplotlib.pyplot as plt
import pandas as pd
from time import time
#import plotly.express as px

def shearer_spectogram(shearer_data, muscle):
    fs = 2148.14814815
    f,t,Zxx = signal.spectrogram(shearer_data[muscle],fs,noverlap=2048-1024,nperseg=2048)    
    #Should we square it????
    Spec = pd.DataFrame(abs(Zxx))
    #Spec = pd.DataFrame(10*scipy.log(abs(Zxx)))
    return f,t,Spec


def motion_spectogram(dataframe, column):
    fs = 60
    f,t,Zxx = signal.spectrogram(dataframe[column],fs,noverlap=64-32,nperseg=64)    
    Spec = pd.DataFrame(abs(Zxx))
    return f,t,Spec
    
if __name__ == '__main__':
    pass
    #start = time()
    #data = moving_mean_freq(shearer_data_test)    
    #end = time()
    #print('The code ran in {0:0.3f} seconds'.format(end-start))