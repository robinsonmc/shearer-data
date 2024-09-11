# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:39:21 2020

@author: mrobinson2
"""

import pandas as pd
import models_movie as mm
import meta_data as md
from meta_data import MetaData
import numpy as np

'''Stitch the files together by searching metadata'''
#reset_index

def get_data_pickle(file, **kwargs):
    '''Keyword args:
        shearer  (1-4)
        day      (mon-fri)
        run      (1-4)
        part     (1-)
        dir_path (str)
        length   (1-)
        '''
        
    import pickle
    
    with open(file,'rb') as file:
        metadata_list = pickle.load(file)[1]
    
    return get_data(metadata_list, **kwargs)

def get_data(metadata_list,**kwargs):
    '''Keyword args:
        shearer  (1-11)
        day      (mon-fri)
        run      (1-4)
        part     (1-)
        dir_path (str)
        length   (1-)
        '''
    
    for key, value in kwargs.items():
        if str(key) == 'length':
            Q = [x for x in metadata_list if getattr(x,key) > value]
        else:
            Q = [x for x in metadata_list if getattr(x,key) == value]
        metadata_list = Q
        
    return metadata_list


def merge_data(metadata_list):
    '''
    Input a metadata list to merge together, into a full run
    '''
    
    from operator import attrgetter
    
    #Sort on part number
    sorted_metadata_list = sorted(metadata_list, key = attrgetter('part'))
    
    DataFile_list = [mm.DataFile(x.dir_path) for x in sorted_metadata_list]
    df_list = [x.xsens_data for x in DataFile_list]
     
    del DataFile_list
    
    df_list = [x.reset_index() for x in df_list]
    result = pd.concat(df_list,ignore_index = True)
    
    return result

def split_data(df, length=30):
    '''
    Input the dataframe and the desired length in minutes
    '''
    
    samples = length*60*60
    
    result = []
    for i in range(10):
        if i*samples+samples > len(df): break
        result.append(df.iloc[i*samples:i*samples+samples])
    
    return [x.reset_index() for x in result]


def norm_3d_df(myData):
    
    A = myData['Pelvis_T8_x']
    B = myData['Pelvis_T8_y']
    C = myData['Pelvis_T8_z']
    
    Q  = np.square(A) + np.square(B) + np.square(C)
    QQ = np.sqrt(Q)
    
    return QQ

def get_lyap(myData,ax):
    from max_lyap_kantz import create_max_dim_embedding,\
    get_n_dim_from_embedding, max_lyap

    #Requires a data load
    Q = norm_3d_df(myData)
    #Q = Q[-230000:-30000]
    
    #Found J = 220
    max_embed = create_max_dim_embedding(220,Q,6)
    result = get_n_dim_from_embedding(max_embed, 6)
    to_fit = max_lyap(result,ax)
    #Import from sklearn
    #First 5 seconds of data
    #Limit the length of the dataset - from the return of nn-calc
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(np.arange(len(to_fit)).reshape(-1,1)/60, to_fit.reshape(-1,1))
    pred = reg.predict(np.arange(len(to_fit)).reshape(-1,1)/60)
    #ax.plot(np.arange(len(to_fit)).reshape(-1,1)/60,pred, '--', alpha=0.6)
    
    return reg

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    shearer = 3
    max_run = 4
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df_list = []
    for element in [15, 20, 30, 45, 60]:
        Q = []
        for i in range(max_run):
            metadata_list = get_data_pickle('metadata.pickle',shearer=shearer, run=i+1)
            full_run = merge_data(metadata_list)
            fr_split = split_data(full_run,length=element)
            
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            regs = []
            for i in range(len(fr_split)):
                regs.append(get_lyap(fr_split[i], ax))
            Q.append([x.coef_ for x in regs])
            
        #plt.figure(figsize=(20,10))    
        #total = 0
        #for i in range(len(Q)):
        #    plt.plot(np.arange(len(Q[i]))+total,[float(x) for x in Q[i]],'x')
        #    total += len(Q[i])
        #    plt.title('Shearer {} - {} minutes'.format(shearer,element))
        #path = Path('C:\\Users\\mrobinson2\\Documents\\lyap_test\\TEST_s{}_{}mins.png'.format(shearer,element))
        #plt.savefig(path)
        
        #Resize the titels and axis labels and crosses to mark
        df = pd.DataFrame(Q).T
        cols = ['Run 1', 'Run 2', 'Run 3', 'Run 4']
        num_cols = len(df.columns)
        df.columns = cols[0:num_cols]
        
        df_list.append(df)
        
    import plotting.lyap_plotting as lp
    lp.get_plots(df_list,shearer=3)
        