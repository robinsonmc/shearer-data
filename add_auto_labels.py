# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:17:52 2019

Using the designed HMM from one run, apply labels to all shearing for shearer 1
Save the labels as an extra column (don't o/write) and re-save as same pickle
file (naming shuold be inferred from folder).

@author: mrobinson2
"""

import pickle
import models_movie as mm
import prepare_features_all as pfa
import hysteresis as hys
from pathlib import Path
import numpy as np
import scipy.signal as ss
import pandas as pd
from config import GBL_DEBUG


#Start with the shearers that worked perfectly
#file_list = ['D:\Data\Shearer 1 - Week 1\Tuesday\s1_week1_tuesday_run2',\
#             'D:\Data\Shearer 3 (Week 2) - Tuesday\s3_week2_tuesday_run4',\
#             'D:\Data\Shearer 4 (Week 2) - Wednesday\s4_week2_wednesday_run2']

#with open('EMBC_markov_model_saggital.pickle','rb') as file:
#    model = pickle.load(file)




#features = pfa.get_features_labels(myData)
#Write a function that:
#    - loads a datafile
#    - extracts the features
#    - loads the model
#    - runs the model and post-processing
#    - saves the labels as a new column in the datafile, and write to file

def label_data_all_shearers(DataFile,model):
    """
    This takes the designed HMM to classify shearing vs catch_drag and applies
    it to the data contained in the given filepath. This should accept a 
    DataFile class and label the data accordingly.
    
    This should return the updated DataFile class now with labels.
    
    NB: there should be a record of what is labelled manually - store as
    HMM_label rather than label.
    """
    
    
    
    myData = DataFile
    
    
    print('Extracting features...')
    features_and_labels = get_features(myData)
    
    
    if 'labels' in features_and_labels.columns:
        feature_df = features_and_labels.drop('labels',axis=1)
    else:
        feature_df = features_and_labels

    print('Done...')
    
    HMM_labels = label_data(feature_df,model)
    
    return HMM_labels
    

def label_data(feature_df,model):
    import numpy as np
    sequences = feature_df
    AA = np.array(sequences,dtype=float)
    BB = [list(x) for x in AA]
    
    #feature_df.values.tolist()
    
    print('Running HMM...')
    result = model.predict(BB,algorithm='viterbi')
    orig_result = hys.asym_hysteresis(result, 150, 95)
    final_result = hys.post_process(orig_result)
    #import matplotlib.pyplot as plt
    #plt.plot(final_result)
    print('Done...')
    return final_result


def update_DF_shearer_new(dir_path,model):
    import pandas as pd
    
    myData = mm.DataFile(dir_path)
    print('Loaded the datafile...')
    #Label, drop first and last, add to dataframe
    labels_to_add = label_data_all_shearers(myData,model)
    if GBL_DEBUG == 1: print(labels_to_add)
    HMM_labels = labels_to_add[1:-1]
    
    start_time = myData.xsens_data.index[0]
    if GBL_DEBUG == 1: print('The start_time is {}'.format(start_time))
    time_range = pd.date_range(start_time, periods = len(HMM_labels), freq ='50ms',tz='Australia/Melbourne')
    if GBL_DEBUG == 1: print('The time range is {}'.format(time_range))
    label_df   = pd.DataFrame(data=HMM_labels,index=time_range,columns=['HMM_labels'])
    if GBL_DEBUG == 1: print('The dataframe label_df is {}'.format(label_df))
    #label_df['labels'] = HMM_labels
    #label_df.index = time_range
    
    if GBL_DEBUG == 1:
        print('the length of HMM_labels is {}'.format(len(HMM_labels)))
        print('the length of the index is {}'.format(len(time_range)))
    
    #myData.xsens_data['HMM_labels'] = HMM_labels
    #Now need to label the emg data
    try:
        del myData.delsys_data['HMM_labels']
        del myData.envelope.env_df['HMM_labels']
        del myData.xsens_data['HMM_labels']
    except KeyError:
        pass
    finally:
        combined = myData.delsys_data.join(label_df['HMM_labels'],how='outer')
        if 'labels' in myData.delsys_data.columns:
            labels_col = myData.delsys_data.labels
            combined = combined.drop('labels',axis=1)
        combined = combined.apply(pd.Series.interpolate,args=('nearest',),dtype='object')
        combined['labels'] = labels_col
        combined = combined[combined['time'].notnull()]
        myData.delsys_data = combined
        
        combined = myData.envelope.env_df.join(label_df['HMM_labels'],how='outer')
        if 'labels' in myData.envelope.env_df.columns:
            labels_col = myData.envelope.env_df.labels
            combined = combined.drop('labels',axis=1)
        combined = combined.apply(pd.Series.interpolate,args=('nearest',),dtype='object')
        combined['labels'] = labels_col
        combined = combined[combined['time'].notnull()]
        myData.envelope.env_df = combined
        print('Labels added to delsys_data...')
        
        combined = myData.xsens_data.join(label_df['HMM_labels'], how='outer')
        if 'labels' in myData.delsys_data.columns:
            labels_col = myData.delsys_data.labels
            combined = combined.drop('labels',axis=1)
        combined = combined.apply(pd.Series.interpolate,args=('nearest',),dtype='object')
        combined['labels'] = labels_col
        combined = combined[combined['time_ms'].notnull()]
        myData.xsens_data = combined
        if GBL_DEBUG == 1: print('The final dataframe is: {}'.format(myData.xsens_data))
        
        myData.write_pickles()
        print('New datafile stored as pickle...')
        if GBL_DEBUG == 1: return (HMM_labels, myData.xsens_data) 
        
def generate_file_list(directory):
    #Generate the file list - need directories with _run*_ in the name
    file_list = []
    for filename in Path(directory).rglob('*_run?'):
        file_list.append(filename)
        
    for filename in Path(directory).rglob('*_run?_part?'):
        file_list.append(filename)
    
    
    #List comprehension
    file_list = [x for x in file_list if 'emg' not in str(x)]
    file_list = [x for x in file_list if 'Shearer 10' not in str(x)]
    file_list = [x for x in file_list if 'Shearer 1 - Week 1\Monday' not in str(x)]
         
    return file_list

#Definitions
def get_features(myData):
    #myData = mm.DataFile(dir_path)
    #dl = myData.xsens_data.resample('50ms').last()
    #dl = myData.resample('50ms').last()
    
    dq = myData.xsens_data
    
    #dl = dq.iloc[::3,:]
    dl = dq.resample('50ms').nearest()
    
    
    if 'D:\Data\Week 2\Shearer 4 (Week 2) - Wednesday\s4_week2_wednesday_run3' in str(myData.dir_path) or \
        'D:\Data\Week 2\Shearer 4 (Week 2) - Wednesday\s4_week2_wednesday_run4' in str(myData.dir_path):
            flex_angle = np.array(ss.savgol_filter(\
                            dl['Pelvis_T8_z'] + dl['jRightHip_z'] - dl['jRightKnee_z'],\
                                        151,3,0,mode='constant')).round(0)
    #Got rid of the head feature - this should actually work (except for normalising)
    else:
        #Extract the features (orig 151,3,0)
        flex_angle = np.array(ss.savgol_filter(\
                                dl['Pelvis_T8_z'] + dl['jRightHip_z'] + dl['jRightKnee_z'],\
                                            151,3,0,mode='constant')).round(0)
    
    mn = np.mean(flex_angle)
    st = np.std(flex_angle)
    
    flex_angle = [x if x < mn+st else mn+st for x in flex_angle]
    
    q = dl['Pelvis_T8_z'] + dl['jRightHip_z']
    q = [x if x < mn+st else mn+st for x in q]
    
    flex_vel   = abs(np.array(ss.savgol_filter(\
                            q, 151,3,1,mode='constant')).round(2))
    
    
    flex_angle = np.divide(flex_angle,max(flex_angle))
    
    flex_vel = np.divide(flex_vel,max(flex_vel))
    
    
    feature_df = pd.DataFrame(data=[flex_angle,flex_vel]).T
    feature_df.columns = ['flex_angle','flex_vel']
    
    return feature_df 

#def get_features(myData):
#    #myData = mm.DataFile(dir_path)
#    #dl = myData.xsens_data.resample('50ms').last()
#    #dl = myData.resample('50ms').last()
#    
#    dq = myData.xsens_data
#    
#    dl = dq.iloc[::3,:]
#    
#    #Extract the features (orig 151)
#    flex_angle = np.array(ss.savgol_filter(\
#                            dl['Pelvis_T8_z'] + dl['jRightHip_z'],\
#                                        151,3,0,mode='constant')).round(0)
#    
#    flex_vel   = abs(np.array(ss.savgol_filter(\
#                            dl['Pelvis_T8_z']+ dl['jRightHip_z'],\
#                                        151,3,1,mode='constant')).round(2))
#    
#    #labels = dl['labels']
#    
#    
#    feature_df = pd.DataFrame(data=[flex_angle,flex_vel]).T
#    feature_df.columns = ['flex_angle','flex_vel']
#    
#    return feature_df
        
if __name__ == '__main__':
    pass
    #    dir_path = 'D:\Data\Week 3\Shearer 2 (Week 3) - Thursday\s2_week3_thursday_run1_part2'
#    update_DF_shearer_new(dir_path,model)
#    myData = mm.DataFile(dir_path)
#    
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.plot(myData.xsens_data['Pelvis_T8_z']+myData.xsens_data['jRightHip_z'],alpha=0.7)
#    plt.plot(myData.xsens_data['HMM_labels']*130)
    
    
    
    #Start with the shearers that worked perfectly
    #file_list = ['D:\Data\Shearer 1 - Week 1\Tuesday\s1_week1_tuesday_run2',\
    #         'D:\Data\Shearer 3 (Week 2) - Tuesday\s3_week2_tuesday_run4',\
    #         'D:\Data\Shearer 4 (Week 2) - Wednesday\s4_week2_wednesday_run2']
    #myData = mm.DataFile(file_list[0])
    
#   test = update_DF_shearer_new(file_list[0],model)
#    import pathlib
#    file_list = generate_file_list('D:\Data')
#    #file_list = file_list[-5:]
#    
#    from tqdm import tqdm
#    for i in tqdm(range(len(file_list))):
#        test = []
#        for filename in file_list[i].rglob('*.mvnx'):
#            test.append(filename)
#        if test:
#            update_DF_shearer_new(str(file_list[i]),model)