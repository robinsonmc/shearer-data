# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:33:52 2020

Need to slice the data
- Each file split in two
- Downsample
- Store
- Separate into 3 for training, 1 for testing
- Write to pickle so it can be loaded easily

@author: mrobinson2
"""

from models_movie import DataFile
import numpy as np
import scipy.signal as ss
import pandas as pd
import pickle
import tqdm
from config import GBL_DEBUG

#Definitions
def get_features_labels(myData):
    #myData = mm.DataFile(dir_path)
    #dl = myData.xsens_data.resample('50ms').last()
    #dl = myData.resample('50ms').last()
    
    dq = myData.xsens_data[myData.xsens_data['labels'].notnull()]
    
    dl = dq.iloc[::3,:]
    
    #Extract the features (orig 151)
    flex_angle = np.array(ss.savgol_filter(\
                            dl['Pelvis_T8_z'] + dl['jRightHip_z'],\
                                        151,3,0,mode='constant')).round(0)
    
    flex_vel   = abs(np.array(ss.savgol_filter(\
                            dl['Pelvis_T8_z']+ dl['jRightHip_z'],\
                                        151,3,1,mode='constant')).round(2))
    
    labels = dl['labels']
    
    
    feature_df = pd.DataFrame(data=[flex_angle,flex_vel,labels]).T
    feature_df.columns = ['flex_angle','flex_vel','labels']
    
    return feature_df

#--------
# List the data, total 14 files to get 28 'lists' split into 75/25
#--------
if __name__ == '__main__':
    
    file_list = [
            'D:\Data\Shearer 1 - Test\s1_test_run1_part2',\
            'D:\Data\Shearer 1 - Test\s1_test_run3_part1',\
            'D:\Data\Shearer 1 - Week 1\Tuesday\s1_week1_tuesday_run1_part1',\
            'D:\Data\Shearer 1 - Week 1\Tuesday\s1_week1_tuesday_run3',\
            'D:\Data\Shearer 2 (Week 2) - Monday\s2_week2_monday_run1_part3',\
            'D:\Data\Shearer 2 (Week 2) - Monday\s2_week2_monday_run2_part2',\
            'D:\Data\Shearer 3 (Week 2) - Tuesday\s3_week2_tuesday_run1_part2',\
            'D:\Data\Shearer 3 (Week 2) - Tuesday\s3_week2_tuesday_run4',\
            'D:\Data\Shearer 4 (Week 2) - Wednesday\s4_week2_wednesday_run1',\
            'D:\Data\Shearer 4 (Week 2) - Wednesday\s4_week2_wednesday_run4_part3',\
            'D:\Data\Shearer 5 (Week 2) - Thursday\s5_week2_thursday_run1_part2',\
            'D:\Data\Shearer 5 (Week 2) - Thursday\s5_week2_thursday_run2_part1',\
            'D:\Data\Shearer 6 (Week 2) - Friday\s6_week2_friday_run2_part2',\
            'D:\Data\Shearer 6 (Week 2) - Friday\s6_week2_friday_run3']
    
    #-------
    # For each (pair of ) file(s) in the list load the DataFile:
    # extract features and separate
    #-------
    
    label_list = []
    feature_list = []
    
    for element in tqdm.tqdm(file_list,ascii=True,desc='extracting features'):
        myData = DataFile(element)
        
        feature_df = get_features_labels(myData)
        
        mid_val = int(feature_df.shape[0]/2)
        
        feature_df['labels'].replace(to_replace='other',\
                                          value='catch_drag',inplace=True)
        
        
        label_first_half = feature_df['labels'].iloc[0:mid_val] 
        label_second_half = feature_df['labels'].iloc[mid_val:-1]
        feature_first_half = feature_df.drop('labels',axis=1).iloc[0:mid_val]
        feature_second_half = feature_df.drop('labels',axis=1).iloc[mid_val:-1]
                    
        
        label_list.append(label_first_half)
        label_list.append(label_second_half)
        
        feature_list.append(feature_first_half)
        feature_list.append(feature_second_half)
       
    save_dict = {}
    save_dict['features'] = feature_list
    save_dict['labels'] = label_list
    
    for i in range(0,len(label_list)):
        assert(len(label_list[i]) == len(feature_list[i]))
    
    #with open('C:\\Users\\mrobinson2\\Documents\\EMBC_eval\\EMBC_test_data_test.pickle','wb') as file:
    #    pickle.dump(save_dict,file,protocol=pickle.HIGHEST_PROTOCOL)
        
    
    
    
        
