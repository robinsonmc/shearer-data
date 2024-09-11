# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:19:55 2020

@author: mrobinson2
"""

'''Segmented feature collection'''

import combine_data as cd
import resample_ensemble as re
import pandas as pd
#import plotting.feature_plotting as fp
from functools import reduce
from tqdm import tqdm
import pathlib
import pickle
from pathlib import Path

debug = 1

class SegmentedDataFile:
    '''
    Class containing a list of Summarys 
        - this is all three datatypes (xsens, emg freq, emg amp) for 
          one shearer for an entire day
          
    If this can't be held in memory then this should be done one run at a
    time, and get features needs to be called over the four runs
    
    Attributes:
        .segmented_data.xsens_all
        .segmented_data.emg_all
        .segmented_data.envelope_all
    
    '''
    def __init__(self, full_metadata_list, shearer, run, day=None):
        '''
        Inputs:
            - the dir_path to the pickle file containing the metadata
            - shearer number (int 1-10)
            - [optional] day (string lower monday-friday)
            
        Attributes:
            - [Summary_1 ... Summary_N]
        ''' 
        
        if day is not None:
            metadata_list = cd.get_data(full_metadata_list, \
                                        shearer=shearer, run=run, day=day)
        else:
            metadata_list = cd.get_data(full_metadata_list, \
                                        shearer=shearer, run=run)
        
        #Check the input
        days = [getattr(x,'day') for x in metadata_list]
        if not all(x == days[0] for x in days):
            raise ValueError('Please specify a day for this shearer: \
                options are - {}'.format(' ,'.join(set(metadata_list))))
        
        
        
        dir_path = getattr(metadata_list[0],'dir_path')
        
        #Need parent dir
        self.Summary = re.Summary(Path(dir_path).parent,run,False)
        
        
        
    
    
class SegmentedFeatureCollection:
    '''
    Class containing a single datafile that is constructed using metadata
    and the get_feature() method.
          
    This will enable a vast reduction in storage as there will be a set of
    features per sheep ~200 for day rather than 2148Hz + 60Hz + 100Hz data
    for 8 hours
    
    Attributes:
    .shearer
        - int: shearer number
    .day
        - str: day of week
    .features
        - List of dataframes: max length 4 (one per run)
        - Within list, dataframe: columns are features, rows are sheep
    '''
    
    
    def __init__(self, full_metadata_list, \
            shearer, day=None, mocap_functions=[], \
                    emg_functions=[], env_functions=[],\
                       mocap_env_functions=[],mocap_shift_functions=[],\
                       emg_shift_functions=[], env_shift_functions=[],\
                       mocap_env_shift_functions=[]):
        '''
        Input the metadata, and a list of functions (for pd.DataFrame)
        
        list of functions to pull from the segmented 
        data files that are run from 
        
        
        '''
        self.shearer = shearer
        self.day = day
        
        
        if (mocap_functions is None) \
            and (emg_functions is None) and (env_functions is None):
                raise ValueError\
                    ('There must be at least one function in the input')
            
        if day is not None:
            metadata_list = cd.get_data(full_metadata_list, \
                                        shearer=shearer, day=day)
        else:
            metadata_list = cd.get_data(full_metadata_list,\
                                        shearer=shearer)
        
        #Check the input
        days = [getattr(x,'day') for x in metadata_list]
        if not all(x == days[0] for x in days):
            raise ValueError('Please specify a day for this shearer: \
                options are - {}'.format(' ,'.join(set(metadata_list))))
        
        
        max_run  = max(set([getattr(x,'run') for x in metadata_list]))
        
        results = []
        for i in range(1,max_run+1):
            segmented_data = SegmentedDataFile(full_metadata_list,\
                                        shearer=shearer, run=i, day=day)
                
            print(segmented_data)
        
            #MoCap
            mocap_features = {}
            for function in tqdm(mocap_functions):
                    tqdm._instances.clear()
                    mocap_features[('mocap_'+function.__name__)]\
                        = [function(x) for x in segmented_data.Summary.xsens_all]
                    
            #EMG
            emg_features = {}
            for function in tqdm(emg_functions):
                    tqdm._instances.clear()
                    emg_features['emg_'+function.__name__]\
                        = [function(x) for x in segmented_data.Summary.emg_all]
        
            #Envelope
            env_features = {}
            for function in tqdm(env_functions):
                    tqdm._instances.clear()
                    env_features['env_'+function.__name__] =\
                        [function(x) for x in segmented_data.Summary.envelope_all]
            
            #Custom - combined: use function(x) for x in segmented_data.Summary...
            mocap_env_features = {}
            for function in tqdm(mocap_env_functions):
                tqdm._instances.clear()
                #[x + y for x,y in zip(l1,l2)]
                mocap_env_features['mocap_env_'+function.__name__]\
                     = [function(df1,df2) for df1,df2 in \
                        zip(segmented_data.Summary.xsens_all,\
                            segmented_data.Summary.envelope_all)]
                        
            #SHIFT FUNCTIONS            
            #MoCap
            mocap_shift_features = {}
            for function in tqdm(mocap_shift_functions):
                    tqdm._instances.clear()
                    mocap_shift_features[('mocap_shift_'+function.__name__)]\
                        = [function(x) for x in segmented_data.Summary.pre_xsens_all]
                    
            #EMG
            emg_shift_features = {}
            for function in tqdm(emg_shift_functions):
                    tqdm._instances.clear()
                    emg_shift_features['emg_shift_'+function.__name__]\
                        = [function(x) for x in segmented_data.Summary.pre_emg_all]
        
            #Envelope
            env_shift_features = {}
            for function in tqdm(env_functions):
                    tqdm._instances.clear()
                    env_shift_features['env_shift_'+function.__name__] =\
                        [function(x) for x in segmented_data.Summary.pre_envelope_all]
            
            #Custom - combined: use function(x) for x in segmented_data.Summary...
            mocap_env_shift_features = {}
            for function in tqdm(mocap_env_shift_functions):
                tqdm._instances.clear()
                #[x + y for x,y in zip(l1,l2)]
                mocap_env_shift_features['mocap_env_shift_'+function.__name__]\
                     = [function(df1,df2) for df1,df2 in \
                        zip(segmented_data.Summary.pre_xsens_all,\
                            segmented_data.Summary.pre_envelope_all)]
            
            #debug = 1
            
            df_to_merge = []
            #This exists
            if mocap_features:
                mocap_feat_df = pd.DataFrame(mocap_features, columns = mocap_features.keys())
                mocap_feat_df['shared_index_to_merge_on'] = mocap_feat_df.index
                df_to_merge.append(mocap_feat_df)
            #This exists
            if emg_features:
                emg_feat_df = pd.DataFrame(emg_features, columns = emg_features.keys())
                emg_feat_df['shared_index_to_merge_on'] = emg_feat_df.index
                df_to_merge.append(emg_feat_df)
            #This exists
            if env_features:
                env_feat_df = pd.DataFrame(env_features, columns = env_features.keys())
                env_feat_df['shared_index_to_merge_on'] = env_feat_df.index
                df_to_merge.append(env_feat_df)
                
            #This exists
            if mocap_env_features:
                mocap_env_feat_df = pd.DataFrame(mocap_env_features, columns = mocap_env_features.keys())
                mocap_env_feat_df['shared_index_to_merge_on'] = mocap_env_feat_df.index
                df_to_merge.append(mocap_env_feat_df)
            
            #--------------
            #SHIFT VERSIONS  
            #--------------
            #This exists
            if mocap_shift_features:
                mocap_shift_feat_df = pd.DataFrame(mocap_shift_features, columns = mocap_shift_features.keys())
                mocap_shift_feat_df['shared_index_to_merge_on'] = mocap_shift_feat_df.index
                df_to_merge.append(mocap_shift_feat_df)
            #This exists
            if emg_shift_features:
                emg_shift_feat_df = pd.DataFrame(emg_shift_features, columns = emg_shift_features.keys())
                emg_shift_feat_df['shared_index_to_merge_on'] = emg_shift_feat_df.index
                df_to_merge.append(emg_shift_feat_df)
            #This exists
            if env_shift_features:
                env_shift_feat_df = pd.DataFrame(env_shift_features, columns = env_shift_features.keys())
                env_shift_feat_df['shared_index_to_merge_on'] = env_shift_feat_df.index
                df_to_merge.append(env_shift_feat_df)
                
            #This exists
            if mocap_env_shift_features:
                mocap_env_shift_feat_df = pd.DataFrame(mocap_env_shift_features, columns = mocap_env_shift_features.keys())
                mocap_env_shift_feat_df['shared_index_to_merge_on'] = mocap_env_shift_feat_df.index
                df_to_merge.append(mocap_env_shift_feat_df)
                
            #run_feat_df = df_to_merge[0]
            run_feat_df = reduce(lambda x, y: pd.merge(x, y, on = 'shared_index_to_merge_on'), df_to_merge)
            #TODO Re-check if this works...
            #run_feat_df.drop('shared_index_to_merge_on',axis=1,inplace=True)
            
            
            
            #for i in range(1,len(df_to_merge)):
            #    print(df_to_merge[i-1].columns)
            #    print('i is: ' + i)
            #    run_feat_df = df_to_merge[i-1].merge(df_to_merge[i], suffixes=(False, False),on='shared_index_to_merge_on')
            
            
            
            results.append(run_feat_df)
            
            #TODO: I would like this stored as dataframes please
            #Run 1 df - columns for everything - all should be same length
            #Each of the emg-features and ev-features can be combined - 1 number per sheep
            #Can interrogate the resulting dataframe for columns
            # to see features... loop through columns and bokeh plot dots
            #colurs for each run, in a scrolling column layout with everything
        
            
        self.features = results


    def plot_features(self, feature_list = None, filename=None):
        if not feature_list:
            feature_list = self.features[0].columns
        shearer = self.shearer
        
        #fig = fp.get_figure()
        #fp.get_plots(self.features, feature_list,shearer=shearer) #P should be a column layout
        #fp.show_plot(p, filename)                 #Does the filename end up here?

    def store_features(self, dir_path):
        feature_list = self.features
        shearer = self.shearer
        day = self.day
        
        for i in range(len(feature_list)):
            save_path = 's{}_{}_run{}_features.csv'.format(shearer,day,i+1)
            saveas    = pathlib.Path(pathlib.Path(dir_path),save_path)
            feature_list[i].to_csv(saveas)
        

if __name__ == '__main__':
    '''['time_ms', 'jL5S1_x', 'jL5S1_y', 'jL5S1_z', 'jL4L3_x', 'jL4L3_y',
   'jL4L3_z', 'jL1T12_x', 'jL1T12_y', 'jL1T12_z', 'jT9T8_x', 'jT9T8_y',
   'jT9T8_z', 'jT1C7_x', 'jT1C7_y', 'jT1C7_z', 'jC1Head_x', 'jC1Head_y',
   'jC1Head_z', 'jRightT4Shoulder_x', 'jRightT4Shoulder_y',
   'jRightT4Shoulder_z', 'jRightShoulder_x', 'jRightShoulder_y',
   'jRightShoulder_z', 'jRightElbow_x', 'jRightElbow_y', 'jRightElbow_z',
   'jRightWrist_x', 'jRightWrist_y', 'jRightWrist_z', 'jLeftT4Shoulder_x',
   'jLeftT4Shoulder_y', 'jLeftT4Shoulder_z', 'jLeftShoulder_x',
   'jLeftShoulder_y', 'jLeftShoulder_z', 'jLeftElbow_x', 'jLeftElbow_y',
   'jLeftElbow_z', 'jLeftWrist_x', 'jLeftWrist_y', 'jLeftWrist_z',
   'jRightHip_x', 'jRightHip_y', 'jRightHip_z', 'jRightKnee_x',
   'jRightKnee_y', 'jRightKnee_z', 'jRightAnkle_x', 'jRightAnkle_y',
   'jRightAnkle_z', 'jRightBallFoot_x', 'jRightBallFoot_y',
   'jRightBallFoot_z', 'jLeftHip_x', 'jLeftHip_y', 'jLeftHip_z',
   'jLeftKnee_x', 'jLeftKnee_y', 'jLeftKnee_z', 'jLeftAnkle_x',
   'jLeftAnkle_y', 'jLeftAnkle_z', 'jLeftBallFoot_x', 'jLeftBallFoot_y',
   'jLeftBallFoot_z', 'T8_Head_x', 'T8_Head_y', 'T8_Head_z',
   'T8_LeftUpperArm_x', 'T8_LeftUpperArm_y', 'T8_LeftUpperArm_z',
   'T8_RightUpperArm_x', 'T8_RightUpperArm_y', 'T8_RightUpperArm_z',
   'Pelvis_T8_x', 'Pelvis_T8_y', 'Pelvis_T8_z', 'CoG_x', 'CoG_y', 'CoG_z']
    '''
    
    '''
    'L1 Erector Spinae LEFT: EMG.A 1', 'L1 Erector Spinae RIGHT: EMG.A 2',
       'L3 Erector Spinae LEFT: EMG.A 3', 'L3 Erector Spinae RIGHT: EMG.A 4',
       'L5 Multifidus LEFT: EMG.A 5', 'L5 Multifidus RIGHT: EMG.A 6',
       'Rectus Abdominis (1cm up - 3cm out) RIGHT: EMG.A 7',
       'Rectus Abdominis (1cm up - 3cm out) LEFT: EMG.A 8',
       'External Oblique (15cm out) RIGHT: EMG.A 9',
       'External Oblique (15cm out) LEFT: EMG.A 10',
       'Gluteus Medius LEFT: EMG.A 11', 'Gluteus Medius RIGHT: EMG.A 12',
       'Vastus Lateralis RIGHT: EMG.A 13', 'Vastus Lateralis LEFT: EMG.A 14',
       'Biceps Femoris LEFT: EMG.A 15', 'Biceps Femoris RIGHT: EMG.A 16',
       'time', 'HMM_labels', 'labels', 'HMM_UP40', 'HMM_DOWN40', 'OR_HMM',
       'time_shift', 'eval'
    '''

    from pathlib import Path
    from meta_data import MetaData
    #Q = SegmentedDataFile(file='metadata_test.pickle',shearer=7,run=1)
    from get_feature_functions import col_average, col_std,\
                                      apply_to_column, mean_freq_dataframe,\
                                      rmsquare, sample_entropy, ApEn,\
                                      apply_ApEn, apply_to_rel_vec, apply_to_two_columns,\
                                      two_col_ratio, apply_dimjerk2,\
                                      two_df_mult, two_col_norm,\
                                      apply_twocol_twodf, DRP, apply_DRP,\
                                      shannon_entropy, apply_spec_ind,\
                                      spectral_ind_ratio,get_datetime
                           
    mocap_feature_list = [#apply_to_column(sample_entropy,'Pelvis_T8_z'),\
                          #apply_to_rel_vec(rmsquare,'Pelvis','T8','angvel','Pelvis_T8','z')]
                          #apply_to_two_columns(two_col_ratio,'Pelvis_T8_z','jRightHip_z'),\
                          #apply_to_column(rmsquare,'Pelvis_T8_y'),\
                          #apply_DRP(DRP,'Pelvis_T8_z','jRightHip_z',1),\
                          #apply_DRP(DRP,'Pelvis_T8_z','jRightHip_z',2),\
                          #apply_DRP(DRP,'Pelvis_T8_z','jRightHip_z',3),\
                          #apply_to_column(shannon_entropy,'Pelvis_T8_z')]
#                          apply_SPARC('T8','Pelvis','Pelvis_T8','z'),\
#                          apply_dimjerk2('T8','Pelvis','Pelvis_T8','z')]                           
#    mocap_feature_list = [apply_to_column(col_average,'Pelvis_T8_z'),\
#                          apply_to_column(rmsquare,'Pelvis_T8_x'),\
#                          apply_to_column(rmsquare,'Pelvis_T8_y'),\
#                          apply_to_column(col_std,'Pelvis_T8_z'),\
#                          apply_to_column(col_average,'jRightHip_z'),\
                          apply_to_column(col_average,'jRightKnee_z')]
#                          apply_to_column(col_average,'jRightT4Shoulder_z'),\
#                          apply_to_column(col_average,'T8_Head_z'),\
#                          gff.time_taken_s]
    
    envelope_feature_list = []
    #[apply_to_two_columns(two_col_ratio,'L1 Erector Spinae LEFT','Rectus Abdominis RIGHT')]
#                              apply_to_column(col_average,'L1 Erector Spinae LEFT: EMG.A 1'),\
#                             apply_to_column(col_average,'L1 Erector Spinae RIGHT: EMG.A 2'),\
#                             apply_to_column(col_average,'L3 Erector Spinae LEFT: EMG.A 3'),\
#                             apply_to_column(col_average,'L3 Erector Spinae RIGHT: EMG.A 4'),\
#                             apply_to_column(col_average,'L5 Multifidus LEFT: EMG.A 5'),\
#                             apply_to_column(col_average,'L5 Multifidus RIGHT: EMG.A 6'),\
#                             apply_to_column(col_average,'Rectus Abdominis (1cm up - 3cm out) RIGHT: EMG.A 7'),\
#                             apply_to_column(col_average,'Rectus Abdominis (1cm up - 3cm out) LEFT: EMG.A 8'),\
#                             apply_to_column(col_average,'External Oblique (15cm out) RIGHT: EMG.A 9'),\
#                             apply_to_column(col_average,'External Oblique (15cm out) LEFT: EMG.A 10'),\
#                             apply_to_column(col_average,'Gluteus Medius LEFT: EMG.A 11'),\
#                             apply_to_column(col_average,'Gluteus Medius RIGHT: EMG.A 12'),\
#                             apply_to_column(col_average,'Vastus Lateralis RIGHT: EMG.A 13'),\
#                             apply_to_column(col_average,'Vastus Lateralis LEFT: EMG.A 14'),\
#                             apply_to_column(col_average,'Biceps Femoris LEFT: EMG.A 15'),\
#                             apply_to_column(col_average,'Biceps Femoris RIGHT: EMG.A 16')]
#    
    emg_feature_list = [get_datetime]
                        #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',1,0),\
                        #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,2),\
                        #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,3),\
                        #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,4),\
                        #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,5),\
                        #apply_to_column(mean_freq_dataframe,'L1 Erector Spinae RIGHT'),\
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',1,0),\
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,2),\
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,3),\
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,4),\
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,5)]
                        #apply_to_column(mean_freq_dataframe,'L5 Multifidus LEFT'),\
                        #apply_to_column(mean_freq_dataframe,'L5 Multifidus RIGHT')]
#                        apply_to_column(mean_freq_dataframe,'Rectus Abdominis (1cm up - 3cm out) RIGHT: EMG.A 7'),\
#                        #Rectus Abdominis (1cm up - 3cm out) LEFT: EMG.A 8
#                        apply_to_column(mean_freq_dataframe,'Rectus Abdominis (1cm up - 3cm out) LEFT: EMG.A 8'),\
#                        apply_to_column(mean_freq_dataframe,'External Oblique (15cm out) RIGHT: EMG.A 9'),\
#                        apply_to_column(mean_freq_dataframe,'External Oblique (15cm out) LEFT: EMG.A 10'),\
#                        apply_to_column(mean_freq_dataframe,'Gluteus Medius LEFT: EMG.A 11'),\
#                        apply_to_column(mean_freq_dataframe,'Gluteus Medius RIGHT: EMG.A 12'),\
#                        apply_to_column(mean_freq_dataframe,'Vastus Lateralis RIGHT: EMG.A 13'),\
#                        apply_to_column(mean_freq_dataframe,'Vastus Lateralis LEFT: EMG.A 14'),\
#                        apply_to_column(mean_freq_dataframe,'Biceps Femoris LEFT: EMG.A 15'),\
#                        apply_to_column(mean_freq_dataframe,'Biceps Femoris RIGHT: EMG.A 16')]
    
    mocap_env_feature_list = []#apply_twocol_twodf(two_df_mult,'Pelvis_T8_z','L3 Erector Spinae LEFT')]
    
    emg_shift_feature_list = [apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',1,0)]
                        #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,2),\
                        #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,3),\
                        #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,4),\
                        #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,5),\
                        #apply_to_column(mean_freq_dataframe,'L1 Erector Spinae RIGHT'),\
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',1,0),\
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,2)]
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,3),\
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,4),\
                        #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,5)]
    
    file = 'metadata_all_shearers.pickle'
    with open(file,'rb') as file:
        full_metadata_list = pickle.load(file)[1]
    
    
    Q = SegmentedFeatureCollection(full_metadata_list,shearer=10, day='thursday',\
                                   mocap_functions = mocap_feature_list,\
                                   env_functions   = envelope_feature_list,\
                                   emg_functions   = emg_feature_list,\
                                   mocap_env_functions = mocap_env_feature_list,\
                                   emg_shift_functions = emg_shift_feature_list)
    #Q.store_features('D:\\Data\\saved_features')
    #Q.plot_features()