# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:22:29 2024

set up the the raw data feature extraction

@author: robin
"""

import pickle
import get_all_metadata as gam
import segmented_feature_collection as sfc
from pathlib import Path
from meta_data import MetaData
import pathlib
from get_feature_functions import col_average, col_std,\
                                      apply_to_column, mean_freq_dataframe,\
                                      rmsquare, sample_entropy, ApEn,\
                                      apply_ApEn, apply_to_rel_vec, apply_to_two_columns,\
                                      two_col_ratio, apply_dimjerk2,\
                                      two_df_mult, two_col_norm,\
                                      apply_twocol_twodf, DRP, apply_DRP,\
                                      shannon_entropy, apply_spec_ind,\
                                      spectral_ind_ratio,get_datetime



#Optional steps
#1
extract_metadata = True
#2
segment_data = True
#3
extract_features = True


if extract_metadata:

    full_metadata_list = gam.get_all_metadata("D:\Data_for_up")
    #a = gam.get_all_metadata("D:\Data_for_up\Week 4")
    
    with open('metadata_all_shearers.pickle','wb') as f:
        pickle.dump(full_metadata_list,f,protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    
    with open('metadata_all_shearers.pickle','rb') as f:
    
        full_metadata_list =  pickle.load(f)[1]
        
if segment_data:
                           
    mocap_feature_list = [apply_to_column(col_average,'jRightKnee_z')]
    
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
    
    Q = sfc.SegmentedFeatureCollection(full_metadata_list,shearer=1, day='tuesday',\
                                   mocap_functions = mocap_feature_list,\
                                   env_functions   = envelope_feature_list,\
                                   emg_functions   = emg_feature_list,\
                                   mocap_env_functions = mocap_env_feature_list,\
                                   emg_shift_functions = emg_shift_feature_list)
    
        
    #Example: get the shearer 1 segmented data
    #Q.features is a list of features split by run
    #dir_path must already exist
    dir_path = 'saved_features_test_extract'
    Q.store_features(pathlib.Path(pathlib.Path.cwd(),dir_path))
    
