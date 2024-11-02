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
import check_file_structure_of_data_dir as cfsdd
import pathlib
import plot_article_figures as paf
from get_feature_functions import col_average, col_std,\
                                      apply_to_column, mean_freq_dataframe,\
                                      rmsquare, sample_entropy, ApEn,\
                                      apply_ApEn, apply_to_rel_vec, apply_to_two_columns,\
                                      two_col_ratio, apply_dimjerk2,\
                                      two_df_mult, two_col_norm,\
                                      apply_twocol_twodf, DRP, apply_DRP,\
                                      shannon_entropy, apply_spec_ind,\
                                      spectral_ind_ratio,get_datetime, quantile_05
from config import GBL_DEBUG,\
                   GBL_PATH_TO_DATA,\
                   GBL_EXTRACTED_FEATURES_SAVE_PATH,\
                   GBL_PATH_TO_METADATA_ORIG,\
                   GBL_SAVEPATH_FOR_GENERATED_METADATA

def get_the_feature_plots(extract_metadata = False, 
                          segment_and_extract_features = False,
                          data_path=GBL_PATH_TO_DATA):
    
    
    
    
    if segment_and_extract_features:
        cfsdd.find_files(GBL_PATH_TO_DATA, 'week')
    
    if extract_metadata:
    
        full_metadata_list = gam.get_all_metadata(data_path)
        
        with open(GBL_SAVEPATH_FOR_GENERATED_METADATA,'wb') as f:
            pickle.dump(full_metadata_list,f,protocol=pickle.HIGHEST_PROTOCOL)
            
    else:
        
        with open(GBL_PATH_TO_METADATA_ORIG,'rb') as f:
        
            full_metadata_list =  pickle.load(f)[1]
            
    if segment_and_extract_features:
                               
        mocap_feature_list = [apply_to_column(quantile_05,'jRightKnee_z'),\
                              apply_to_column(quantile_05, 'jRightHip_z'),\
                                  apply_to_column(quantile_05, 'Pelvis_T8_z')]
        
        envelope_feature_list = [
        #[apply_to_two_columns(two_col_ratio,'L1 Erector Spinae LEFT','Rectus Abdominis RIGHT')]
                                  apply_to_column(quantile_05,'L1 Erector Spinae LEFT'),\
                                 apply_to_column(quantile_05,'L1 Erector Spinae RIGHT'),\
                                 apply_to_column(quantile_05,'L3 Erector Spinae LEFT'),\
                                 apply_to_column(quantile_05,'L3 Erector Spinae RIGHT'),\
                                 apply_to_column(quantile_05,'L5 Multifidus LEFT'),\
                                 apply_to_column(quantile_05,'L5 Multifidus RIGHT')]
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
        emg_feature_list = [get_datetime,
                            apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',1,0),\
                            apply_to_column(shannon_entropy,'L3 Erector Spinae RIGHT'),\
                            #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,2),\
                            #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,3),\
                            #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,4),\
                            #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,5),\
                            #apply_to_column(mean_freq_dataframe,'L1 Erector Spinae RIGHT'),\
                            apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',1,0),\
                            apply_to_column(shannon_entropy,'L3 Erector Spinae LEFT'),\
                            #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,2),\
                            #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,3),\
                            #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,4),\
                            #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,5)]
                            apply_to_column(mean_freq_dataframe,'L5 Multifidus LEFT'),\
                                apply_to_column(shannon_entropy,'L5 Multifidus LEFT'),\
                                    apply_to_column(shannon_entropy,'L5 Multifidus RIGHT'),\
                            apply_to_column(mean_freq_dataframe,'L5 Multifidus RIGHT')]
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
        
        emg_shift_feature_list = [apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',1,0),
                            #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,2),\
                            #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,3),\
                            #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,4),\
                            #apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,5),\
                            #apply_to_column(mean_freq_dataframe,'L1 Erector Spinae RIGHT'),\
                            apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',1,0)]
                            #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,2)]
                            #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,3),\
                            #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,4),\
                            #apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,5)]
        
        
       
        #Shearer days    
        #shearer_days = [(8,'tuesday'),(1,'tuesday'),(1,'wednesday'),\
        #                (1,'thursday'), (7,'monday'),(9,'wednesday')]
            
        shearer_days = [(1,'wednesday'),\
                        (7,'monday')]
        
        for shearer_day in shearer_days:
            Q = sfc.SegmentedFeatureCollection(full_metadata_list,shearer=shearer_day[0], day=shearer_day[1],\
                                           mocap_functions = mocap_feature_list,\
                                           env_functions   = envelope_feature_list,\
                                           emg_functions   = emg_feature_list,\
                                           mocap_env_functions = mocap_env_feature_list,\
                                           emg_shift_functions = emg_shift_feature_list)
            
                
            #Example: get the shearer 1 segmented data
            #Q.features is a list of features split by run
            
            #dir_path must already exist
            dir_path = GBL_EXTRACTED_FEATURES_SAVE_PATH
            Q.store_features(pathlib.Path(pathlib.Path.cwd(),dir_path))
        
        paf.plot_figures(dir_path)
        
    else:
        paf.plot_figures()
        

