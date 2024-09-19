# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:53:52 2020

@author: mrobinson2
"""

import sys
import meta_data as md
from meta_data import MetaData
from pathlib import Path
from config import GBL_DEBUG

def get_all_metadata(directory, *args):
    #Generalised RUN ON ALL
    from add_auto_labels import generate_file_list
    import pathlib
    
    from tqdm import tqdm
    
    result = []
    dir_path_list = []
    
    file_list = generate_file_list(directory)
    
    for i in tqdm(range(len(file_list))):
        tqdm._instances.clear()
        test = []
        for filename in file_list[i].rglob('*_mocap.csv'):
            test.append(filename)
        if test:
            try:
                if GBL_DEBUG == 1: print(str(file_list[i]))
                result.append(md.MetaData(str(file_list[i]),*args))
                dir_path_list.append(str(file_list[i]))
            except IndexError:
                print('{} did not have enough data points'.format(str(file_list[i])))
            except ValueError as e:
                print('{}: {}'.format(str(file_list[i]),e))
    if GBL_DEBUG == 1: print(result[0].run)        
    return (dir_path_list,result)


if __name__ == '__main__':
    #full_metadata_list = get_all_metadata(sys.argv[1], *sys.argv[2:-1])[1]
    
    
    import segmented_feature_collection as sfc
    import pickle
    
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
    #Q = SegmentedDataFile(file='metadata_test.pickle',shearer=7,run=1)
    from get_feature_functions import col_average, col_std,\
                                      apply_to_column, mean_freq_dataframe,\
                                      rmsquare, sample_entropy, ApEn,\
                                      apply_ApEn, apply_to_rel_vec,\
                                      apply_SPARC, apply_to_two_columns,\
                                      two_col_ratio, apply_dimjerk2,\
                                      two_df_mult, two_col_norm,\
                                      apply_twocol_twodf, DRP, apply_DRP,\
                                      shannon_entropy, apply_spec_ind,\
                                      spectral_ind_ratio, col_var, col_autocorr,\
                                      col_skew, col_kurtosis, time_taken_s,\
                                      quantile_01, quantile_025, quantile_05,\
                                      quantile_075, quantile_09, col_max
                           
    mocap_feature_list = [apply_to_column(col_average,'Pelvis_T8_z'),\
                          apply_to_column(col_average,'jRightHip_z'),\
                          apply_to_column(col_average,'jLeftHip_z'),\
                          apply_to_column(col_average,'jRightKnee_z'),\
                          apply_to_column(col_average,'jLeftKnee_z'),\
                          apply_to_column(col_average,'jRightAnkle_z'),\
                          apply_to_column(col_average,'jLeftAnkle_z'),\
                          apply_to_column(rmsquare,'Pelvis_T8_y'),\
                          apply_to_column(rmsquare,'Pelvis_T8_x'),\
                          apply_to_column(col_var,'Pelvis_T8_z'),\
                          apply_to_column(col_var,'jRightHip_z'),\
                          apply_to_column(col_var,'jLeftHip_z'),\
                          apply_to_column(col_var,'jRightKnee_z'),\
                          apply_to_column(col_var,'jLeftKnee_z'),\
                          apply_to_column(col_var,'jRightAnkle_z'),\
                          apply_to_column(col_var,'jLeftAnkle_z'),\
                          apply_to_column(rmsquare,'Pelvis_T8_y'),\
                          apply_to_column(rmsquare,'Pelvis_T8_x'),\
                          apply_to_two_columns(two_col_ratio,'Pelvis_T8_z','jRightHip_z'),\
                          apply_to_two_columns(two_col_ratio,'Pelvis_T8_z','jLeftHip_z'),\
                          apply_SPARC('Pelvis','T8','Pelvis_T8','z'),\
                          apply_SPARC('Pelvis','RightUpperLeg','jRightHip','z'),\
                          apply_SPARC('Pelvis','LeftUpperLeg','jLeftHip','z'),\
                          apply_SPARC('T8','RightUpperArm','T8_RightUpperArm','z'),\
                          apply_SPARC('T8','LeftUpperArm','T8_LeftUpperArm','z'),\
                          apply_DRP(DRP,'Pelvis_T8_z','jRightHip_z',1),\
                          apply_DRP(DRP,'Pelvis_T8_z','jRightHip_z',2),\
                          apply_DRP(DRP,'Pelvis_T8_z','jRightHip_z',3),\
                          apply_to_column(rmsquare,'T8_acc_x'),\
                          apply_to_column(rmsquare,'T8_acc_y'),\
                          apply_to_column(rmsquare,'T8_acc_z'),\
                          apply_to_rel_vec(rmsquare,'Pelvis','T8','angvel','Pelvis_T8','z'),\
                          apply_to_rel_vec(rmsquare,'Pelvis','T8','angvel','Pelvis_T8','y'),\
                          apply_to_rel_vec(rmsquare,'Pelvis','T8','angvel','Pelvis_T8','x'),\
                          apply_to_rel_vec(rmsquare,'Pelvis','T8','angacc','Pelvis_T8','z'),\
                          apply_to_rel_vec(rmsquare,'Pelvis','T8','angacc','Pelvis_T8','y'),\
                          apply_to_rel_vec(rmsquare,'Pelvis','T8','angacc','Pelvis_T8','x'),\
                          apply_dimjerk2('T8','Pelvis','Pelvis_T8','z'),\
                          apply_dimjerk2('T8','Pelvis','Pelvis_T8','y'),\
                          apply_dimjerk2('T8','Pelvis','Pelvis_T8','x'),\
                          apply_dimjerk2('RightUpperLeg','Pelvis','jRightHip','z'),\
                          apply_dimjerk2('RightUpperLeg','Pelvis','jRightHip','y'),\
                          apply_dimjerk2('RightUpperLeg','Pelvis','jRightHip','x'),\
                          apply_dimjerk2('LeftUpperLeg','Pelvis','jLeftHip','z'),\
                          apply_dimjerk2('LeftUpperLeg','Pelvis','jLeftHip','y'),\
                          apply_dimjerk2('LeftUpperLeg','Pelvis','jLeftHip','x'),\
                          apply_to_column(shannon_entropy,'Pelvis_T8_z'),\
                          apply_to_column(shannon_entropy,'jRightHip_z'),\
                          apply_to_column(shannon_entropy,'jLeftHip_z'),\
                          apply_to_column(col_autocorr,'Pelvis_T8_z'),\
                          apply_to_column(col_autocorr,'jRightHip_z'),\
                          apply_to_column(col_autocorr,'jLeftHip_z'),\
                          apply_to_column(col_skew,'Pelvis_T8_z'),\
                          apply_to_column(col_skew,'jRightHip_z'),\
                          apply_to_column(col_skew,'jLeftHip_z'),\
                          apply_to_column(col_kurtosis,'Pelvis_T8_z'),\
                          apply_to_column(col_kurtosis,'jRightHip_z'),\
                          apply_to_column(col_kurtosis,'jLeftHip_z'),\
                          time_taken_s,\
                          apply_ApEn(ApEn,'Pelvis_T8_z','Pelvis_T8_x','Pelvis_T8_y'),\
                          apply_ApEn(ApEn,'jRightHip_z','jRightHip_x','jRightHip_y'),\
                          apply_ApEn(ApEn,'jLeftHip_z','jLeftHip_x','jLeftHip_y'),\
                          apply_to_column(sample_entropy,'Pelvis_T8_z'),\
                          apply_to_column(sample_entropy,'jRightHip_z'),\
                          apply_to_column(sample_entropy,'jLeftHip_z'),\
                          apply_to_two_columns(two_col_norm,'CoG_x','CoG_y'),\
                          apply_to_column(quantile_01,'Pelvis_T8_z'),\
                          apply_to_column(quantile_01,'jRightHip_z'),\
                          apply_to_column(quantile_01,'jLeftHip_z'),\
                          apply_to_column(quantile_025,'Pelvis_T8_z'),\
                          apply_to_column(quantile_025,'jRightHip_z'),\
                          apply_to_column(quantile_025,'jLeftHip_z'),\
                          apply_to_column(quantile_05,'Pelvis_T8_z'),\
                          apply_to_column(quantile_05,'jRightHip_z'),\
                          apply_to_column(quantile_05,'jLeftHip_z'),\
                          apply_to_column(quantile_075,'Pelvis_T8_z'),\
                          apply_to_column(quantile_075,'jRightHip_z'),\
                          apply_to_column(quantile_075,'jLeftHip_z'),\
                          apply_to_column(quantile_09,'Pelvis_T8_z'),\
                          apply_to_column(quantile_09,'jRightHip_z'),\
                          apply_to_column(quantile_09,'jLeftHip_z')]
            

    
    envelope_feature_list = [apply_to_two_columns(two_col_ratio,'L1 Erector Spinae LEFT','Rectus Abdominis RIGHT'),\
                             apply_to_two_columns(two_col_ratio,'L3 Erector Spinae LEFT','Rectus Abdominis RIGHT'),\
                             apply_to_two_columns(two_col_ratio,'L5 Multifidus LEFT','Rectus Abdominis RIGHT'),\
                             apply_to_two_columns(two_col_ratio,'L1 Erector Spinae RIGHT','Rectus Abdominis LEFT'),\
                             apply_to_two_columns(two_col_ratio,'L3 Erector Spinae RIGHT','Rectus Abdominis LEFT'),\
                             apply_to_two_columns(two_col_ratio,'L5 Multifidus RIGHT','Rectus Abdominis LEFT'),\
                             apply_to_two_columns(two_col_ratio,'L1 Erector Spinae LEFT','External Oblique RIGHT'),\
                             apply_to_two_columns(two_col_ratio,'L3 Erector Spinae LEFT','External Oblique RIGHT'),\
                             apply_to_two_columns(two_col_ratio,'L5 Multifidus LEFT','External Oblique RIGHT'),\
                             apply_to_two_columns(two_col_ratio,'L1 Erector Spinae RIGHT','External Oblique LEFT'),\
                             apply_to_two_columns(two_col_ratio,'L3 Erector Spinae RIGHT','External Oblique LEFT'),\
                             apply_to_two_columns(two_col_ratio,'L5 Multifidus RIGHT','External Oblique LEFT'),\
                             apply_to_column(col_var,'L1 Erector Spinae LEFT'),\
                             apply_to_column(col_var,'L3 Erector Spinae LEFT'),\
                             apply_to_column(col_var,'L5 Multifidus LEFT'),\
                             apply_to_column(col_var,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(col_var,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(col_var,'L5 Multifidus RIGHT'),\
                             apply_to_column(col_var,'Rectus Abdominis RIGHT'),\
                             apply_to_column(col_var,'Rectus Abdominis LEFT'),\
                             apply_to_column(col_var,'External Oblique LEFT'),\
                             apply_to_column(col_var,'External Oblique LEFT'),\
                             apply_to_column(col_average,'L1 Erector Spinae LEFT'),\
                             apply_to_column(col_average,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(col_average,'L3 Erector Spinae LEFT'),\
                             apply_to_column(col_average,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(col_average,'L5 Multifidus LEFT'),\
                             apply_to_column(col_average,'L5 Multifidus RIGHT'),\
                             apply_to_column(col_average,'Rectus Abdominis RIGHT'),\
                             apply_to_column(col_average,'Rectus Abdominis LEFT'),\
                             apply_to_column(col_average,'External Oblique RIGHT'),\
                             apply_to_column(col_average,'External Oblique LEFT'),\
                             apply_to_column(col_kurtosis,'L1 Erector Spinae LEFT'),\
                             apply_to_column(col_kurtosis,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(col_kurtosis,'L3 Erector Spinae LEFT'),\
                             apply_to_column(col_kurtosis,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(col_kurtosis,'L5 Multifidus LEFT'),\
                             apply_to_column(col_kurtosis,'L5 Multifidus RIGHT'),\
                             apply_to_column(col_kurtosis,'Rectus Abdominis RIGHT'),\
                             apply_to_column(col_kurtosis,'Rectus Abdominis LEFT'),\
                             apply_to_column(col_kurtosis,'External Oblique RIGHT'),\
                             apply_to_column(col_kurtosis,'External Oblique LEFT'),\
                             apply_to_column(col_skew,'L1 Erector Spinae LEFT'),\
                             apply_to_column(col_skew,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(col_skew,'L3 Erector Spinae LEFT'),\
                             apply_to_column(col_skew,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(col_skew,'L5 Multifidus LEFT'),\
                             apply_to_column(col_skew,'L5 Multifidus RIGHT'),\
                             apply_to_column(col_skew,'Rectus Abdominis RIGHT'),\
                             apply_to_column(col_skew,'Rectus Abdominis LEFT'),\
                             apply_to_column(col_skew,'External Oblique RIGHT'),\
                             apply_to_column(col_skew,'External Oblique LEFT'),\
                             apply_to_column(shannon_entropy,'L1 Erector Spinae LEFT'),\
                             apply_to_column(shannon_entropy,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(shannon_entropy,'L3 Erector Spinae LEFT'),\
                             apply_to_column(shannon_entropy,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(shannon_entropy,'L5 Multifidus LEFT'),\
                             apply_to_column(shannon_entropy,'L5 Multifidus RIGHT'),\
                             apply_to_column(shannon_entropy,'Rectus Abdominis RIGHT'),\
                             apply_to_column(shannon_entropy,'Rectus Abdominis LEFT'),\
                             apply_to_column(shannon_entropy,'External Oblique RIGHT'),\
                             apply_to_column(shannon_entropy,'External Oblique LEFT'),\
                             apply_to_column(col_autocorr,'L1 Erector Spinae LEFT'),\
                             apply_to_column(col_autocorr,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(col_autocorr,'L3 Erector Spinae LEFT'),\
                             apply_to_column(col_autocorr,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(col_autocorr,'L5 Multifidus LEFT'),\
                             apply_to_column(col_autocorr,'L5 Multifidus RIGHT'),\
                             apply_to_column(col_autocorr,'Rectus Abdominis RIGHT'),\
                             apply_to_column(col_autocorr,'Rectus Abdominis LEFT'),\
                             apply_to_column(col_autocorr,'External Oblique RIGHT'),\
                             apply_to_column(col_autocorr,'External Oblique LEFT'),\
                             apply_to_column(col_autocorr,'L1 Erector Spinae LEFT'),\
                             apply_to_column(quantile_01,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_01,'L3 Erector Spinae LEFT'),\
                             apply_to_column(quantile_01,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_01,'L5 Multifidus LEFT'),\
                             apply_to_column(quantile_01,'L5 Multifidus RIGHT'),\
                             apply_to_column(quantile_01,'Rectus Abdominis RIGHT'),\
                             apply_to_column(quantile_01,'Rectus Abdominis LEFT'),\
                             apply_to_column(quantile_01,'External Oblique RIGHT'),\
                             apply_to_column(quantile_01,'External Oblique LEFT'),\
                             apply_to_column(quantile_025,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_025,'L3 Erector Spinae LEFT'),\
                             apply_to_column(quantile_025,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_025,'L5 Multifidus LEFT'),\
                             apply_to_column(quantile_025,'L5 Multifidus RIGHT'),\
                             apply_to_column(quantile_025,'Rectus Abdominis RIGHT'),\
                             apply_to_column(quantile_025,'Rectus Abdominis LEFT'),\
                             apply_to_column(quantile_025,'External Oblique RIGHT'),\
                             apply_to_column(quantile_025,'External Oblique LEFT'),\
                             apply_to_column(quantile_05,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_05,'L3 Erector Spinae LEFT'),\
                             apply_to_column(quantile_05,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_05,'L5 Multifidus LEFT'),\
                             apply_to_column(quantile_05,'L5 Multifidus RIGHT'),\
                             apply_to_column(quantile_05,'Rectus Abdominis RIGHT'),\
                             apply_to_column(quantile_05,'Rectus Abdominis LEFT'),\
                             apply_to_column(quantile_05,'External Oblique RIGHT'),\
                             apply_to_column(quantile_05,'External Oblique LEFT'),\
                             apply_to_column(quantile_075,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_075,'L3 Erector Spinae LEFT'),\
                             apply_to_column(quantile_075,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_075,'L5 Multifidus LEFT'),\
                             apply_to_column(quantile_075,'L5 Multifidus RIGHT'),\
                             apply_to_column(quantile_075,'Rectus Abdominis RIGHT'),\
                             apply_to_column(quantile_075,'Rectus Abdominis LEFT'),\
                             apply_to_column(quantile_075,'External Oblique RIGHT'),\
                             apply_to_column(quantile_075,'External Oblique LEFT'),\
                             apply_to_column(quantile_09,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_09,'L3 Erector Spinae LEFT'),\
                             apply_to_column(quantile_09,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(quantile_09,'L5 Multifidus LEFT'),\
                             apply_to_column(quantile_09,'L5 Multifidus RIGHT'),\
                             apply_to_column(quantile_09,'Rectus Abdominis RIGHT'),\
                             apply_to_column(quantile_09,'Rectus Abdominis LEFT'),\
                             apply_to_column(quantile_09,'External Oblique RIGHT'),\
                             apply_to_column(quantile_09,'External Oblique LEFT'),\
                             apply_to_column(col_max,'L1 Erector Spinae LEFT'),\
                             apply_to_column(col_max,'L3 Erector Spinae LEFT'),\
                             apply_to_column(col_max,'L5 Multifidus LEFT'),\
                             apply_to_column(col_max,'L1 Erector Spinae RIGHT'),\
                             apply_to_column(col_max,'L3 Erector Spinae RIGHT'),\
                             apply_to_column(col_max,'L5 Multifidus RIGHT'),\
                             apply_to_column(col_max,'Rectus Abdominis RIGHT'),\
                             apply_to_column(col_max,'Rectus Abdominis LEFT'),\
                             apply_to_column(col_max,'External Oblique LEFT'),\
                             apply_to_column(col_max,'External Oblique LEFT')]
#                             
#    
    emg_feature_list = [apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae LEFT',-1,5),\
                        apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae LEFT',-1,5),\
                        apply_spec_ind(spectral_ind_ratio,'L5 Multifidus LEFT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'L5 Multifidus LEFT',-1,5),\
                        apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae RIGHT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'L1 Erector Spinae RIGHT',-1,5),\
                        apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae RIGHT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'L3 Erector Spinae RIGHT',-1,5),\
                        apply_spec_ind(spectral_ind_ratio,'L5 Multifidus RIGHT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'L5 Multifidus RIGHT',-1,5),\
                        apply_spec_ind(spectral_ind_ratio,'Rectus Abdominis RIGHT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'Rectus Abdominis RIGHT',-1,5),\
                        apply_spec_ind(spectral_ind_ratio,'Rectus Abdominis LEFT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'Rectus Abdominis LEFT',-1,5),\
                        apply_spec_ind(spectral_ind_ratio,'External Oblique RIGHT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'External Oblique RIGHT',-1,5),\
                        apply_spec_ind(spectral_ind_ratio,'External Oblique LEFT',1,0),\
                        apply_spec_ind(spectral_ind_ratio,'External Oblique LEFT',-1,5),\
                        apply_to_column(shannon_entropy,'L1 Erector Spinae LEFT'),\
                        apply_to_column(shannon_entropy,'L3 Erector Spinae LEFT'),\
                        apply_to_column(shannon_entropy,'L5 Multifidus LEFT'),\
                        apply_to_column(shannon_entropy,'L1 Erector Spinae RIGHT'),\
                        apply_to_column(shannon_entropy,'L3 Erector Spinae RIGHT'),\
                        apply_to_column(shannon_entropy,'L5 Multifidus RIGHT'),\
                        apply_to_column(shannon_entropy,'Rectus Abdominis RIGHT'),\
                        apply_to_column(shannon_entropy,'External Oblique RIGHT'),\
                        apply_to_column(shannon_entropy,'Rectus Abdominis LEFT'),\
                        apply_to_column(shannon_entropy,'External Oblique LEFT'),\
                        apply_to_column(col_autocorr,'L1 Erector Spinae LEFT'),\
                        apply_to_column(col_autocorr,'L3 Erector Spinae LEFT'),\
                        apply_to_column(col_autocorr,'L5 Multifidus LEFT'),\
                        apply_to_column(col_autocorr,'L1 Erector Spinae RIGHT'),\
                        apply_to_column(col_autocorr,'L3 Erector Spinae RIGHT'),\
                        apply_to_column(col_autocorr,'L5 Multifidus RIGHT'),\
                        apply_to_column(col_autocorr,'Rectus Abdominis RIGHT'),\
                        apply_to_column(col_autocorr,'External Oblique RIGHT'),\
                        apply_to_column(col_autocorr,'Rectus Abdominis LEFT'),\
                        apply_to_column(col_autocorr,'External Oblique LEFT')]
                        
                        
    
    mocap_env_feature_list = [apply_twocol_twodf(two_df_mult,'Pelvis_T8_z','L3 Erector Spinae LEFT'),\
                              apply_twocol_twodf(two_df_mult,'Pelvis_T8_z','L3 Erector Spinae RIGHT'),\
                              apply_twocol_twodf(two_df_mult,'Pelvis_T8_z','L1 Erector Spinae LEFT'),\
                              apply_twocol_twodf(two_df_mult,'Pelvis_T8_z','L1 Erector Spinae RIGHT'),\
                              apply_twocol_twodf(two_df_mult,'Pelvis_T8_z','L5 Multifidus LEFT'),\
                              apply_twocol_twodf(two_df_mult,'Pelvis_T8_z','L5 Multifidus RIGHT')]
    
    file = 'metadata_cloud_pickle.pickle'
    with open(file,'rb') as file:
        full_metadata_list = pickle.load(file)[1]
    
    
    Q = sfc.SegmentedFeatureCollection(full_metadata_list,shearer=7, day='monday',\
                                   mocap_functions = mocap_feature_list,\
                                   env_functions   = envelope_feature_list,\
                                   emg_functions   = emg_feature_list,\
                                   mocap_env_functions = mocap_env_feature_list,\
                                   emg_shift_functions = emg_feature_list,\
                                   env_shift_functions = envelope_feature_list,\
                                   mocap_env_shift_function = mocap_env_feature_list)
    
    
    Q.store_features('/data/cephfs/punim1234/Data/saved_features')
    #Q.plot_features()