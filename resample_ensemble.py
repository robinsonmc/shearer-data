# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:41:57 2019

Segment (resampled) the data from one run based on the shearing cycle

does it make sense to resample here?
    - resampling was for the ensemble average
    
- Now we just want the raw segmented data
- Rejecting the incorrect sheep as much as possible


@author: mrobinson2

['time_ms', 'jL5S1_x', 'jL5S1_y', 'jL5S1_z', 'jL4L3_x', 'jL4L3_y',
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
"""
import numpy as np
import pandas as pd
from pathlib import Path
import models_movie as mm
import matplotlib.pyplot as plt
import mean_freq_calc as mfc
import STFT as stft

class Summary:
    
    def __init__(self, dir_path, run, extended):
        '''
        Create summary for a run (separate the data for every sheep into a
        list to perform analysis per sheep)
        
        - Per sheep
        - Exponential regression?
        - Linear & segmented linear regression?
        - Average flexion angle per sheep?
        - Regression?
        
        dir_path is the days directory and the run should be an int (1-4)
        
        '''
        self.dir_path       = dir_path
        self.run_num        = run
                            
        if extended: 
            '''Only use the extended version'''                   
            self.xsens_all_ext, self.envelope_all_ext, self.emg_all_ext = \
                self.extract_all_extend(self.dir_path, self.run_num, 'shearing')
        else:
            self.xsens_all, self.envelope_all, self.emg_all, \
            self.pre_xsens_all, self.pre_envelope_all, self.pre_emg_all = \
                self.extract_all(self.dir_path, self.run_num, 'shearing')

        
    def per_sheep(self):
        import tqdm
        '''
        This needs to plot an average of the frequency for each sheep in the
        run, this should be for each of the 6 muscles
        
        Also plot a second figure which represents the catch drag phase of the
        task - each sheep in the run
        
        This should be possible to do also for each run of the day
        
        '''
        list_all_data = self.emg_all
        
        time_list      = []
        mean_freq_list = []
        for element in tqdm.tqdm(list_all_data):
            mean_freq, RMS = Summary.mean_freq_dataframe(element)
            mean_freq_av = mean_freq.mean()
            final_time = element.index[-1]
            mean_freq_list.append(mean_freq_av)
            time_list.append(final_time)
        
        return time_list, mean_freq_list
    
    def per_sheep_env(self):
        import tqdm
        '''
        This needs to plot an average of the frequency for each sheep in the
        run, this should be for each of the 6 muscles
        
        Also plot a second figure which represents the catch drag phase of the
        task - each sheep in the run
        
        This should be possible to do also for each run of the day
        
        '''
        list_all_data = self.envelope_all_ext
        
        time_list      = []
        env_list = []
        for element in tqdm.tqdm(list_all_data):
            env_av = element.mean()
            final_time = element.index[-1]
            env_list.append(env_av)
            time_list.append(final_time)
        
        return time_list, env_list
    
  
    @staticmethod
    def nice_print(dataframe):
        with pd.option_context('display.max_rows', None,\
                               'display.max_columns', None,\
                               'max_colwidth', 1000):  # more options can be specified also
            print(dataframe)
    @staticmethod
    def plot_muscle(regression, data, muscle):
        with pd.option_context('display.max_rows', None,\
                               'display.max_columns', None,\
                               'max_colwidth', 1000):  # more options can be specified also
            print(regression['p'][muscle])
            
        plt.figure()
        plt.plot(data['time'],data[muscle],alpha=0.7)
        plt.plot(data['time'], Summary.piecewise_linear(np.array(\
                                 data['time']),*regression['p'][muscle]))
        
        
    @staticmethod
    def piecewise_linear(x, x0, y0, k1, k2):
        #Add constraint to x0
        return np.piecewise(x, [x < x0, x>= x0],\
                    [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    @staticmethod
    def get_run_freq_data(dir_path,run):
        #Load all of the data in run
        #Need list of dataframes
        dir_path = Path(dir_path)
        file_list = sorted(dir_path.glob('*run'+str(run)+'*'))
        
        data_list    = [mm.DataFile(x) for x in file_list]
        mmf_tuples   = [Summary.mean_freq(x) for x in data_list]
        unzip        = [[*x] for x in zip(*mmf_tuples)]
        mmf_list     = unzip[0]
        mmf_RMS_list = unzip[1]
        
        del data_list
        del mmf_tuples
        del unzip
        
        if len(mmf_list) > 1:
            for i in range(len(mmf_list)-1):
                orig  = mmf_list[i]
                addon = mmf_list[i+1]
                
                t_end = orig['time'].iloc[-1]
                t_diff = orig['time'].iloc[-1] - orig['time'].iloc[-2]
                t_start = t_end + t_diff
                
                addon['time'] = addon['time'] + t_start
                
                mmf_list[i+1]['time'] = addon['time']
                mmf_list[i+1].index   += mmf_list[i].index[-1] + 1
                
        return pd.concat(mmf_list), mmf_list

    #TODO: possibly move this elsewhere
    @staticmethod
    def mean_freq_dataframe(df):
        #Reduce the column names to the muscle names to avoid the different
        #sensor numbers on certain muscles (this should potentially be done
        #in the datafile class so it is consistent)
        cols = []
        for i in range(len(df.columns)):
            s = str(df.columns[i])
            result = s.split(':')[0]
            cols.append(result)
        df.columns = cols
        
        #Every muscle is labelled as left or right
        #Reject non-muscle columns (e.g. time, labels)
        muscles = [x for x in cols if ('LEFT' in x) | ('RIGHT' in x)]
        
        df_mean_freq = pd.DataFrame()
        df_RMS = pd.DataFrame()
        #Get the data
        for muscle in muscles:
            #Add the mean frequency calc muscle by muscle
            f, t, spec = stft.shearer_spectogram(df, muscle)
            temp_df_mean_freq, temp_df_RMS =\
                                        mfc.muscle_mean_freq(f, t, spec)
            df_mean_freq[muscle] = temp_df_mean_freq['mean_f']
            df_mean_freq['time'] = temp_df_mean_freq['time']
            df_RMS[muscle] = temp_df_RMS['RMS']
            df_RMS['time'] = temp_df_RMS['time']
            
        return (df_mean_freq[:-1],df_RMS[:-1])
    
    
        #Others (store or calc?)
        # - Sum squares emg
        # - Average emg activation
        # - Average movement (whole df)
        # - Variance movement (whole dF)
        
        #Get 10 slices
        #Resample the 10 slices
        #Summarise
        
    def extract_all(self, dir_path, run, shearing):
        '''
        Return a list of dataframes with xsens information from all sheep in 
        run
        
        Input directory to the day shearing
        the run (int 1-4) that they should be taken from
        '''
        import pathlib
        from scipy.stats import zscore
        
        print('extracting all...')
        
        if shearing == 'shearing':
            
            # Load the correct file
            dir_path = Path(dir_path)
            file_list = sorted(dir_path.glob('*run'+str(run)+'*'))
            
            if not file_list: 
                raise ValueError('No files were found in the given directory')
            
            xsens_extracted = []
            delsys_extracted = []
            raw_EMG_extracted = []
            xsens_extracted_pre = []
            delsys_extracted_pre = []
            raw_EMG_extracted_pre = []
            for file in file_list:
                xsens_extracted_temp, delsys_extracted_temp, raw_EMG_extracted_temp,\
                    xsens_extracted_pre_temp, delsys_extracted_pre_temp, raw_EMG_extracted_pre_temp,\
                       = self.extract_shearing_part(mm.DataFile(file))
                
                xsens_extracted.extend(xsens_extracted_temp)
                delsys_extracted.extend(delsys_extracted_temp)
                raw_EMG_extracted.extend(raw_EMG_extracted_temp)
                xsens_extracted_pre.extend(xsens_extracted_pre_temp)
                delsys_extracted_pre.extend(delsys_extracted_pre_temp)
                raw_EMG_extracted_pre.extend(raw_EMG_extracted_pre_temp)
            
            
            #Reject short sheeps based on conservative actual lengths first
            #Thsi will fix and help with the next one
            #Then finalise a format for plotting
            assert(len(xsens_extracted)  == len(delsys_extracted))
            assert(len(xsens_extracted)  == len(raw_EMG_extracted))
            assert(len(delsys_extracted) == len(raw_EMG_extracted))
            
            #Reject sheep that are too short (equivalent about 45 seconds or so)
            xsens_len  = [len(x) for x in xsens_extracted]
            xsens_mask  = np.array([x > 2700 for x in xsens_len])
            
            # Need to reject sheeps that are too short
            # Do this based on zscore of the lengths of the dataframes
            xsens_zlen  = zscore([len(x) for x in xsens_extracted])
            
            # Reject sheep that are more than 1.2std below the mean (which is
            # shifted down because of the errors)
            xsens_zmask = np.array([x > -1.2 for x in xsens_zlen])
            
            final_mask = list(xsens_mask & xsens_zmask)
            
            
            #Apply the mask
            xsens_shortened = [i for idx,i in enumerate(xsens_extracted) if \
                                   final_mask[idx] == True]
            delsys_shortened = [i for idx,i in enumerate(delsys_extracted) if \
                                   final_mask[idx] == True]
            EMG_shortened = [i for idx,i in enumerate(raw_EMG_extracted) if \
                                   final_mask[idx] == True]
            
            
            xsens_pre_short = [i for idx,i in enumerate(xsens_extracted_pre) if \
                                   final_mask[idx] == True]
            
            delsys_pre_short = [i for idx,i in enumerate(delsys_extracted_pre) if \
                                   final_mask[idx] == True]
            
            raw_EMG_pre_short = [i for idx,i in enumerate(raw_EMG_extracted_pre) if \
                                   final_mask[idx] == True]
            
            
            #Check for empty dataframes here - and short ones?? then delete
            #the correct index from all three
            print('entering for loop...')
            for i in reversed(range(len(EMG_shortened))):
                print('if statement to be evaluated...')
                if (len(EMG_shortened[i]) < 15) or \
                   (len(xsens_shortened[i]) < 15) or \
                   (len(delsys_shortened[i]) < 15):
                       print('If statement true... trying to delete...')
                       del EMG_shortened[i]
                       del xsens_shortened[i]
                       del delsys_shortened[i]
                       del xsens_pre_short[i]
                       del delsys_pre_short[i]
                       del raw_EMG_pre_short[i]
            
            assert(len(xsens_shortened)  == len(delsys_shortened))
            assert(len(xsens_shortened)  == len(EMG_shortened))
            assert(len(delsys_shortened) == len(EMG_shortened))
            
            
            return xsens_shortened, delsys_shortened, EMG_shortened,\
                    xsens_pre_short, delsys_pre_short, raw_EMG_pre_short
        else:
            raise Exception('shearing needs to take value "shearing" or "catch_drag (not implemented)". The value of shearing was {}'.format(shearing))

    
    def extract_all_extend(self, dir_path, run_num, shearing):
        '''
        Return a list of dataframes with xsens information from all sheep in 
        run
        
        Input directory to the day shearing
        the run (int 1-4) that they should be taken from
        '''
        from scipy.stats import zscore
        
        if shearing == 'shearing':
            
            # Load the correct file
            dir_path = Path(dir_path)
            file_list = sorted(dir_path.glob('*run'+str(run_num)+'*'))
            
            if not file_list: 
                raise ValueError('No files were found in the given directory')
            
            xsens_extracted = []
            delsys_extracted = []
            raw_EMG_extracted = []
            for file in file_list:
                xsens_extracted_temp, delsys_extracted_temp, raw_EMG_extracted_temp = \
                        self.extract_shearing_part_extend(mm.DataFile(file))
                
                xsens_extracted.extend(xsens_extracted_temp)
                delsys_extracted.extend(delsys_extracted_temp)
                raw_EMG_extracted.extend(raw_EMG_extracted_temp)
            
            #Reject short sheeps based on conservative actual lengths first
            #Thsi will fix and help with the next one
            #Then finalise a format for plotting
            assert(len(xsens_extracted)  == len(delsys_extracted))
            assert(len(xsens_extracted)  == len(raw_EMG_extracted))
            assert(len(delsys_extracted) == len(raw_EMG_extracted))
            
            #Reject sheep that are too short (equivalent about 45 seconds or so)
            xsens_len  = [len(x) for x in xsens_extracted]
            xsens_mask  = np.array([x > 2700 for x in xsens_len])
            
            # Need to reject sheeps that are too short
            # Do this based on zscore of the lengths of the dataframes
            xsens_zlen  = zscore([len(x) for x in xsens_extracted])
            
            # Reject sheep that are more than 1.2std below the mean (which is
            # shifted down because of the errors)
            xsens_zmask = np.array([x > -1.2 for x in xsens_zlen])
            
            final_mask = list(xsens_mask & xsens_zmask)
            
            
            #Apply the mask
            xsens_shortened = [i for idx,i in enumerate(xsens_extracted) if \
                                   final_mask[idx] == True]
            delsys_shortened = [i for idx,i in enumerate(delsys_extracted) if \
                                   final_mask[idx] == True]
            EMG_shortened = [i for idx,i in enumerate(raw_EMG_extracted) if \
                                   final_mask[idx] == True]
            
            return xsens_shortened, delsys_shortened, EMG_shortened
        else:
            raise Exception('shearing needs to take value "shearing" or "catch_drag (not implemented)". The value of shearing was {}'.format(shearing))
   
    def extract_shearing_part(self, DataFile):
        from copy import copy
        
        delsys_data = DataFile.delsys_data
        xsens_data  = DataFile.xsens_data 
        emg_env     = DataFile.env_df
        
        #Make list of dataframes separated in time
        #1 for Shearing
        temp_df = xsens_data.loc[xsens_data['HMM_labels'] == 1]
        time_shift            = temp_df['time_ms'].shift(-1)
        temp_df.insert(len(temp_df.columns),'time_shift',time_shift,False)
        temp_df.insert(len(temp_df.columns),'eval',temp_df['time_shift'] - temp_df['time_ms'],False)
        index_list            = temp_df.loc[temp_df['eval'] > 1000].index
        
        #Get rid of na values here
        #temp_df.fillna(method='bfill',inplace=True)
        
        #Make list of dataframes separated in time
        temp_df_emg = emg_env.loc[emg_env['HMM_labels'] == 1]
        time_shift                = temp_df_emg['time'].shift(-1)
        temp_df_emg.insert(len(temp_df_emg.columns),'time_shift',time_shift,False)
        temp_df_emg.insert(len(temp_df_emg.columns),'eval',temp_df_emg['time_shift'] - temp_df_emg['time'],False)
        index_list_emg            = temp_df_emg.loc[temp_df_emg['eval'] > 1].index
        
        #Get rid of na values here
       # temp_df_emg.fillna(method='bfill',inplace=True)
        
        #Make list of dataframes separated in time
        #Set allow dupes to true?????
        temp_df_raw_emg = delsys_data.loc[delsys_data['HMM_labels'] == 1]
        time_shift                    = temp_df_raw_emg['time'].shift(-1)
        
        temp_df_raw_emg = temp_df_raw_emg[~temp_df_raw_emg.index.duplicated(keep='first')]
        print(temp_df_raw_emg.columns)
        print(temp_df_raw_emg)
        print(len(temp_df_raw_emg))
        print(temp_df_raw_emg['time'].values)
        print(len(temp_df_raw_emg['time'].values))
        print(time_shift.fillna(method='ffill'))
        print(len(time_shift.fillna(method='ffill').values))
        print(time_shift.fillna(method='ffill').values-temp_df_raw_emg['time'].values)
        #temp_df_raw_emg.drop(columns='time_shift',inplace=True)
        
        temp_df_raw_emg.insert(len(temp_df_raw_emg.columns),'time_shift',time_shift.fillna(method='ffill').values,False)
        temp_df_raw_emg.insert(len(temp_df_raw_emg.columns),'eval',temp_df_raw_emg['time_shift'].values - temp_df_raw_emg['time'].values,True)
        #temp_df_raw_emg['time_shift'] = time_shift.fillna(method='ffill').values
        #temp_df_raw_emg['eval'] = temp_df_raw_emg['time_shift'].values - temp_df_emg['time'].values
        
        index_list_raw_emg            = temp_df_raw_emg.loc[temp_df_raw_emg['eval'] > 1].index
        
        #Get rid of na values here
        #temp_df_raw_emg.fillna(method='bfill',inplace=True)
        
        xsens_df_list  = []
        delsys_df_list = []
        delsys_emg_df_list = []
        for i in range(len(index_list)-1):
            if i == 0:
                xsens_df_list.append(temp_df.loc[temp_df.index\
                                        <= index_list[i]])
                delsys_df_list.append(temp_df_emg.loc[temp_df_emg.index\
                                        <= index_list[i]])
                delsys_emg_df_list.append(temp_df_raw_emg.loc[temp_df_raw_emg.index\
                                        <= index_list[i]])
            else:
                xsens_df_list.append(temp_df.loc[(temp_df.index > \
                                        index_list[i]) & (temp_df.index <= \
                                            index_list[i+1])])
                delsys_df_list.append(temp_df_emg.loc[(temp_df_emg.index > \
                                        index_list[i]) & (temp_df_emg.index <= \
                                            index_list[i+1])])
                delsys_emg_df_list.append(temp_df_raw_emg.loc[(temp_df_raw_emg.index > \
                                        index_list[i]) & (temp_df_raw_emg.index <= \
                                            index_list[i+1])])
                
        del temp_df_emg, temp_df, temp_df_raw_emg
        
        
        #We now have the splits, reject ones that are too short
        xsens_full_length = []
        for item in xsens_df_list:
            xsens_full_length.append(item)

        del xsens_df_list

        delsys_full_length = []
        for item in delsys_df_list:
            delsys_full_length.append(item)  
                
        del delsys_df_list
        
        delsys_raw_full_length = []
        for item in delsys_emg_df_list:
            delsys_raw_full_length.append(item)
            
        del delsys_emg_df_list 
            
        #Shift the HMM_label up (8seconds not 5)
        delsys_data['HMM_5s'] = delsys_data['HMM_labels'].shift(-17186).fillna(method='ffill')
        xsens_data['HMM_5s'] = xsens_data['HMM_labels'].shift(-480).fillna(method='ffill')
        emg_env['HMM_5s'] = emg_env['HMM_labels'].shift(-80).fillna(method='ffill')
        
        #HMM_5s must = 1, HMM must = 0
        delsys_data['5s_pre'] = delsys_data.apply(lambda row: (np.logical_xor(row['HMM_labels'],row['HMM_5s'])) & (row['HMM_5s']==1),axis=1)
        xsens_data['5s_pre'] = xsens_data.apply(lambda row: (np.logical_xor(row['HMM_labels'],row['HMM_5s'])) & (row['HMM_5s']==1),axis=1)
        emg_env['5s_pre'] = emg_env.apply(lambda row: (np.logical_xor(row['HMM_labels'],row['HMM_5s'])) & (row['HMM_5s']==1),axis=1)

        
        temp_df = xsens_data.loc[xsens_data['5s_pre'] == 1]
        time_shift            = temp_df['time_ms'].shift(-1)
        temp_df.insert(len(temp_df.columns),'time_shift',time_shift,False)
        temp_df.insert(len(temp_df.columns),'eval',temp_df['time_shift'] - temp_df['time_ms'],False)
        index_list            = temp_df.loc[temp_df['eval'] > 1000].index
        
        #Get rid of na values here
        #temp_df.fillna(method='bfill',inplace=True)
        
        #Make list of dataframes separated in time
        temp_df_emg = emg_env.loc[emg_env['5s_pre'] == 1]
        time_shift                = temp_df_emg['time'].shift(-1)
        temp_df_emg.insert(len(temp_df_emg.columns),'time_shift',time_shift,False)
        temp_df_emg.insert(len(temp_df_emg.columns),'eval',temp_df_emg['time_shift'] - temp_df_emg['time'],False)
        index_list_emg            = temp_df_emg.loc[temp_df_emg['eval'] > 1].index
        
        #Get rid of na values here
        #temp_df_emg.fillna(method='bfill',inplace=True)
        
        #Make list of dataframes separated in time
        temp_df_raw_emg = delsys_data.loc[delsys_data['5s_pre'] == 1]
        time_shift                    = temp_df_raw_emg['time'].shift(-1)
        temp_df_raw_emg.insert(len(temp_df_raw_emg.columns),'time_shift',time_shift,False)
        temp_df_raw_emg.insert(len(temp_df_raw_emg.columns),'eval',temp_df_raw_emg['time_shift'] - temp_df_raw_emg['time'],False)
        index_list_raw_emg            = temp_df_raw_emg.loc[temp_df_raw_emg['eval'] > 1].index
        
        #Get rid of na values here
        #temp_df_raw_emg.fillna(method='bfill',inplace=True)
        
        xsens_df_list  = []
        delsys_df_list = []
        delsys_emg_df_list = []
        for i in range(len(index_list)-1):
            if i == 0:
                xsens_df_list.append(temp_df.loc[temp_df.index\
                                        <= index_list[i]])
                delsys_df_list.append(temp_df_emg.loc[temp_df_emg.index\
                                        <= index_list[i]])
                delsys_emg_df_list.append(temp_df_raw_emg.loc[temp_df_raw_emg.index\
                                        <= index_list[i]])
            else:
                xsens_df_list.append(temp_df.loc[(temp_df.index > \
                                        index_list[i]) & (temp_df.index <= \
                                            index_list[i+1])])
                delsys_df_list.append(temp_df_emg.loc[(temp_df_emg.index > \
                                        index_list[i]) & (temp_df_emg.index <= \
                                            index_list[i+1])])
                delsys_emg_df_list.append(temp_df_raw_emg.loc[(temp_df_raw_emg.index > \
                                        index_list[i]) & (temp_df_raw_emg.index <= \
                                            index_list[i+1])])

        
        del temp_df_emg, temp_df, temp_df_raw_emg
        
        #We now have the splits, reject ones that are too short
        xsens_pre = []
        for item in xsens_df_list:
            xsens_pre.append(item)

        del xsens_df_list

        delsys_pre = []
        for item in delsys_df_list:
            delsys_pre.append(item)  
                
        del delsys_df_list
        
        delsys_raw_pre = []
        for item in delsys_emg_df_list:
            delsys_raw_pre.append(item)
    
        return (xsens_full_length, delsys_full_length, delsys_raw_full_length,\
                xsens_pre, delsys_pre, delsys_raw_pre)
    
    def extract_shearing_part_extend(self, DataFile):
        
        delsys_data = DataFile.delsys_data
        xsens_data  = DataFile.xsens_data 
        emg_env     = DataFile.env_df
        
        #Make list of dataframes separated in time
        #1 for Shearing
        
        #Shift the HMM_label up
        xsens_data['HMM_UP240'] = xsens_data['HMM_labels'].shift(-480).fillna(method='ffill')
        emg_env['HMM_UP40'] = emg_env['HMM_labels'].shift(-80).fillna(method='ffill')
        #Shift the HMM_label down
        xsens_data['HMM_DOWN240'] = xsens_data['HMM_labels'].shift(480).fillna(method='bfill')
        emg_env['HMM_DOWN40']     = emg_env['HMM_labels'].shift(80).fillna(method='bfill')
        #OR them together
        xsens_data['OR_HMM'] = np.maximum(xsens_data['HMM_UP240'], xsens_data['HMM_DOWN240'])
        emg_env['OR_HMM'] = np.maximum(emg_env['HMM_UP40'],emg_env['HMM_DOWN40'])
        
        
        temp_df = xsens_data.loc[xsens_data['OR_HMM'] == 1]
        time_shift            = temp_df['time_ms'].shift(-1)
        temp_df.insert(len(temp_df.columns),'time_shift',time_shift,False)
        temp_df.insert(len(temp_df.columns),'eval',temp_df['time_shift'] - temp_df['time_ms'],False)
        index_list            = temp_df.loc[temp_df['eval'] > 1000].index
        
        #Make list of dataframes separated in time
        temp_df_emg = emg_env.loc[emg_env['OR_HMM'] == 1]
        time_shift                = temp_df_emg['time'].shift(-1)
        temp_df_emg.insert(len(temp_df_emg.columns),'time_shift',time_shift,False)
        temp_df_emg.insert(len(temp_df_emg.columns),'eval',temp_df_emg['time_shift'] - temp_df_emg['time'],False)
        index_list_emg            = temp_df_emg.loc[temp_df_emg['eval'] > 1].index
        
        #Make list of dataframes separated in time
        temp_df_raw_emg = delsys_data.loc[delsys_data['HMM_labels'] == 1]
        time_shift                    = temp_df_raw_emg['time'].shift(-1)
        temp_df_raw_emg.insert(len(temp_df_raw_emg.columns),'time_shift',time_shift,False)
        temp_df_raw_emg.insert(len(temp_df_raw_emg.columns),'eval',temp_df_raw_emg['time_shift'] - temp_df_emg['time'],False)
        index_list_raw_emg            = temp_df_raw_emg.loc[temp_df_raw_emg['eval'] > 1].index
        
        xsens_df_list  = []
        delsys_df_list = []
        delsys_emg_df_list = []
        for i in range(len(index_list)-1):
            if i == 0:
                xsens_df_list.append(temp_df.loc[temp_df.index\
                                        <= index_list[i]])
                delsys_df_list.append(temp_df_emg.loc[temp_df_emg.index\
                                        <= index_list[i]])
                delsys_emg_df_list.append(temp_df_raw_emg.loc[temp_df_raw_emg.index\
                                        <= index_list[i]])
            else:
                xsens_df_list.append(temp_df.loc[(temp_df.index > \
                                        index_list[i]) & (temp_df.index <= \
                                            index_list[i+1])])
                delsys_df_list.append(temp_df_emg.loc[(temp_df_emg.index > \
                                        index_list[i]) & (temp_df_emg.index <= \
                                            index_list[i+1])])
                delsys_emg_df_list.append(temp_df_raw_emg.loc[(temp_df_raw_emg.index > \
                                        index_list[i]) & (temp_df_raw_emg.index <= \
                                            index_list[i+1])])

        #We now have the splits, reject ones that are too short
        xsens_full_length = []
        for item in xsens_df_list:
            xsens_full_length.append(item)

        del xsens_df_list

        delsys_full_length = []
        for item in delsys_df_list:
            delsys_full_length.append(item)  
                
        del delsys_df_list
        
        delsys_raw_full_length = []
        for item in delsys_emg_df_list:
            delsys_raw_full_length.append(item)
    
        return (xsens_full_length, delsys_full_length, delsys_raw_full_length)
    
    def extract_shearing_part_pre(self, DataFile):
        from copy import copy
        
        delsys_data = DataFile.delsys_data
        xsens_data = DataFile.xsens_data
        emg_env = DataFile.env_df

        
        #Make list of dataframes separated in time
        #1 for Shearing
        
        #Shift the HMM_label up (8seconds not 5)
        delsys_data['HMM_5s'] = delsys_data['HMM_labels'].shift(-17186).fillna(method='ffill')
        xsens_data['HMM_5s'] = xsens_data['HMM_labels'].shift(-480).fillna(method='ffill')
        emg_env['HMM_5s'] = emg_env['HMM_labels'].shift(-80).fillna(method='ffill')
        
        #HMM_5s must = 1, HMM must = 0
        delsys_data['5s_pre'] = delsys_data.apply(lambda row: (np.logical_xor(row['HMM_labels'],row['HMM_5s'])) & (row['HMM_5s']==1),axis=1)
        xsens_data['5s_pre'] = xsens_data.apply(lambda row: (np.logical_xor(row['HMM_labels'],row['HMM_5s'])) & (row['HMM_5s']==1),axis=1)
        emg_env['5s_pre'] = emg_env.apply(lambda row: (np.logical_xor(row['HMM_labels'],row['HMM_5s'])) & (row['HMM_5s']==1),axis=1)

        
        temp_df = xsens_data.loc[xsens_data['5s_pre'] == 1]
        time_shift            = temp_df['time_ms'].shift(-1)
        temp_df.insert(len(temp_df.columns),'time_shift',time_shift,False)
        temp_df.insert(len(temp_df.columns),'eval',temp_df['time_shift'] - temp_df['time_ms'],False)
        index_list            = temp_df.loc[temp_df['eval'] > 1000].index
        
        #Make list of dataframes separated in time
        temp_df_emg = emg_env.loc[emg_env['5s_pre'] == 1]
        time_shift                = temp_df_emg['time'].shift(-1)
        temp_df_emg.insert(len(temp_df_emg.columns),'time_shift',time_shift,False)
        temp_df_emg.insert(len(temp_df_emg.columns),'eval',temp_df_emg['time_shift'] - temp_df_emg['time'],False)
        index_list_emg            = temp_df_emg.loc[temp_df_emg['eval'] > 1].index
        
        #Make list of dataframes separated in time
        temp_df_raw_emg = delsys_data.loc[delsys_data['5s_pre'] == 1]
        time_shift                    = temp_df_raw_emg['time'].shift(-1)
        temp_df_raw_emg.insert(len(temp_df_raw_emg.columns),'time_shift',time_shift,False)
        temp_df_raw_emg.insert(len(temp_df_raw_emg.columns),'eval',temp_df_raw_emg['time_shift'] - temp_df_emg['time'],False)
        index_list_raw_emg            = temp_df_raw_emg.loc[temp_df_raw_emg['eval'] > 1].index
        
        xsens_df_list  = []
        delsys_df_list = []
        delsys_emg_df_list = []
        for i in range(len(index_list)-1):
            if i == 0:
                xsens_df_list.append(temp_df.loc[temp_df.index\
                                        <= index_list[i]])
                delsys_df_list.append(temp_df_emg.loc[temp_df_emg.index\
                                        <= index_list[i]])
                delsys_emg_df_list.append(temp_df_raw_emg.loc[temp_df_raw_emg.index\
                                        <= index_list[i]])
            else:
                xsens_df_list.append(temp_df.loc[(temp_df.index > \
                                        index_list[i]) & (temp_df.index <= \
                                            index_list[i+1])])
                delsys_df_list.append(temp_df_emg.loc[(temp_df_emg.index > \
                                        index_list[i]) & (temp_df_emg.index <= \
                                            index_list[i+1])])
                delsys_emg_df_list.append(temp_df_raw_emg.loc[(temp_df_raw_emg.index > \
                                        index_list[i]) & (temp_df_raw_emg.index <= \
                                            index_list[i+1])])

        #We now have the splits, reject ones that are too short
        xsens_pre = []
        for item in xsens_df_list:
            xsens_pre.append(item)

        del xsens_df_list

        delsys_pre = []
        for item in delsys_df_list:
            delsys_pre.append(item)  
                
        del delsys_df_list
        
        delsys_raw_pre = []
        for item in delsys_emg_df_list:
            delsys_raw_pre.append(item)
    
        return (xsens_full_length, delsys_full_length, delsys_raw_full_length)
    
    #TODO: this can probably be removed        
    def bhk_ensemble_compare(self,title):
        joints = ['Pelvis_T8_z','jRightHip_z','jLeftHip_z','jRightKnee_z',\
                  'jLeftKnee_z']
        
        for joint in joints:
            plt.figure()
            early_stat_df = Summary.ensemble_average(self.early_10_xsens,2000,joint)
            late_stat_df = Summary.ensemble_average(self.late_10_xsens,2000,joint)
            
            Summary.plot_ensemble(early_stat_df,'r',joint,str(self.run),True, title)
            Summary.plot_ensemble(late_stat_df,'b',joint, str(self.run),False, title)
    
    #Good method - make optional? Some analysis would need the raw data    
    @staticmethod
    def ensemble_average(list_dataframes,num_samples,column_name):
        
        for i in range(0,len(list_dataframes)):
            if i == 0:
                resampled_df = Summary.resample_df(\
                                list_dataframes[i],num_samples, column_name)
            else:
                temp_df = Summary.resample_df(list_dataframes[i], num_samples, column_name)
                resampled_df['Angle (deg) {}'.format(i+1)] = temp_df['Angle (deg)']
                
        #Make these in new dataframes... something is going on re: axes.....
        resampled_stat_df = pd.DataFrame()
        resampled_stat_df['mean'] = resampled_df.drop('time (N)',axis=1).mean(axis=1)
        resampled_stat_df['std'] = resampled_df.drop('time (N)',axis=1).std(axis=1)
        resampled_stat_df['+sig'] = resampled_df.drop('time (N)',axis=1).std(axis=1)\
                                        + resampled_stat_df['mean']
        resampled_stat_df['-sig'] = -resampled_df.drop('time (N)',axis=1).std(axis=1)\
                                        + resampled_stat_df['mean']
        resampled_stat_df['time (N)'] = resampled_df['time (N)']
                                        
        return resampled_stat_df
    @staticmethod
    def resample_df(df, num_samples,STR):
        #Enter the joint angle here
        #STR = 'jRightKnee_z'
        test = df[STR]
        adata = np.zeros(num_samples)
        atime = np.linspace(0,1,num_samples)
        #For x in range of desired number of samples
        for x in range(0,num_samples):
            sample = x
            multiple = len(test)/num_samples
            actual_sample = sample*multiple
            
            lower_val = test[int(np.floor(actual_sample))]
            higher_val = test[int(np.ceil(actual_sample))]
            distance = np.mod(actual_sample,1)
            
            #Interpolation
            actual_value = (1-distance)*lower_val + distance*higher_val
            adata[x] = actual_value
        
        temp_df = pd.DataFrame(data=[atime,adata])
        return_df = temp_df.T
        return_df.columns = ['time (N)','Angle (deg)']
        return return_df

        
    @staticmethod
    def plot_ensemble(resampled_stat_df,linestyle,joint, run, early, title):
        line, = plt.plot(resampled_stat_df['time (N)'],resampled_stat_df['mean'],\
                                          linestyle+'-', label="Run "+str(run))
        line1, = plt.plot(resampled_stat_df['time (N)'],resampled_stat_df['+sig'],\
                                           linestyle+'--', label="+std",alpha=0.2)
        line2, = plt.plot(resampled_stat_df['time (N)'],resampled_stat_df['-sig'],\
                                           linestyle+'--', label="-std",alpha=0.2)
        plt.title(title + ' ' + str(joint) +' Early vs. Late '+'Run '+ str(run))
        plt.xlabel('Time (N)')
        plt.ylabel('Joint angle (deg)')
    

def compare_day_back(dir_path,joint,runs):
        title = joint + ' comparison across day'
        
        ax = plt.figure()
        
        for i in range(1,runs+1):
            mySummary = Summary(dir_path,i)
            
            early_stat_df = Summary.ensemble_average(mySummary.xsens_all,\
                                                     2500,joint)
            plt.plot(early_stat_df['time (N)'],early_stat_df['mean'],\
                                                label="Run "+str(i),\
                                                alpha=0.7)
            
            
            #late_stat_df = Summary.ensemble_average(mySummary.late_10_xsens,\
            #                                        2000,joint)
            #plt.plot(late_stat_df['time (N)'],late_stat_df['mean'],\
            #                                    label="End of run "+str(i),\
            #                                    alpha=0.7)

        ax.legend()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
            
            
def compare_day_LP(dir_path,runs):
    '''
    Enter a directory containing the days shearing data
    The joints will look at saggital plane hip and back angle
    Enter how many runs there are in the folder
    '''
    from tqdm import tqdm
    
    fig, axes = plt.subplots(4,1,sharex=True)
    fig.suptitle(dir_path + ' comparison across day')
    
    joints = ['Pelvis_T8_z','jRightHip_z','jRightKnee_z']
        
    for i in tqdm(range(1,runs+1)):
        mySummary = Summary(dir_path,i)
        
        lumbar_df = Summary.ensemble_average(mySummary.xsens_all,\
                                                 2500,joints[0])
        
        pelvis_df = Summary.ensemble_average(mySummary.xsens_all,\
                                                 2500,joints[1])
        
        knee_df = Summary.ensemble_average(mySummary.xsens_all,\
                                                 2500,joints[2])
        
        axes[1].plot(lumbar_df['time (N)'],lumbar_df['mean'],\
                                            label="Run "+str(i),\
                                            alpha=0.7)
        axes[2].plot(pelvis_df['time (N)'],pelvis_df['mean'],\
                                            label="Run "+str(i),\
                                            alpha=0.7)
        axes[3].plot(lumbar_df['time (N)'],\
                        lumbar_df['mean']+pelvis_df['mean'],\
                            label="Run "+str(i),alpha=0.7)
        
        axes[0].plot(lumbar_df['time (N)'],\
                        knee_df['mean'],\
                            label="Run "+str(i),alpha=0.7)

    axes[0].set_title('Knee Flexion angle')
    axes[1].set_title('Lumbar Flexion angle')
    axes[2].set_title('Hip flexion angle')
    axes[3].set_title('Combined hip/lumbar flexion angle')
    
    
    for i in range(4):
        axes[i].set_ylabel('Angle (degrees)')
        axes[i].legend()
    
    axes[2].set_xlabel('Time (N)')

def compare_day(dir_path,joints,runs):        
        fig,ax = plt.subplots(2,1,sharex = True)
        
        for i in range(1,runs+1):
            mySummary = Summary(dir_path,i)
            
            Q = mySummary.late_10_xsens
            A = [len(x) for x in Q]
            
            min(A)
            print('The number of sheep in run is {}'.format(len(Q)))
            print('The minimum length of a sheep is {}'.format(min(A)))
            
            Q = [x for x in Q if len(x) > 2000]
            A = [len(x) for x in Q]
            print('The number of sheep in run is now {}'.format(len(Q)))
            print('The length of min x is now {}'.format(min(A)))
            
            early_stat_df = Summary.ensemble_average(Q,\
                                                     2000,joints[0])
            ax[0].plot(early_stat_df['time (N)'],early_stat_df['mean'],\
                                                label="Run "+str(i),\
                                                alpha=0.7)
            
            early_stat_df = Summary.ensemble_average(Q,\
                                                     2000,joints[1])
            ax[1].plot(early_stat_df['time (N)'],early_stat_df['mean'],\
                                                label="Run "+str(i),\
                                                alpha=0.7)
            
            
            #late_stat_df = Summary.ensemble_average(mySummary.late_10_xsens,\
            #                                        2000,joint)
            #plt.plot(late_stat_df['time (N)'],late_stat_df['mean'],\
            #                                    label="End of run "+str(i),\
            #                                    alpha=0.7)

        ax[0].legend()
        ax[1].legend()
        for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] +
             ax[0].get_xticklabels() + ax[0].get_yticklabels()):
            item.set_fontsize(16)
            
        for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] +
             ax[1].get_xticklabels() + ax[1].get_yticklabels()):
            item.set_fontsize(16)
            
            
def plot_example_data(dir_path):
    
    plt.figure()
    plt.subplot(2,1,1)
    mySummary = Summary(dir_path,1)
    early_stat_df = Summary.ensemble_average(mySummary.early_10_xsens,\
                                                     2000,'Pelvis_T8_z')
    plt.plot(early_stat_df['time (N)'],early_stat_df['mean'],\
                                                label="Lumbar flexion",\
                                                alpha=0.7)
    plt.subplot(2,1,2)
    early_stat_df = Summary.ensemble_average(mySummary.early_10_env,\
                                                     2000,'L3 Erector Spinae LEFT: EMG.A 3')
    

def plot_1_sheep(mySummary,i):
    xsens  = mySummary.xsens_all[i]
    env    = mySummary.envelope_all[i]
    delsys = mySummary.emg_all[i]
    
    import mean_freq_calc as mfc
    import STFT
    delsys_mean_freq, delsys_RMS = \
                            mfc.muscle_mean_freq(\
                                *STFT.shearer_spectogram(delsys,\
                                     'L3 Erector Spinae LEFT: EMG.A 3'))
    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(xsens['Pelvis_T8_z'])
    plt.subplot(3,1,2)
    plt.plot(env['L3 Erector Spinae LEFT: EMG.A 3'])
    plt.ylim((0,1))
    plt.subplot(3,1,3)
    plt.plot(delsys_mean_freq['time'],delsys_mean_freq['mean_f'])
    plt.ylim((0,100))
    
 
def plot_BM_per_sheep(time_list, mean_freq_list):
    L1_left = [x['L1 Erector Spinae LEFT'] for x in mean_freq_list]
    L3_left = [x['L3 Erector Spinae LEFT'] for x in mean_freq_list]
    L5_left = [x['L5 Multifidus LEFT'] for x in mean_freq_list]
    
    L1_right = [x['L1 Erector Spinae RIGHT'] for x in mean_freq_list]
    L3_right = [x['L3 Erector Spinae RIGHT'] for x in mean_freq_list]
    L5_right = [x['L5 Multifidus RIGHT'] for x in mean_freq_list]
    
    fig,axs = plt.subplots(3, sharey = True, sharex = True,)
    fig.suptitle('Mean frequency (per sheep)', fontweight = 'bold', size=16)


    axs[0].set_title('L1 Erector Spinae',size=14)
    axs[0].set_xlabel('Time',size=14)
    axs[0].set_ylabel('Frequency (Hz)',size=14)
    axs[0].plot(time_list, L1_left, 'o', label='L1 ES Left')
    axs[0].plot(time_list, L1_right, 'v', label='L1 ES Right')
    axs[0].legend()
    
    axs[1].set_title('L3 Erector Spinae',size=14)
    axs[1].set_xlabel('Time',size=14)
    axs[1].set_ylabel('Frequency (Hz)',size=14)
    axs[1].plot(time_list, L3_left, 'o', label = 'L3 ES Left')
    axs[1].plot(time_list, L3_right, 'v', label = 'L3 ES Right')
    axs[1].legend()
    
    axs[2].set_title('L5 Multifidus',size=14)
    axs[2].set_xlabel('Time',size=14)
    axs[2].set_ylabel('Frequency (Hz)',size=14)
    axs[2].plot(time_list, L5_left, 'o', label = 'L5 Multifidus Left')
    axs[2].plot(time_list, L5_right, 'v', label = 'L5 Multifidus Right')
    axs[2].legend()
    
    for ax in axs:
        ax.label_outer()
        
def plot_BM_env_per_sheep(time_list, mean_freq_list):
    L1_left = [x['L1 Erector Spinae LEFT: EMG.A 1'] for x in mean_freq_list]
    L3_left = [x['L3 Erector Spinae LEFT: EMG.A 3'] for x in mean_freq_list]
    L5_left = [x['L5 Multifidus LEFT: EMG.A 5'] for x in mean_freq_list]
    
    L1_right = [x['L1 Erector Spinae RIGHT: EMG.A 2'] for x in mean_freq_list]
    L3_right = [x['L3 Erector Spinae RIGHT: EMG.A 4'] for x in mean_freq_list]
    L5_right = [x['L5 Multifidus RIGHT: EMG.A 6'] for x in mean_freq_list]
    
    fig,axs = plt.subplots(3, sharey = True, sharex = True,)
    fig.suptitle('Mean frequency (per sheep)', fontweight = 'bold', size=16)


    axs[0].set_title('L1 Erector Spinae',size=14)
    axs[0].set_xlabel('Time',size=14)
    axs[0].set_ylabel('Frequency (Hz)',size=14)
    axs[0].plot(time_list, L1_left, 'o', label='L1 ES Left')
    axs[0].plot(time_list, L1_right, 'v', label='L1 ES Right')
    axs[0].legend()
    
    axs[1].set_title('L3 Erector Spinae',size=14)
    axs[1].set_xlabel('Time',size=14)
    axs[1].set_ylabel('Frequency (Hz)',size=14)
    axs[1].plot(time_list, L3_left, 'o', label = 'L3 ES Left')
    axs[1].plot(time_list, L3_right, 'v', label = 'L3 ES Right')
    axs[1].legend()
    
    axs[2].set_title('L5 Multifidus',size=14)
    axs[2].set_xlabel('Time',size=14)
    axs[2].set_ylabel('Frequency (Hz)',size=14)
    axs[2].plot(time_list, L5_left, 'o', label = 'L5 Multifidus Left')
    axs[2].plot(time_list, L5_right, 'v', label = 'L5 Multifidus Right')
    axs[2].legend()
    
    for ax in axs:
        ax.label_outer()        
        
def plot_env_per_sheep(time_list, env_list):
    
    
    muscle_list_left = ['L1 Erector Spinae LEFT: EMG.A 1',\
                   'L3 Erector Spinae LEFT: EMG.A 3',\
                   'L5 Multifidus LEFT: EMG.A 5' ]
    
    muscle_list_right = ['L1 Erector Spinae RIGHT: EMG.A 2',\
                   'L3 Erector Spinae RIGHT: EMG.A 4',\
                   'L5 Multifidus RIGHT: EMG.A 6' ]
    
    L1_left = [x[muscle_list_left[0]] for x in env_list]
    L3_left = [x[muscle_list_left[1]] for x in env_list]
    L5_left = [x[muscle_list_left[2]] for x in env_list]
    
    L1_right = [x[muscle_list_right[0]] for x in env_list]
    L3_right = [x[muscle_list_right[1]] for x in env_list]
    L5_right = [x[muscle_list_right[2]] for x in env_list]
    
    fig,axs = plt.subplots(3, sharey = True, sharex = True,)
    fig.suptitle('Ave. EMG Envelope magnitude (per sheep)', fontweight = 'bold', size=16)


    axs[0].set_title(muscle_list_left[0],size=14)
    axs[0].set_xlabel('Time',size=14)
    axs[0].set_ylabel('Magnitude (N)',size=14)
    axs[0].plot(time_list, L1_left, 'o', label='L1 ES Left')
    axs[0].plot(time_list, L1_right, 'v', label='L1 ES Right')
    axs[0].legend()
    
    axs[1].set_title(muscle_list_left[1],size=14)
    axs[1].set_xlabel('Time',size=14)
    axs[1].set_ylabel('Magnitude (N)',size=14)
    axs[1].plot(time_list, L3_left, 'o', label = 'L3 ES Left')
    axs[1].plot(time_list, L3_right, 'v', label = 'L3 ES Right')
    axs[1].legend()
    
    axs[2].set_title(muscle_list_left[2],size=14)
    axs[2].set_xlabel('Time',size=14)
    axs[2].set_ylabel('Magnitude (N)',size=14)
    axs[2].plot(time_list, L5_left, 'o', label = 'L5 Multifidus Left')
    axs[2].plot(time_list, L5_right, 'v', label = 'L5 Multifidus Right')
    axs[2].legend()
    
    for ax in axs:
        ax.label_outer()
        
def get_full_run_plot(dir_path,run):
    mySummary = Summary(dir_path,run)
    time_list, mean_freq_list = mySummary.per_sheep()
    plot_BM_per_sheep(time_list, mean_freq_list)
    
def get_full_run_env(dir_path,run):
    mySummary = Summary(dir_path,run)
    time_list, env_list = mySummary.per_sheep_env()
    plot_env_per_sheep(time_list, env_list)
    
if __name__ == '__main__':    
    pass
    #mySummary_test1 = Summary('D:\Data\Shearer 1 - Week 1\Tuesday',4,False)
    #get_full_run_env('D:\Data\Week 2\Shearer 3 (Week 2) - Tuesday',1)
    #pass
    
    
   