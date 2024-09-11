# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:40:26 2019

@author: mrobinson2
"""

import numpy as np
#from read_xsens import read_xsens_xml
from read_xsens_new import read_xsens_xml
from read_delsys import read_delsys_csv
import pandas as pd
import os
from pathlib import Path, PosixPath
import envelope
import pickle
import read_csv_datatypes as rcd

debug = 1

class DataFile:
    '''
    Class containing data of combined motion capture and EMG data, along with
    the EMG envelope calculated from MVC data.
    
    Initiate an empty DataFile with DataFile()
    or creaate a DataFile from a directory with DataFile(dir_path)
    '''
    def __init__(self, dir_path = None):
        #We need to load in A the movie file
        #B the xsens file
        #C the emg file
        #Keep a record and sync the two files together
        
        #Preferentially load from pickle file with same name
        #On completion write the pickle file, this will overwrite the previous
        #one
        
        self.dir_path = dir_path
        self.xsens_data = None
        self.delsys_data = None
        self.meta_data = {}
        self.dir_name = None
        self.envelope = None
        self.env_df = None
        
        self.MVC_FREQ = 3
        
        if debug == 1: 
            print('the filepath is: {}'.format(self.dir_path))
        
        if dir_path is not None:
            self.create_from_directory(dir_path)
            
        self.label_list = {}
        
    #TODO encapsulate the variables in the create from directory
    def create_from_directory(self, dir_path):
        '''Read the data from the specified directory and keep in memory. 
           Input is a directory containing the 
           data files collected, either as mvnx & csv or as possibly already
           labelled pickles.
           
           No output
        '''
        self.dir_path = Path(dir_path)
        self.dir_name = self.dir_path.parts[-1]
        files = os.listdir(dir_path)
        parent = Path(dir_path).parents[0]
        
        #TODO:
        #Try pickles first???
        #Do pickles already exist?
        #Do csv's already exist?
        if self.dir_name + '_mocap.csv' and self.dir_name + '_emg.csv' and \
        self.dir_name + '_envelope.csv' in files:
            print('reading from csv...')
            self.read_csv()
        
        else:
            if self.dir_name + '_emg.pickle' and self.dir_name + '_mocap.pickle'\
            in files:
                print('reading from pickles...')
                self.read_pickles()
            
                
            
            #If not, load from mvnx and csv files
            #Also create a metadata file that should contain the start-time etc?
            else:
                print('reading original files...')
                self.read_originals()
            
            if self.dir_name + '_envelope.pickle' in files:
                self.read_envelope_pickle()
            elif self.dir_name + '_envelope.csv' not in files:
                #When envelope is fixed, enable this    
                self.envelope =\
                    envelope.Envelope(parent,self.delsys_data,self.MVC_FREQ)
                self.write_envelope_pickle()
                self.env_df = self.envelope.env_df
        
    
    def read_pickles(self):
        self.read_xsens_data_from_pickle()
        self.read_delsys_data_from_pickle()
        
        
    def read_envelope_pickle(self):
        import pickle
        
        proper_path = self.get_path_from_dir(self.envelope_pickle_filename,\
                                         self.dir_path)
        
        with open(proper_path, 'rb') as handle:
            self.envelope = pickle.load(handle)
            
        self.env_df = self.envelope.env_df

    
    def read_originals(self):
        #Xsens motion capture
        file_path_xsens = self.dir_path / Path(self.dir_path.parts[-1]\
                                               + '.mvnx')
        if debug == 1: print('Loading: ' + str(file_path_xsens) + '...')
        self.xsens_data, start_time = self.read_xsens_data_from_file\
                                                        (file_path_xsens)
        self.meta_data['start_time'] = start_time
        self.xsens_data.index.name = 'datetime'
        assert(self.meta_data['start_time'] is not None)
        
        if debug == 1: 
            print('Done...')
            print(self.xsens_data)
        
        #Delsys EMG
        file_path_delsys = self.dir_path / Path(self.dir_path.parts[-1]\
                                                + '.csv')
        if debug == 1: print('Loading: '+ str(file_path_delsys)+'...')
        self.delsys_data = self.read_delsys_data_from_file(file_path_delsys)
        
        if debug == 1: print('Done...')
        
        #Set the index using the start_time
        self.delsys_data['timestamp'] = self.meta_data['start_time'] +\
                    pd.to_timedelta(self.delsys_data['time'],unit='seconds')
        self.delsys_data.index = np.array(self.delsys_data['timestamp'])
        self.delsys_data = self.delsys_data.tz_localize('Australia/Melbourne')
        self.delsys_data = self.delsys_data.drop(columns='timestamp')
        self.delsys_data.index.name = 'datetime'
        
        if debug == 1: print(self.delsys_data)

    
    def saveLabels(self):
        '''
        This is the overarching method call to trigger writing the 
        labels to the dataframe, and the saving both the xsens
        and delsys dataframes with labels to a pickle file.
        
        This should call other methods:
            1- write the labels to dataframes
            2- write to file
        '''
        self.modify_data_from_labels()
        
        self.write_pickles()
        print('Labels written to pickle...')
    
    def modify_data_from_labels(self):
        '''
        For each (sorted) key,value pair in the list:
            1- nothing for the first
            2- values between i-1 and i apply label i-1
        '''
        s_to_ms = 1000
        labels = self.label_list
        count = 0
        
        #IF custom labels this needs to change
        cat_type = pd.CategoricalDtype(categories=['shearing','catch_drag',\
                                                   'other'])
 
        #These are initialised to nan to match the placeholder created
        #by pandas when the labels column is created
        previous_k = np.nan
        previous_v = np.nan
        for k,v in sorted(labels.items()):
            if count > 1:
                label = previous_v
                if label != 'Start':
                    mocap_mask = (self.xsens_data['time_ms'] > previous_k) &\
                                    (self.xsens_data['time_ms'] <= k)
                    self.xsens_data.loc[mocap_mask,'labels'] = label
                    self.xsens_data['labels'] = \
                        self.xsens_data['labels'].astype(cat_type)
    
                    emg_mask = (self.delsys_data['time']*s_to_ms > previous_k)\
                                    &  (self.delsys_data['time']*s_to_ms <= k)
                    self.delsys_data.loc[emg_mask,'labels'] = label
                    self.delsys_data['labels'] =\
                        self.delsys_data['labels'].astype(cat_type)
                
            previous_k = k
            previous_v = v
            count += 1
    

    #def __str__(self):
    #    return 'OpenCV file {} Data: {} Start time: {}'.format(self.file_path,\
    #                        '\n',type(self.data[0]),'\n',self.data[1])
    
    def read_xsens_data_from_file(self, file_path):
        (data, starttime) = read_xsens_xml(str(file_path))
        data['labels'] = np.nan
        return (data, starttime)
    
    def read_delsys_data_from_file(self, file_path):
        data = read_delsys_csv(str(file_path))
        data['labels'] = np.nan
        return data 
    
    def read_xsens_data_from_pickle(self):
        self.xsens_data = pd.read_pickle(self.get_path_from_dir(\
                                self.mocap_pickle_filename,self.dir_path))
    
    def read_delsys_data_from_pickle(self):
        self.delsys_data = pd.read_pickle(self.get_path_from_dir(\
                                self.emg_pickle_filename,self.dir_path))
    
    def add_label(self,time_ms,label):
        self.label_list[time_ms] = label
        
    def write_pickles(self):
        import pickle
        #Xsens data 
        mocap_pickle_path = self.get_path_from_dir(self.mocap_pickle_filename,\
                                                   self.dir_path)
        self.xsens_data.to_pickle(mocap_pickle_path)
        
        #Delsys data
        emg_pickle_path = self.get_path_from_dir(self.emg_pickle_filename,\
                                                   self.dir_path)
        self.delsys_data.to_pickle(emg_pickle_path)
        
        #Envelope data
        envelope_pickle_path = self.get_path_from_dir(self.envelope_pickle_filename,\
                                                   self.dir_path)
        
        with open(envelope_pickle_path,'wb') as handle:
            pickle.dump(self.envelope, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
    def write_envelope_pickle(self):
        import pickle
        #Envelope data
        envelope_pickle_path = self.get_path_from_dir(self.envelope_pickle_filename,\
                                                   self.dir_path)
        
        with open(envelope_pickle_path,'wb') as handle:
            pickle.dump(self.envelope, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def remove_all_labels(self):
        #TODO Placeholder to check a confirmation
        if False:
            self.xsens_data['labels'] = np.nan
            self.delsys_data['labels'] = np.nan
    
    def emg_pickle_filename(self):
        return '_emg.pickle'
    
    def mocap_pickle_filename(self):
        return '_mocap.pickle'
    
    def envelope_pickle_filename(self):
        return '_envelope.pickle'
    
    def mocap_filename(self):
        return '.mvnx'
    
    def emg_filename(self):
        return '.csv'
    
    def label_pickle_filename(self):
        return '_labels.pickle'
        
    def get_path_from_dir(self,function,dir_path):
        dir_name = dir_path.parts[-1]
        #Changed / to + as on linux its backslash
        #Added Path to dir_name + function()
        #Used Path(,,) to join
        return Path(dir_path, Path(dir_name + function()))
    
    def save_labels_as_pickle(self):
        save_dir = self.get_path_from_dir(self.label_pickle_filename, \
                                          self.dir_path)
        
        with open(save_dir, 'wb') as handle:
            pickle.dump(self.xsens_data['labels'], handle, protocol = \
                        pickle.HIGHEST_PROTOCOL)
            
    def read_csv(self):
        
        dir_path = self.dir_path
        file_name_start = Path(dir_path).parts[-1]
        
        xsens_filepath    = Path(dir_path, file_name_start + '_mocap.csv')
        delsys_filepath   = Path(dir_path, file_name_start + '_emg.csv')
        envelope_filepath = Path(dir_path, file_name_start + '_envelope.csv')
        
              
        #Rename function
        def rename_apply(string):
            if ':' in string:
                parts = string.split(':')
                return parts[0]
            else:
                return string
            
        #Rename the column names w/ brackets
        def remove_brackets(string):
            import re
            A = re.sub("[\(\[].*?[\)\]]", "", string)
            B = B = re.sub("  ", " ", A)
            return B
        
        self.xsens_data = pd.read_csv(xsens_filepath,parse_dates=True, dtype=rcd.xsens_dtypes)
        self.delsys_data = pd.read_csv(delsys_filepath,parse_dates=True, dtype=rcd.delsys_dtypes)
        self.env_df = pd.read_csv(envelope_filepath,parse_dates=True, dtype=rcd.envelope_dtypes)
        
        print("Delsys Data columns initial: ")
        print(self.delsys_data.columns)
        
        print("Delsys Envelope columns initial: ")
        print(self.env_df.columns)
        
        columns_names_error_list = ['datetime','1 - Gluteus Medius (Left): EMG 1', '2 - L1 ES Left: EMG 2',
       '3 - Vastus Lateralis (Right): EMG 3',
       '4 - Vastus Lateralis (Left): EMG 4', '5 - L1 ES Right: EMG 5',
       '6 - L3 ES Left: EMG 6', '7 - L3 ES Right: EMG 7',
       '8 - L5 MF Left: EMG 8', '9 - L5 MF Right: EMG 9',
       '10 - Rectus Abdominis Right: EMG 10',
       '11 - Rectus Abdominis Left: EMG 11',
       '12 - External Oblique Right: EMG 12',
       '13 - External Oblique Left: EMG 13',
       '14 - Gluteus Medius Right: EMG 14',
       '15 - Biceps femoris (Hamstring) left: EMG 15',
       '16 - Biceps femoris (hamstring) Right: EMG 16', 'time', 'labels',
       'HMM_labels']
        
        env_columns_names_error_list = ['datetime', 'timestamp', '1 - Gluteus Medius ', '2 - L1 ES Left',
       '3 - Vastus Lateralis ', '4 - Vastus Lateralis ', '5 - L1 ES Right',
       '6 - L3 ES Left', '7 - L3 ES Right', '8 - L5 MF Left',
       '9 - L5 MF Right', '10 - Rectus Abdominis Right',
       '11 - Rectus Abdominis Left', '12 - External Oblique Right',
       '13 - External Oblique Left', '14 - Gluteus Medius Right',
       '15 - Biceps femoris left', '16 - Biceps femoris Right', 'time',
       'labels', 'HMM_labels']
        
        env_columns_names_error_list_2 = ['datetime', '1 - Gluteus Medius ', '2 - L1 ES Left',
       '3 - Vastus Lateralis ', '4 - Vastus Lateralis ', '5 - L1 ES Right',
       '6 - L3 ES Left', '7 - L3 ES Right', '8 - L5 MF Left',
       '9 - L5 MF Right', '10 - Rectus Abdominis Right',
       '11 - Rectus Abdominis Left', '12 - External Oblique Right',
       '13 - External Oblique Left', '14 - Gluteus Medius Right',
       '15 - Biceps femoris left', '16 - Biceps femoris Right', 'time',
       'labels', 'HMM_labels']
        
        
        if list(self.delsys_data.columns) == columns_names_error_list:
            self.delsys_data.columns = ['datetime','Gluteus Medius LEFT: EMG.A 11', 'L1 Erector Spinae LEFT: EMG.A 1',
       'Vastus Lateralis RIGHT: EMG.A 13',
       'Vastus Lateralis LEFT: EMG.A 14', 'L1 Erector Spinae RIGHT: EMG.A 2',
       'L3 Erector Spinae LEFT: EMG.A 3', 'L3 Erector Spinae RIGHT: EMG.A 4',
       'L5 Multifidus LEFT: EMG.A 5', 'L5 Multifidus RIGHT: EMG.A 6',
       'Rectus Abdominis (1cm up, 3cm out) RIGHT: EMG.A 7',
       'Rectus Abdominis (1cm up, 3cm out) LEFT: EMG.A 8',
       'External Oblique (15cm out) RIGHT: EMG.A 9',
       'External Oblique (15cm out) LEFT: EMG.A 10',
       'Gluteus Medius RIGHT: EMG.A 12',
      'Biceps Femoris LEFT: EMG.A 15',
       'Biceps Femoris RIGHT: EMG.A 16', 'time', 'labels',
       'HMM_labels']
            
        if list(self.env_df.columns) == env_columns_names_error_list:
            self.env_df.columns = ['datetime','timestamp','Gluteus Medius LEFT: EMG.A 11', 'L1 Erector Spinae LEFT: EMG.A 1',
       'Vastus Lateralis RIGHT: EMG.A 13',
       'Vastus Lateralis LEFT: EMG.A 14', 'L1 Erector Spinae RIGHT: EMG.A 2',
       'L3 Erector Spinae LEFT: EMG.A 3', 'L3 Erector Spinae RIGHT: EMG.A 4',
       'L5 Multifidus LEFT: EMG.A 5', 'L5 Multifidus RIGHT: EMG.A 6',
       'Rectus Abdominis (1cm up, 3cm out) RIGHT: EMG.A 7',
       'Rectus Abdominis (1cm up, 3cm out) LEFT: EMG.A 8',
       'External Oblique (15cm out) RIGHT: EMG.A 9',
       'External Oblique (15cm out) LEFT: EMG.A 10',
       'Gluteus Medius RIGHT: EMG.A 12',
      'Biceps Femoris LEFT: EMG.A 15',
       'Biceps Femoris RIGHT: EMG.A 16', 'time', 'labels',
       'HMM_labels']
            
        if list(self.env_df.columns) == env_columns_names_error_list_2:
            self.env_df.columns = ['datetime','Gluteus Medius LEFT: EMG.A 11', 'L1 Erector Spinae LEFT: EMG.A 1',
       'Vastus Lateralis RIGHT: EMG.A 13',
       'Vastus Lateralis LEFT: EMG.A 14', 'L1 Erector Spinae RIGHT: EMG.A 2',
       'L3 Erector Spinae LEFT: EMG.A 3', 'L3 Erector Spinae RIGHT: EMG.A 4',
       'L5 Multifidus LEFT: EMG.A 5', 'L5 Multifidus RIGHT: EMG.A 6',
       'Rectus Abdominis (1cm up, 3cm out) RIGHT: EMG.A 7',
       'Rectus Abdominis (1cm up, 3cm out) LEFT: EMG.A 8',
       'External Oblique (15cm out) RIGHT: EMG.A 9',
       'External Oblique (15cm out) LEFT: EMG.A 10',
       'Gluteus Medius RIGHT: EMG.A 12',
      'Biceps Femoris LEFT: EMG.A 15',
       'Biceps Femoris RIGHT: EMG.A 16', 'time', 'labels',
       'HMM_labels']
        
        #self.xsens_data.index.name = 'datetime'
        print("Delsys Envelope columns final: ")
        print(self.env_df.columns)
        
        try:
            self.xsens_data['datetime'] = pd.to_datetime(self.xsens_data['datetime'])#,format='ISO8601')#,infer_datetime_format = True)
            self.delsys_data['datetime'] = pd.to_datetime(self.delsys_data['datetime'])#,format='ISO8601')#,infer_datetime_format = True)
            self.env_df['datetime'] = pd.to_datetime(self.env_df['datetime'])#,format='ISO8601')#,infer_datetime_format = True)
        
            
            self.xsens_data.set_index('datetime',inplace=True)
            self.delsys_data.set_index('datetime',inplace=True)
            self.env_df.set_index('datetime',inplace=True)
            
        except KeyError:
            self.xsens_data['datetime'] = pd.to_datetime(self.xsens_data.index)#,format='ISO8601')#,infer_datetime_format = True)
            self.delsys_data['datetime'] = pd.to_datetime(self.delsys_data.index)#,format='ISO8601')#,infer_datetime_format = True)
            self.env_df['datetime'] = pd.to_datetime(self.env_df.index)#,format='ISO8601')#,infer_datetime_format = True)
        
            
            self.xsens_data.set_index('datetime',inplace=True)
            self.delsys_data.set_index('datetime',inplace=True)
            self.env_df.set_index('datetime',inplace=True)
        
        self.delsys_data.rename(columns=rcd.remove_number, inplace=True)
        self.env_df.rename(columns=rcd.remove_number,inplace=True)
        
        self.delsys_data.rename(columns=rcd.remove_brackets, inplace=True)
        self.env_df.rename(columns=rcd.remove_brackets,inplace=True)
        
        print(self.env_df.columns)
        
    def save_csv(self):
        import os
        
        dir_path = self.dir_path
        file_name_start = Path(dir_path).parts[-1]
        
        xsens_filepath    = Path(dir_path, file_name_start + '_mocap.csv')
        delsys_filepath   = Path(dir_path, file_name_start + '_emg.csv')
        envelope_filepath = Path(dir_path, file_name_start + '_envelope.csv')
        
        try:
            os.remove(xsens_filepath)
            os.remove(delsys_filepath)
            os.remove(envelope_filepath)
        except FileNotFoundError:
            pass
    
        self.xsens_data.to_csv(xsens_filepath,index_label='datetime')
        self.delsys_data.to_csv(delsys_filepath,index_label='datetime')
        self.env_df.to_csv(envelope_filepath,index_label='datetime')
        
        
    def set_files(self,xsens_data, delsys_data):
        self.xsens_data = xsens_data
        self.delsys_data = delsys_data
        
        
    def redo_index(self):

        try:
            #Xsens_data
            tempindex = self.xsens_data['Unnamed: 0']
            self.xsens_data.index = tempindex
            self.xsens_data.index.name = None
            self.xsens_data.drop('Unnamed: 0',axis=1,inplace=True)
            
            #delsys_data
            tempindex = self.delsys_data['Unnamed: 0']
            self.delsys_data.index = tempindex
            self.delsys_data.index.name = None
            self.delsys_data.drop('Unnamed: 0',axis=1,inplace=True)
        
        except ValueError:
            pass
        
if __name__ == '__main__':
    
    try:
        import IPython
        shell = IPython.get_ipython()
        shell.enable_matplotlib(gui='qt')
    except:
        pass 
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timezone


    def get_figsize(columnwidth, wf=0.5, hf=(5.**0.5-1.0)/2.0, ):
      """Parameters:
        - wf [float]:  width fraction in columnwidth units
        - hf [float]:  height fraction in columnwidth units.
                       Set by default to golden ratio.
        - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                               using \showthe\columnwidth
      Returns:  [fig_width,fig_height]: that should be given to matplotlib
      """
      fig_width_pt = columnwidth*wf 
      inches_per_pt = 1.0/72.27               # Convert pt to inch
      fig_width = fig_width_pt*inches_per_pt  # width in inches
      fig_height = fig_width*hf      # height in inches
      return [fig_width, fig_height]
  
    
    import matplotlib
    params = {'backend': 'Agg',
               'axes.labelsize': 10,
               'font.size': 10 } # extend as needed
    matplotlib.rcParams.update(params)
    plt.rcParams["font.family"] = "Times New Roman"
    hfont = {'fontname':'Times New Roman Regular'}
    #plt.rcParams['figure.constrained_layout.use'] = True
    
    # set figure size as a fraction of the columnwidth
    columnwidth = 245.71811
    #fig = plt.figure(figsize=get_figsize(columnwidth, wf=1.0, hf=0.3))
    
    # #Plot the subplots!
    # #2 x 2 with all sharing x
    #fig = plt.figure(figsize=get_figsize(505.89, hf = 1.41, wf=1.0))
    
    fig, axes = plt.subplots(3,1,figsize=get_figsize(500.484, hf = 0.5, wf=1.0),sharex=True)

    # print('start:')
    # new_df = DataFile('D:\Data\Shearer 1 - Week 1\Tuesday\s1_week1_tuesday_run4_part1')
    # #new_df = DataFile('D:\Data\Shearer 1 - Week 1\Tuesday')
    # print('Finished')
    
    # QQ = new_df.xsens_data[['Pelvis_T8_z']]
    # #zz = QQ[['jRightKnee_z','jRightKnee_y','jRightKnee_x']]
    # QQ.plot()
    
    # ZZ = new_df.delsys_data['L1 Erector Spinae LEFT']
    # ZZ.plot(secondary_y=True)
    
    # ZZ = new_df.delsys_data['L1 Erector Spinae RIGHT']
    # ZZ.plot(secondary_y=True)
    
    # del new_df
    plt.rcParams["timezone"] = 'Australia/Sydney'
    # plt.figure()
    
    #Shearer 2 not big diff
    print('start 2:')
    #new_df = DataFile('D:\Data\Week 3\Shearer 8 (Week 3) - Tuesday\s8_week3_tuesday_run3_part3')
    new_df = DataFile('D:\Data\Shearer 1 - Week 1\Tuesday\s1_week1_tuesday_run1_part1')
    print('Finished')
    
    
    #dt1 = pd.to_datetime('2020-02-11 14:40:00.984000+11:00')
    #dt2 = pd.to_datetime('2020-02-11 14:48:31.917000+11:00')
    
    dt1 = pd.to_datetime('2019-07-23 08:07:00.902000+10:00')
    dt2 = pd.to_datetime('2019-07-23 08:13:00.902000+10:00')
    
    
    
    dt_qq_1 = new_df.xsens_data.index.get_loc(dt1,method='nearest')
    dt_qq_2 = new_df.xsens_data.index.get_loc(dt2,method='nearest')
    
    #QQ = new_df.xsens_data[['Pelvis_T8_z']].iloc[dt_qq_1:dt_qq_2]
    #zz = QQ[['jRightKnee_z','jRightKnee_y','jRightKnee_x']]
    #QQ.plot()
    
    new_df.xsens_data = new_df.xsens_data.iloc[dt_qq_1:dt_qq_2]
    
    
    
    
    dt_zz_1 = new_df.delsys_data.index.get_loc(dt1,method='nearest')
    dt_zz_2 = new_df.delsys_data.index.get_loc(dt2,method='nearest')
    
    #ZZ = new_df.delsys_data['L1 Erector Spinae LEFT'].iloc[dt_zz_1:dt_zz_2]
    #ZZ.plot(secondary_y=True)
    new_df.delsys_data = new_df.delsys_data.iloc[dt_zz_1:dt_zz_2]
    
    new_df.delsys_data[['L1 Erector Spinae LEFT', 'L1 Erector Spinae RIGHT',
       'L3 Erector Spinae LEFT', 'L3 Erector Spinae RIGHT',
       'L5 Multifidus LEFT', 'L5 Multifidus RIGHT', 'Rectus Abdominis RIGHT',
       'Rectus Abdominis LEFT', 'External Oblique RIGHT',
       'External Oblique LEFT', 'Gluteus Medius LEFT', 'Gluteus Medius RIGHT',
       'Vastus Lateralis RIGHT', 'Vastus Lateralis LEFT',
       'Biceps Femoris LEFT', 'Biceps Femoris RIGHT']] = \
    new_df.delsys_data[['L1 Erector Spinae LEFT', 'L1 Erector Spinae RIGHT',
       'L3 Erector Spinae LEFT', 'L3 Erector Spinae RIGHT',
       'L5 Multifidus LEFT', 'L5 Multifidus RIGHT', 'Rectus Abdominis RIGHT',
       'Rectus Abdominis LEFT', 'External Oblique RIGHT',
       'External Oblique LEFT', 'Gluteus Medius LEFT', 'Gluteus Medius RIGHT',
       'Vastus Lateralis RIGHT', 'Vastus Lateralis LEFT',
       'Biceps Femoris LEFT', 'Biceps Femoris RIGHT']] * 1000
    
    
    
    xformatter = mdates.DateFormatter('%H:%M')
    
    
    
    
    #WITHOUT THE SHEARING
    #emg_no_shear = new_df.delsys_data.loc[new_df.delsys_data['HMM_labels'] == 1]
    #imu_no_shear = new_df.xsens_data.loc[new_df.xsens_data['HMM_labels'] == 1]
    
    #XX = new_df.delsys_data['L5 Multifidus LEFT'].iloc[dt_zz_1:dt_zz_2]
    #XX.plot(secondary_y=True)
    
    #del new_df
    
    #fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=1, sharex=True,sharey='none')
    #ax2.plot(ZZ,'k')
    #ax1.plot(QQ)
    #a = 5
    #b = 15
    plt.rcParams["timezone"] = 'Australia/Sydney'
    
    #new_df.xsens_data.rename(index={'':'index'},inplace='True')
    import seaborn as sns
    #plt.figure()
    #sns.scatterplot(data=new_df.xsens_data,x='datetime',y='Pelvis_T8_z', hue='HMM_labels')
    #plt.figure()
    #sns.scatterplot(data=new_df.delsys_data,x='datetime',y='L1 Erector Spinae LEFT', hue='HMM_labels')
    
    
    #new_df.xsens_data['datetime'].head(5)
    
    sns.set_style("ticks")
    
    #define dimensions of subplots (rows, columns)
    #fig, axes = plt.subplots(3, 1, sharex=True)
    sns.lineplot(data=new_df.xsens_data,x='datetime',y='Pelvis_T8_z', ax=axes[0])
    sns.lineplot(data=new_df.delsys_data,x='datetime',y='L1 Erector Spinae LEFT',ax=axes[1])
    sns.lineplot(data=new_df.delsys_data,x='datetime',y='L5 Multifidus LEFT',ax=axes[2])
    #sns.lineplot(data=emg_no_shear,x='datetime',y='L1 Erector Spinae LEFT',ax=axes[1])
    #sns.lineplot(data=emg_no_shear,x='datetime',y='L5 Multifidus LEFT',ax=axes[2])
    #sns.lineplot(data=imu_no_shear,x='datetime',y='Pelvis_T8_z',ax=axes[0])
    
    # dtA = pd.to_datetime('2020-02-11 14:40:35.004000+11:00')
    # dtB = pd.to_datetime('2020-02-11 14:40:55.917000+11:00')
    # axes[0].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    # axes[1].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    # axes[2].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    
    dtA = pd.to_datetime('2019-07-23 08:07:23.004000+10:00')
    dtB = pd.to_datetime('2019-07-23 08:07:44.917000+10:00')
    axes[0].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    axes[1].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    axes[2].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    
    # dtA = pd.to_datetime('2020-02-11 14:43:19.004000+11:00')
    # dtB = pd.to_datetime('2020-02-11 14:43:42.917000+11:00')
    # axes[0].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    # axes[1].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    # axes[2].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    
    dtA = pd.to_datetime('2019-07-23 08:09:33.004000+10:00')
    dtB = pd.to_datetime('2019-07-23 08:09:50.917000+10:00')
    axes[0].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    axes[1].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    axes[2].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    
    # dtA = pd.to_datetime('2020-02-11 14:46:01.004000+11:00')
    # dtB = pd.to_datetime('2020-02-11 14:46:20.917000+11:00')
    # axes[0].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    # axes[1].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    # axes[2].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    
    dtA = pd.to_datetime('2019-07-23 08:11:30.004000+10:00')
    dtB = pd.to_datetime('2019-07-23 08:11:45.917000+10:00')
    axes[0].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    axes[1].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    axes[2].axvspan(dtA, dtB, color='b', alpha=0.2, lw=0)
    
    plt.rcParams["timezone"] = 'Australia/Sydney'
    axes[2].get_xaxis().set_major_formatter(xformatter)
    
    axes[0].set_xlim(xmin=new_df.xsens_data.index[0],xmax=new_df.xsens_data.index[-1])
    axes[1].set_xlim(xmin=new_df.delsys_data.index[0],xmax=new_df.xsens_data.index[-1])
    axes[2].set_xlim(xmin=new_df.delsys_data.index[0],xmax=new_df.xsens_data.index[-1])
    
   #axes[0].get_xaxis().set_ticks([])
    #axes[1].get_xaxis().set_ticks([])
    
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[2].set_xlabel('Time of Day (H:m)')
    axes[0].set_ylabel('P-T8 Flex ($^\circ$)')
    axes[1].set_ylabel('L1 ES-L (mV)')
    axes[2].set_ylabel('L1 MF-L (mV)')
    axes[0].set_title('Subject 1: Spinal flexion and muscle activity raw data')
    
    #fig.tight_layout()
    
    fig.savefig("C:\\Users\\robin\\Documents\\raw_data_constrained_first.pdf", format = 'pdf')
    
    del new_df
    
    #Shearer 2 not big diff
    print('start final:')
    new_df = DataFile('D:\Data\Week 3\Shearer 8 (Week 3) - Tuesday\s8_week3_tuesday_run3_part3')
    #new_df = DataFile('D:\Data\Shearer 1 - Week 1\Tuesday\s1_week1_tuesday_run1_part1')
    print('Finished')
    
    params = {'backend': 'Agg',
               'axes.labelsize': 10,
               'font.size': 10 } # extend as needed
    matplotlib.rcParams.update(params)
    plt.rcParams["font.family"] = "Times New Roman"
    hfont = {'fontname':'Times New Roman Regular'}
    
    fig, axes = plt.subplots(3,1,figsize=get_figsize(500.484, hf = 0.5, wf=1.0),sharex=True)
    
    
    dt1 = pd.to_datetime('2020-02-11 14:40:00.984000+11:00')
    dt2 = pd.to_datetime('2020-02-11 14:48:31.917000+11:00')
    
    dt_qq_1 = new_df.xsens_data.index.get_loc(dt1,method='nearest')
    dt_qq_2 = new_df.xsens_data.index.get_loc(dt2,method='nearest')
    
    #QQ = new_df.xsens_data[['Pelvis_T8_z']].iloc[dt_qq_1:dt_qq_2]
    #zz = QQ[['jRightKnee_z','jRightKnee_y','jRightKnee_x']]
    #QQ.plot()
    
    new_df.xsens_data = new_df.xsens_data.iloc[dt_qq_1:dt_qq_2]
    
    
    dt_zz_1 = new_df.delsys_data.index.get_loc(dt1,method='nearest')
    dt_zz_2 = new_df.delsys_data.index.get_loc(dt2,method='nearest')
    
    #ZZ = new_df.delsys_data['L1 Erector Spinae LEFT'].iloc[dt_zz_1:dt_zz_2]
    #ZZ.plot(secondary_y=True)
    new_df.delsys_data = new_df.delsys_data.iloc[dt_zz_1:dt_zz_2]
    
    new_df.delsys_data[['L1 Erector Spinae LEFT', 'L1 Erector Spinae RIGHT',
       'L3 Erector Spinae LEFT', 'L3 Erector Spinae RIGHT',
       'L5 Multifidus LEFT', 'L5 Multifidus RIGHT', 'Rectus Abdominis RIGHT',
       'Rectus Abdominis LEFT', 'External Oblique RIGHT',
       'External Oblique LEFT', 'Gluteus Medius LEFT', 'Gluteus Medius RIGHT',
       'Vastus Lateralis RIGHT', 'Vastus Lateralis LEFT',
       'Biceps Femoris LEFT', 'Biceps Femoris RIGHT']] = \
    new_df.delsys_data[['L1 Erector Spinae LEFT', 'L1 Erector Spinae RIGHT',
       'L3 Erector Spinae LEFT', 'L3 Erector Spinae RIGHT',
       'L5 Multifidus LEFT', 'L5 Multifidus RIGHT', 'Rectus Abdominis RIGHT',
       'Rectus Abdominis LEFT', 'External Oblique RIGHT',
       'External Oblique LEFT', 'Gluteus Medius LEFT', 'Gluteus Medius RIGHT',
       'Vastus Lateralis RIGHT', 'Vastus Lateralis LEFT',
       'Biceps Femoris LEFT', 'Biceps Femoris RIGHT']] * 1000
    
    
    xformatter = mdates.DateFormatter('%H:%M')
    
    sns.lineplot(data=new_df.xsens_data,x='datetime',y='Pelvis_T8_z', ax=axes[0],color='#ff7f0e')
    sns.lineplot(data=new_df.delsys_data,x='datetime',y='L1 Erector Spinae LEFT',ax=axes[1],color='#ff7f0e')
    sns.lineplot(data=new_df.delsys_data,x='datetime',y='L5 Multifidus LEFT',ax=axes[2],color='#ff7f0e')
    
    dtA = pd.to_datetime('2020-02-11 14:40:35.004000+11:00')
    dtB = pd.to_datetime('2020-02-11 14:40:55.917000+11:00')
    axes[0].axvspan(dtA, dtB, color='orange', alpha=0.2, lw=0)
    axes[1].axvspan(dtA, dtB, color='orange', alpha=0.2, lw=0)
    axes[2].axvspan(dtA, dtB, color='orange', alpha=0.2, lw=0)
    
    dtA = pd.to_datetime('2020-02-11 14:43:19.004000+11:00')
    dtB = pd.to_datetime('2020-02-11 14:43:42.917000+11:00')
    axes[0].axvspan(dtA, dtB, color='orange', alpha=0.2, lw=0)
    axes[1].axvspan(dtA, dtB, color='orange', alpha=0.2, lw=0)
    axes[2].axvspan(dtA, dtB, color='orange', alpha=0.2, lw=0)
    
    dtA = pd.to_datetime('2020-02-11 14:46:01.004000+11:00')
    dtB = pd.to_datetime('2020-02-11 14:46:20.917000+11:00')
    axes[0].axvspan(dtA, dtB, color='orange', alpha=0.2, lw=0)
    axes[1].axvspan(dtA, dtB, color='orange', alpha=0.2, lw=0)
    axes[2].axvspan(dtA, dtB, color='orange', alpha=0.2, lw=0)
    
    plt.rcParams["timezone"] = 'Australia/Sydney'
    axes[2].get_xaxis().set_major_formatter(xformatter)
    
    
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[2].set_xlabel('Time of Day (H:m)')
    
    
    axes[0].set_ylabel('P-T8 Flex ($^\circ$)')
    axes[1].set_ylabel('L1 ES-L (mV)')
    axes[2].set_ylabel('L1 MF-L (mV)')
    
    
    axes[0].set_title('Subject 6: Spinal flexion and muscle activity raw data')
    
    
    #axes[0].get_xaxis().set_ticks([])
    #axes[1].get_xaxis().set_ticks([])
    
    axes[0].set_xlim(xmin=new_df.xsens_data.index[0],xmax=new_df.xsens_data.index[-1])
    axes[1].set_xlim(xmin=new_df.delsys_data.index[0],xmax=new_df.xsens_data.index[-1])
    axes[2].set_xlim(xmin=new_df.delsys_data.index[0],xmax=new_df.xsens_data.index[-1])
    
    
    #from matplotlib.ticker import ScalarFormatter
    #xfmt = ScalarFormatter()
    #xfmt.set_powerlimits((-4,4))  # Or whatever your limits are . . .
    #{{ Make your plot }}
    #axes[1].get_yaxis().set_major_formatter(xfmt)
    #axes[2].get_yaxis().set_major_formatter(xfmt)
    #axes[4].get_yaxis().set_major_formatter(xfmt)
    #axes[5].get_yaxis().set_major_formatter(xfmt)
    
    plt.rcParams["timezone"] = 'Australia/Sydney'
    
    #fig.tight_layout()
    fig.savefig("C:\\Users\\robin\\Documents\\raw_data_constrained_second.pdf", format = 'pdf')
    #ax3.plot(XX,'k')
    
    #plt.figure()
    #new_df.delsys_data['HMM_labels'].plot()
    
    
    #YY = new_df.delsys_data()
    #new_df.save_csv()
    
    #test_df = DataFile('D:\Data\Week 2\Shearer 2 (Week 2) - Monday\s2_week2_monday_run2_part2')
    #test_df.xsens_data
#    #Get all the files
#    def generate_file_list(directory):
#    #Generate the file list - need directories with _run*_ in the name
#        file_list = []
#        for filename in Path(directory).rglob('*_run?'):
#            file_list.append(filename)
#            
#        for filename in Path(directory).rglob('*_run?_part?'):
#            file_list.append(filename)
#        
#        indices_to_delete = []        
#        for i in range(len(file_list)):
#            if 'emg' in str(file_list[i]): indices_to_delete.append(i)
#            if 'Shearer 10' in str(file_list[i]): indices_to_delete.append(i)
#            if 'Shearer 1 - Week 1\Monday' in str(file_list[i]): indices_to_delete.append(i)
#            
#        for index in sorted(indices_to_delete,reverse=True):
#            del file_list[index]
#            
#        return file_list
#    
#    
#    import pathlib
#    file_list = generate_file_list('D:\Data')
#    
#    #file_list = file_list[-5:]
#    
#    from tqdm import tqdm
#    for element in tqdm(file_list):
#        myData = DataFile(str(element))
#        if myData.xsens_data['labels'].notna().sum() > 0:
#            myData.save_labels_as_pickle()
        
    


