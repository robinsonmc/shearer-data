# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:57:34 2020

@author: mrobinson2
"""

import pathlib
import pandas as pd
import numpy as np

#Boiler plate which prevents Spyder crashing when graphing
try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass

debug = 0

class FeatureSet:
    #default dir path is in the repository
    def __init__(self,shearer=1,day='tuesday',dir_path='new_saved_features_11_subjects'):
        filelist = list(pathlib.Path(dir_path).glob('s{}_{}*'.format(shearer,day)))
        
        self.shearer = shearer
        self.day = day
        
        
        self.feature_list = []
        for element in filelist:
            temp = pd.read_csv(element)
            temp.drop('Unnamed: 0',axis=1,inplace=True)
            self.feature_list.append(temp)


    def store_features(self, dir_path):
        feature_list = self.features
        shearer = self.shearer
        day = self.day
        
        for i in range(len(feature_list)):
            save_path = 's{}_{}_run{}_features.csv'.format(shearer,day,i+1)
            saveas    = pathlib.Path(pathlib.Path(dir_path),save_path)
            feature_list[i].to_csv(saveas)
            
            
    def plot_features(self, features = None, filename=None):
        if not features:
            features = self.feature_list[0].columns
        
        
class AllShearersFeatures:
    def __init__(self,dir_path='new_saved_features_11_subjects',shearers=None):
        
        if not shearers:
            #Default all shearers
            shearers = [(1,'tuesday'),(1,'wednesday'),(1,'thursday'),\
                        (2,'thursday'),(3,'tuesday'),(4,'wednesday'),\
                        (5,'thursday'),(6,'friday'),(7,'monday'),(8,'tuesday'),\
                        (9,'wednesday'),(10,'thursday'),(11,'friday')]
                
        if shearers == 'LOPO': #Drop shearer 6 friday
            shearers = [(1,'tuesday'),(1,'wednesday'),(1,'thursday'),\
                        (2,'thursday'),(3,'tuesday'),(4,'wednesday'),\
                        (5,'thursday'),(7,'monday'),(8,'tuesday'),\
                        (9,'wednesday')]
        
        self.shearers = shearers
        self.dict_all_FS = {}
        for element in shearers:
            self.dict_all_FS[element] = FeatureSet(shearer=element[0],day=element[1],dir_path=dir_path)
            
       
    def get_subject_feature_raw(self, subject, feature):
        '''
        Return a list of numpy arrays of the feature (with run information)
    
        Parameters
        ----------
        subject   : int
                    Subject number (1-11)
        feature   : string
                    The name of the feature (column in dataframe)
        
        Returns
        -------
        feature   : list of pd.DataFrames
                    All datapoints of that feature 
                    (or random values if none) - includes run information  
        '''
        
        df_list = self.dict_all_FS[self.shearers[subject]].feature_list
        
        feature_list = [x[feature].fillna(method='bfill') for x in df_list]
        
        #TODO:
        #Need to be careful with empty features
        return feature_list


    
    def get_subject_feature_list(self, subject, feature):
        '''
        Return a numpy array of the feature (no run information)
    
        Parameters
        ----------
        subject   : int
                    Subject number (1-11)
        feature   : string
                    The name of the feature (column in dataframe)
        
        Returns
        -------
        feature   : np.array
                    All datapoints of that feature 
                    (or random values if none) - no run information.  
        '''
        
        df_list = self.dict_all_FS[self.shearers[subject]].feature_list
        
        feature_list = [x[feature].fillna(method='bfill') for x in df_list]
        
        flat_feature_list = np.concatenate(feature_list)
        
        #TODO:
        #Need to be careful with empty features
        return np.array(flat_feature_list)
           
    def get_subject_runs_feature(self, subject, feature):
        '''
        Return a numpy array of the feature (no run information)
    
        Parameters
        ----------
        subject   : int
                    Subject number (1-11)
        feature   : string
                    The name of the feature (column in dataframe)
        
        Returns
        -------
        feature   : np.array
                    All datapoint of that feature (or random values if none)  
        '''
        
        df_list = self.dict_all_FS[self.shearers[subject]].feature_list
        
        feature_list = [x[feature].fillna(method='bfill') for x in df_list]
        
        return feature_list
        
if __name__ == '__main__':
    
    #Harness no harness conditions
    harness = [(2,'thursday'),\
                        (5,'thursday'),(7,'monday'),(8,'tuesday'),\
                        (9,'wednesday')]
    
    no_harness = [(1,'tuesday'),(1,'wednesday'),(1,'thursday'),\
                        (3,'tuesday'),(4,'wednesday'),\
                        (6,'friday'),(10,'thursday'),(11,'friday')]
    
    #Shearers = none for all subjects
    A = AllShearersFeatures(dir_path='new_saved_features_11_subjects')


    
    
    