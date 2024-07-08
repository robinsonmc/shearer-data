# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:57:34 2020

@author: mrobinson2
"""

import pathlib
import pandas as pd
#import plotting.feature_plotting as fp
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
from distcorr import distcorr

try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass


debug = 0

class FeatureSet:
    #Changed to E from D:/ data because of usb-c dock
    def __init__(self,shearer=1,day='tuesday',dir_path='D:\Data\saved_features'):
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
        
        #fig = fp.get_figure()
        #fp.get_plots(self.feature_list, features, shearer=self.shearer) #P should be a column layout
        #fp.show_plot(p, filename)                 #Does the filename end up here?
        
        
class AllShearersFeatures:
    def __init__(self,dir_path='D:\Data\new_saved_features_11_subjects',shearers=None):
        
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
            
    def plot_for_all_shearers(self,feature):
        #fp.compare_feature_plot(self.dict_all_FS,feature)
        pass
        
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
        
        #flat_feature_list = [item for sublist in feature_list\
        #                     for item in sublist]
        
        #TODO:
        #Need to be careful with empty features
        return np.array(flat_feature_list)
    
    def get_correlation(self, subject_1, subject_2, feature):
        '''
        Return the pearson correlation between two sets of features
    
        Parameters
        ----------
        subject_1 : int
                    Subject number (1-11)
        subject_2 : int
                    Subject number (1-11)
        feature   : string
                    The name of the feature (column in dataframe)
        
        Returns
        -------
        correlation: (int, int)
                    (Pearson's correlation coefficient, 2-tailed p-value)   
        '''
        
        #Correct for length (pad with vector average)
        sub_1_feature_list = self.get_subject_feature_list(subject_1, feature)
        sub_2_feature_list = self.get_subject_feature_list(subject_2, feature)
        

        if len(sub_1_feature_list) < len(sub_2_feature_list):
            num = len(sub_2_feature_list) - len(sub_1_feature_list)
            sub_1_features = np.concatenate((sub_1_feature_list, np.ones(int(num)) * np.mean(sub_1_feature_list)))
            sub_2_features = sub_2_feature_list
        else:
            num = len(sub_1_feature_list) - len(sub_2_feature_list)
            sub_2_features = np.concatenate((sub_2_feature_list, np.ones(int(num)) * np.mean(sub_2_feature_list)))
            sub_1_features = sub_1_feature_list
    
        #Get Correlation
        return ss.pearsonr(sub_1_features, sub_2_features)[0]
    
    
    def get_monotonicity(self, subject, feature, delta=0):
        
        sub_feature_list = self.get_subject_feature_list(subject, feature)
        
        s_mo_plus = 0
        s_mo_min  = 0
        
        for i in range(len(sub_feature_list)-1):
            #Trend of the feature
            if np.mean(sub_feature_list[-11:-1]) > np.mean(sub_feature_list[0:10]):
                trend = 1
            else:
                trend = -1
            
            if trend > 0:
                if sub_feature_list[i+1] > sub_feature_list[i] - delta*sub_feature_list[i]:
                    s_mo_plus += 1
                else:
                    s_mo_min += 1
            else:
                if sub_feature_list[i+1] < sub_feature_list[i] + delta*sub_feature_list[i]:
                    s_mo_min += 1
                else:
                    s_mo_plus += 1
        
        MO = s_mo_plus/(len(sub_feature_list)-1) - s_mo_min/(len(sub_feature_list)-1)
        return MO
    
    #TODO
    def get_monotonicity_based_MannKendall(self, subject, feature, delta=0):
        
        sub_feature_list = self.get_subject_feature_list(subject, feature)
        
        s_mo_plus = 0
        s_mo_min = 0
        
        for i in range(len(sub_feature_list)-1):
            #Trend of the feature
            if np.mean(sub_feature_list[-11:-1]) > np.mean(sub_feature_list[0:10]):
                trend = 1
            else:
                trend = -1
            
            if trend > 0:
                #Get prob positive trend
                s_mo_plus += ss.norm(0,delta*sub_feature_list[i])\
                    .cdf(sub_feature_list[i+1] - sub_feature_list[i])-0.5
            else:
                #Get prob negative trend
                s_mo_min += 1 - ss.norm(0,delta*sub_feature_list[i])\
                    .cdf(sub_feature_list[i+1] - sub_feature_list[i])-0.5
                
        if trend > 0:
            MO = s_mo_plus/(len(sub_feature_list)-1)
        elif trend < 0:
            MO = -s_mo_min/(len(sub_feature_list)-1)
        else:
            return 0
        
        return MO
    
    #CORRECTED ERROR HERE
    def get_AMK(self, subject_1, subject_2, feature, delta = 0, omega=1):
        #ORIGINAL
        MO_sub1 = self.get_monotonicity(subject_1, feature, delta=delta)
        #ORIGINAL LINE: #MO_sub2 = self.get_monotonicity(subject_1, feature, delta=delta)
        MO_sub2 = self.get_monotonicity(subject_2, feature, delta=delta)
             
        Tr_12 = self.get_correlation(subject_1, subject_2, feature)
        
        AM_12 = omega*(MO_sub1 + MO_sub2)*Tr_12
        
        #Correlation only -> return Tr_12, original AM_12
        return AM_12
    
    #CORRECTED ERROR HERE
    def get_AMK_SRC(self, subject_1, subject_2, feature, delta = 0, omega=1):
        #ORIGINAL
        #MO_sub1 = self.get_monotonicity(subject_1, feature, delta=delta)
        #ORIGINAL LINE: #MO_sub2 = self.get_monotonicity(subject_1, feature, delta=delta)
        #MO_sub2 = self.get_monotonicity(subject_2, feature, delta=delta)
             
        Tr_12 = self.get_correlation_SRC(subject_1, subject_2, feature)
        
        AM_12 = omega*(1)*Tr_12
        
        #Correlation only -> return Tr_12, original AM_12
        return AM_12
       
    
    def  get_suitability(self, feature, delta):
        
        #omega = self.get_omega(feature)
        omega = self.get_omega_MK(feature)
        
        S_k = 0
        #How to choose delta??
        # - max 0.15
        # - can choose adaptively
        for i in range(len((self.shearers))-1):
            for j in range(i,len((self.shearers))):
                temp = self.get_AMK(i,j,feature,delta=delta, omega=omega)
                S_k += temp
        
        S_k = S_k / (len(self.shearers)*(len(self.shearers)-1))
        
        return S_k
    
    def  get_suitability_SRC(self, feature, delta):
        
        #omega = self.get_omega(feature)
        omega = self.get_omega_MK(feature)
        
        S_k = 0
        #How to choose delta??
        # - max 0.15
        # - can choose adaptively
        for i in range(len((self.shearers))-1):
            for j in range(i,len((self.shearers))):
                temp = self.get_AMK_SRC(i,j,feature,delta=delta, omega=omega)
                S_k += temp
        
        S_k = S_k / (len(self.shearers)*(len(self.shearers)-1))
        
        return S_k
    
   
    def get_omega(self,feature):
        omega = []
        
        for i in range(len(self.shearers)):
            sub_feature_list = self.get_subject_feature_list(i, feature)
            if np.mean(sub_feature_list[-11:-1]) > np.mean(sub_feature_list[0:10]):
                trend = 1
            else:
                trend = -1
            omega.append(trend)
            
        if sum(omega) >= 0:
            return 1
        else:
            return -1
        
    def get_omega_MK(self,feature):
        import pymannkendall as mk
        
        omega = []
        
        for i in range(len(self.shearers)):
            #do with the mann-kendall test
            sub_feature_list = self.get_subject_feature_list(i, feature)
            try:
                result = mk.original_test(sub_feature_list)
                if debug == 1: 
                    print('the trend is {}, with p = {}'\
                          .format(result.trend,result.p))
                if result.trend == 'increasing':
                    trend = 1
                elif result.trend == 'decreasing':
                    trend = -1
                else:
                    trend = 0
            except (ZeroDivisionError):
                trend = 0
            
            omega.append(trend)
            
        if debug == 1: print('sum of omega = {}'.format(sum(omega)))    
        if sum(omega) > 0:
            return 1
        elif sum(omega) == 0:
            return 0
        else:
            return -1
        
    def get_subject_feature_correlation(self,subject,target_feature,*args):
        #Lengths will all be the same
        if not args: raise ValueError(\
                    'Must have at least one feature to correlate against')
        if len(args) == 1: 
            target_data = self.get_subject_feature_list(subject,target_feature)
            corr_features = self.get_subject_feature_list(subject,args[0])
            return ss.pearsonr(target_data, corr_features)[0]**2
        else:
            #Multi correlation - create multi-regression of target feature
            #in terms of the features in args - then return r value
            
            #Build dataframes - one for target_feature
            #Get the features
            target_data = self.get_subject_feature_list(subject,target_feature)
            target_df = pd.DataFrame(target_data, columns = [target_feature])
            target_df = target_df - target_df.mean()
            #The rest
            corr_features = []
            for element in args:
                corr_features.append(self.get_subject_feature_list(subject,element))
                
            feature_tuple = tuple(corr_features)
            
            corr_df = pd.DataFrame(np.vstack(feature_tuple).T,columns = args)
            corr_df = corr_df - corr_df.mean()
            #OLS
            result = sm.OLS(target_df,corr_df).fit()
            if debug == 1: print(result.summary())
            
            
            return result.rsquared
            
        
    def get_feature_average_correlation(self,target_feature,*args):
        cum_total = 0
        for i in range(len(self.shearers)):
            cum_total += self.get_subject_feature_correlation(i, target_feature, *args)
        av_total = cum_total / len(self.shearers)
        
        return av_total
        
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
    
    
    def get_correlation_SRC(self, subject_1, subject_2, feature):
        '''
        Return the pearson correlation between two sets of features
    
        Parameters
        ----------
        subject_1 : int
                    Subject number (1-11)
        subject_2 : int
                    Subject number (1-11)
        feature   : string
                    The name of the feature (column in dataframe)
        
        Returns
        -------
        correlation: (int, int)
                    (Spearman's R correlation coefficient, 2-tailed p-value non parametric)   
        '''
        
        #Correct for length (pad with vector average)
        sub_1_feature_list = self.get_subject_feature_list(subject_1, feature)
        sub_2_feature_list = self.get_subject_feature_list(subject_2, feature)
        
        if len(sub_1_feature_list) < len(sub_2_feature_list):
            num = len(sub_2_feature_list) - len(sub_1_feature_list)
            sub_1_features = np.concatenate((sub_1_feature_list, np.ones(int(num)) * np.mean(sub_1_feature_list)))
            sub_2_features = sub_2_feature_list
        else:
            num = len(sub_1_feature_list) - len(sub_2_feature_list)
            sub_2_features = np.concatenate((sub_2_feature_list, np.ones(int(num)) * np.mean(sub_2_feature_list)))
            sub_1_features = sub_1_feature_list
    
        #Get Correlation
        print('using distcorr')
        #return ss.pearsonr(sub_1_features, sub_2_features)[0]
        return distcorr(sub_1_features, sub_2_features)
        
if __name__ == '__main__':
    import math
    
    #Harness no harness conditions
    harness = [(2,'thursday'),\
                        (5,'thursday'),(7,'monday'),(8,'tuesday'),\
                        (9,'wednesday')]
    
    no_harness = [(1,'tuesday'),(1,'wednesday'),(1,'thursday'),\
                        (3,'tuesday'),(4,'wednesday'),\
                        (6,'friday'),(10,'thursday'),(11,'friday')]
    
    #Shearers = none for all subjects
    A = AllShearersFeatures(dir_path='D:\\Data\\new_saved_features_11_subjects')

    #MOVED TO get_paper_results.py
    import get_feature_names as gfn
    feature_names = gfn.get_feature_names()
    
    test = ['emg_shannon_entropy_L1 Erector Spinae LEFT',\
 'emg_shannon_entropy_L3 Erector Spinae LEFT',\
 'emg_shannon_entropy_L5 Multifidus LEFT',\
 'emg_shannon_entropy_L1 Erector Spinae RIGHT',\
 'emg_shannon_entropy_L3 Erector Spinae RIGHT',\
 'emg_shannon_entropy_L5 Multifidus RIGHT',\
 'env_col_average_L1 Erector Spinae LEFT',\
 'env_col_average_L1 Erector Spinae RIGHT',\
 'env_col_average_L3 Erector Spinae LEFT',\
 'env_col_average_L3 Erector Spinae RIGHT',\
 'env_col_average_L5 Multifidus LEFT',\
 'env_col_average_L5 Multifidus RIGHT',\
 'emg_spectral_ind_ratio_L1 Erector Spinae LEFT_1div0',\
 'emg_spectral_ind_ratio_L1 Erector Spinae LEFT_-1div5',\
 'emg_spectral_ind_ratio_L3 Erector Spinae LEFT_1div0',\
 'emg_spectral_ind_ratio_L3 Erector Spinae LEFT_-1div5',\
 'emg_spectral_ind_ratio_L5 Multifidus LEFT_1div0',\
 'emg_spectral_ind_ratio_L5 Multifidus LEFT_-1div5']
#    
#    result_dict = {}
#    for element in feature_names:
#        try:
#            temp =  A.get_suitability(element)
#            if not math.isnan(temp):
#                result_dict[element] = temp
#        except (ValueError):
#            pass
#        
#    import operator
#    sorted_x = sorted(result_dict.items(), key=operator.itemgetter(1))
#    
#    
#    feature_name_list = [x[0] for x in sorted_x]
#    value_list = [x[1] for x in sorted_x]
#    
#    #import pandas as pd
#    
#    Q = pd.DataFrame(data=value_list, index = feature_name_list, columns = ['Value'])
#    Q.index.name = 'Feature Name'

    #Want to rebuild a dataframe (just do this as part of the plotting), 
    #using the get_datetime index, then we can 
    #easily plot every single feature using builtin functions
    

    
    
    