# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:42:38 2020

@author: mrobinson2
"""

'''contains the get_feature() functions'''

import pandas as pd
import STFT as stft
import mean_freq_calc as mfc
import numpy as np
import smoothness.SPARC as SP
import quaternion
import scipy.stats as ss
import TEST_MFC as tm
from config import GBL_DEBUG

#Rename a function - set funciton.__name__ to custom
#This is needed because the apply function creates a function
#to be passed to get_feature and these should be uniquely named
def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

#Create a function to apply a column function to a dataframe
#i.e. "pre-fill" one of the functions arguments so that only arg is df
def apply_to_column(function,col_name):
    @rename(function.__name__+'_'+col_name)
    def new_f(df):
        return function(df,col_name)
    return new_f


def apply_to_two_columns(function,column_A, column_B):
    @rename(function.__name__+'_' + column_A + '_' + column_B)
    def new_f(df):
        return function(df, column_A, column_B)
    return new_f

def apply_twocol_twodf(function, column_A, column_B):
    @rename(function.__name__+'_' + column_A + '_' + column_B)
    def new_f(df1,df2):
            return function(df1, df2, column_A, column_B)
    return new_f

def apply_ApEn(function, column_A, column_B, column_C):
    @rename(function.__name__+'_' + column_A + '_' + column_B + '_' + column_C)
    def new_f(df):
        norm_3col = (df[column_A]**2 + df[column_B]**2 + df[column_C]**2)**(1/2)
        #U is norm_3col, m is 2 (length), r = 1 (radius of n neighbors)
        return function(norm_3col, 2, 0.5)
    return new_f

def apply_quantile(col_name, *args):
    @rename(pd.Series.quantile.__name__+'_'+col_name)
    def new_f(df):
        return df[col_name].quantile(*args)
    return new_f

def apply_DRP(function,column_A, column_B, section):
    @rename(function.__name__+'_'+str(section)+'_'+ column_A + '_' + column_B)
    def new_f(df):
        return function(df,column_A,column_B,section)
    return new_f

def apply_spec_ind(function, col_name, index1, index2):
    @rename(function.__name__+'_'+col_name+'_'+str(index1)+'div'+str(index2))
    def new_f(df):
        return function(df,col_name,index1,index2)
    return new_f


def apply_to_rel_vec(summary_func, parent, child, vector, joint_name, xyz):
    '''
    Parameters
    ----------
    summary_func : function 
                   function to apply to column - e.g. col_average / rmsquare
    parent       : string
                   parent segment name
    child        : string
                   child segment name
    vector       : string
                   encoded name of kinematic vector (e.g. angacc, angvel)
    joint_name   : string
                   joint name of parent-child relationship (e.g. jLeftElbow)
    xyz          : string ('x','y','z')
                   x, y, or z component
    '''
    @rename(vector+'_'+summary_func.__name__+'_'+joint_name+'_'+xyz)
    def new_f(df):
        #Add functionality by replacing relative_vec_body with function
        #with the same params if applicable
        data = relative_vec_body(df, parent, child, vector, joint_name)
        col_name = joint_name+'_'+vector+'_'+xyz
        if col_name not in data.columns:
            raise ValueError('joint name not found...')
        return summary_func(data,col_name)
    return new_f

def apply_SPARC(parent, child, joint_name, xyz):
    @rename(SP.sparc.__name__+'_angvel_'+joint_name)
    def new_f(df):
        mov_prof =  relative_vec_body(df, parent, child, '', joint_name)
        mov_prof.fillna(method='bfill', axis = 0, inplace=True)
        col_name = joint_name+'_'+xyz
        movement = mov_prof[col_name].astype(float)
        movement = movement.fillna(method='backfill')
        
        return SP.sparc(np.array(movement), 60)[0]
    return new_f

def apply_dimjerk2(parent, child, joint_name, xyz):
    @rename(SP.dimensionless_jerk2.__name__+'_angvel_'+joint_name)
    def new_f(df):
        mov_prof =  relative_vec_body(df, parent, child, 'angacc', joint_name)
        col_name = joint_name+'_angacc_'+xyz
        return SP.dimensionless_jerk2(np.array(mov_prof[col_name]), 60,\
                                      data_type='accl')
    return new_f
        

#--------------------------------------------------------------------------
# Define your functions below this point
#--------------------------------------------------------------------------

def col_average(df, column):
    return df[column].mean()

def col_std(df, column):
    return df[column].std()

def col_max(df,column):
    return df[column].max()

def col_var(df,column):
    return df[column].var()

def time_taken_s(df):
    temp = df['time_ms'].iloc[-1] - df['time_ms'].iloc[0]
    result = temp/1000
    return result

def get_time(df):
    result = df['time_ms'].iloc[0]/1000
    return result

def get_datetime(df):
    if GBL_DEBUG == 1: print('Dataframe shape is: {}'.format(df.shape))
    result = df.index[0] 
    return result

#TODO: just return the average mean freq from the dataframe,
#this should include a column to run on (and only run on 1)
#Also get rid of the RMS not very useful for current application
def mean_freq_dataframe(df, col_name):
        
    df_mean_freq = pd.DataFrame()
    #df_RMS = pd.DataFrame()
    
    
    #Add the mean frequency calc muscle by muscle
    f, t, spec = stft.shearer_spectogram(df, col_name)
    temp_df_mean_freq, temp_df_RMS =\
                                mfc.muscle_mean_freq(f, t, spec)
    df_mean_freq[col_name] = temp_df_mean_freq['mean_f']
    df_mean_freq['time'] = temp_df_mean_freq['time']
    #df_RMS[muscle] = temp_df_RMS['RMS']
    #df_RMS['time'] = temp_df_RMS['time']
        
    #return (df_mean_freq[:-1],df_RMS[:-1])
    return df_mean_freq[col_name].mean()

def spectral_ind_ratio(df, col_name, index1, index2):

    #Add the mean frequency calc muscle by muscle
    #f, t, spec = stft.shearer_spectogram(df, col_name)
    psd, f = tm.psd(df[col_name])
    ratio = tm.muscle_spec_moment_ratio_from_psd(f,psd,index1,index2)
    
    #ratio = mfc.muscle_spec_moment_ratio(f,t,spec,index1,index2)
    
    #result = ratio['spec_moment_p']

    return ratio #result.mean()

def rmsquare(df,column):
    sq_all = df[column]**2
    ss_all = sq_all.sum()
    return (ss_all**(1/2))/len(df)

def two_col_ratio(df, column_A, column_B):
    df.fillna(method='bfill',axis=0,inplace=True)
    division = df[column_A].div(df[column_B])
    return division.mean()

def two_col_norm(df, column_A, column_B):
    sq_all = (df[column_A]-df[column_A].iloc[0])**2 + \
                                (df[column_B]-df[column_B].iloc[0])**2
    ss_all = sq_all.sum()
    norm = np.sqrt(ss_all)
    return norm/len(df)

def two_df_ratio(df1 ,df2, column_A, column_B):
    assert(len(df1) == len(df2))
    df1.fillna(method='bfill',axis=0,inplace=True)
    df2.fillna(method='bfill',axis=0,inplace=True)
    
    try:
        division = df1[column_A].div(df2[column_B])
    except Exception as e:
          print('Oops!', e.__class__, 'ocurred')
          division.loc[:,:] = -1
    return division.mean()

def two_df_mult(df1 ,df2, column_A, column_B):
    df1.fillna(method='bfill',inplace=True, axis=0)
    df2.fillna(method='bfill',inplace=True, axis=0)
    
    dfA = df1.resample('50ms').bfill()
    dfB = df2.resample('50ms').bfill()
    
    del df1, df2
    
    #if dfA.index[0] != dfB.index[0]:
        #raise ValueError('Inital time does not match')
    #    pass
    lengthA = len(dfA)
    lengthB = len(dfB)
    min_len = min(lengthA,lengthB)
    
    
    dfA = dfA.iloc[0:min_len]
    dfB = dfB.iloc[0:min_len]
    
            
    assert(len(dfA) == len(dfB))
    
    
    col_A = np.array(dfA[column_A])
    col_B = np.array(dfB[column_B])
    
    multiple = np.multiply(col_A,col_B)
    
    #multiply = dfA[column_A].multiply(dfB[column_B])
    if len(dfA) == 0 & len(dfB) == 0:
        return -1
    else:
        return np.mean(multiple)

def elem_mult_sum(df, column_A, column_B):
    multiply = df[column_A].multiply(df[column_B])
    return multiply.mean()


def ApEn(U, m, r):
    

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

# Usage example
#U = np.array([85, 80, 89] * 17)
    
def sample_entropy(df, col_name):
    from sampen import sampen2
    
    sampen_result = sampen2(df[col_name])
    return sampen_result[1]

def quantile_01(df,col_name):
    return df[col_name].quantile(q=0.1)

def quantile_025(df,col_name):
    return df[col_name].quantile(q=0.25)

def quantile_05(df,col_name):
    return df[col_name].quantile(q=0.5)

def quantile_075(df,col_name):
    return df[col_name].quantile(q=0.75)

def quantile_09(df,col_name):
    return df[col_name].quantile(q=0.9)

def DRP(df, column_A, column_B, section):
    length = len(df)
    max_index = int(np.floor(section/3*length))
    min_index = int(np.floor((section-1)/3*length))
    
    col_A = df[column_A].iloc[min_index:max_index]
    col_B = df[column_B].iloc[min_index:max_index]
    max_A = col_A.idxmax(axis=0)
    max_B = col_B.idxmax(axis=0)
    delta = max_A-max_B
    full_time = (df.iloc[-1]['time_ms'] - df.iloc[0]['time_ms'])/1000
    return (delta.total_seconds()/(full_time/3))*100


def shannon_entropy(df, col_name):
    values = df[col_name].value_counts()
    ent = ss.entropy(values)
    return ent

def col_autocorr(df,col_name):
    return df[col_name].autocorr()

def col_skew(df,col_name):
    return df[col_name].skew()

def col_kurtosis(df,col_name):
    return df[col_name].kurtosis()
    
    
#--------------------------------------------------------------------------
# Helper functions for transformations
#--------------------------------------------------------------------------        

def get_vector(dataframe, vector, joint_seg):
    '''
    Get a kinematic vector from the dataframe for a given joint or segment

    Parameters
    ----------
    dataframe : pd.DataFrame
                The dataframe containing the exported xsens data
    vector    : string
                The vector that should be obtained, e.g. vel, acc, angvel
    joint_seg : string
               The joint or segment that the vectors should be taken from
    
    Returns
    -------
    res_df    : pd.DataFrame
               The vector for each point in the dataframe (relevant columns)    
    '''
    
    col_name_base = joint_seg+'_'+vector
    x             = col_name_base+'_x'
    y             = col_name_base+'_y'
    z             = col_name_base+'_z'
    
    res_df = dataframe[[x,y,z]].copy()
    
    return res_df

def get_quaternion(dataframe, segment):
    '''
    Get the orientation quaternion components from the exported xsens data

    Parameters
    ----------
    dataframe : pd.DataFrame
                The dataframe containing the exported xsens data
    segment   : string
                The segment that the orientation should be taken from
    
    Returns
    -------
    res_df    : pd.DataFrame
                The quaternion params (relevant columns)    
    '''
    col_name_base = segment+'_orient'
    q1            = col_name_base+'_q1'
    q2            = col_name_base+'_q2'
    q3            = col_name_base+'_q3'
    q4            = col_name_base+'_q4'
    
    return dataframe[[q1,q2,q3,q4]].copy()

def relative_vec_body(dataframe, parent, child, vector, joint_name):
    '''
    Get the orientation quaternion components from the exported xsens data

    Parameters
    ----------
    dataframe : pd.DataFrame
                The dataframe containing the exported xsens data
    parent    : string
                The parent segment (e.g. LeftUpperArm for elbow)
    child     : string
                The child segment
    vector    : string
                The vector that should be obtained, e.g. vel, acc, angvel
    joint_name: string
                Column name to return under e.g. LeftElbow_vector_x
    
    Returns
    -------
    res_df    : pd.DataFrame
                The vector in the parent centred frame (relevant columns)    
    '''
    
    parent_name_base = parent+'_'+vector
    xp                = parent_name_base+'_x'
    yp                = parent_name_base+'_y'
    zp                = parent_name_base+'_z'
    
    child_name_base = child+'_'+vector
    xc                = child_name_base+'_x'
    yc                = child_name_base+'_y'
    zc                = child_name_base+'_z'
    
    parent_name_base = parent+'_orient'
    q1  = parent_name_base+'_q1'
    q2  = parent_name_base+'_q2'
    q3  = parent_name_base+'_q3'
    q4  = parent_name_base+'_q4'
    
    
    p_vec_0 = get_vector(dataframe, vector, parent)
    c_vec_0 = get_vector(dataframe, vector, child)
    
    
    rel_vec_0 = pd.DataFrame(columns=['x','y','z'])
    rel_vec_0['x'] = c_vec_0[xc] - p_vec_0[xp]
    rel_vec_0['y'] = c_vec_0[yc] - p_vec_0[yp]
    rel_vec_0['z'] = c_vec_0[zc] - p_vec_0[zp]
    
    del p_vec_0
    del c_vec_0
    
    p_frame = get_quaternion(dataframe, parent)
    
    quat = pd.DataFrame()
    quat['rotation'] =  p_frame.apply(lambda row: \
                    quaternion.quaternion(row[q1],row[q2], row[q3], row[q4]),\
                          axis=1)
                          
    del p_frame
    
    quat['relvec'] = rel_vec_0.apply(lambda row: \
                    quaternion.quaternion(0,row.x, row.y, row.z),\
                          axis=1)
    
    del rel_vec_0
    
    quat['result'] = quat.apply(lambda row: \
                        row['rotation']*row['relvec']*row['rotation'].inverse()\
                        ,axis=1)
    
    x_name = joint_name+'_'+vector+'_x'
    y_name = joint_name+'_'+vector+'_y'
    z_name = joint_name+'_'+vector+'_z'
    
    quat[x_name] = quat.apply(lambda row: \
                    row['result'].x, axis=1)
    
    quat[y_name] = quat.apply(lambda row: \
                    row['result'].y, axis=1)
    
    quat[z_name] = quat.apply(lambda row: \
                    row['result'].z, axis=1)
    
    quat.fillna(method='bfill')
    
    return quat[[x_name,y_name,z_name]]
    

#--------------------------------------------------------------------------
# Define the lookup table for the function units
#--------------------------------------------------------------------------
def get_yaxis_label(col_name):
   
    lookup = {'index_to_merge_on':'Sheep (N)',\
              'mocap_col_average':'Angle (deg)',\
              'mocap_col_std':'Angle (deg)',\
              'mocap_time_taken':'Seconds (s)',\
              'env_col_average':'Magnitude (N)',\
              'mean_freq_dataframe': 'Mean Frequency (Hz)',\
              'env_col_std':'Magnitude (N)',\
              'mocap_two_col_ratio': 'Ratio',\
              'env_quantile_01' : 'Magnitude (NEMG)',\
              'env_quantile_025' : 'Magnitude (NEMG)',\
              'env_quantile_05' : 'Magnitude (NEMG)',\
              'env_quantile_075' : 'Magnitude (NEMG)'}
    
    for k in lookup.keys():
        if k in col_name:
            return lookup[k]
        
    return None
        