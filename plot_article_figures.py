# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:38:46 2024

Plot the example features from two subjects to appear in the paper

@author: mrobinson2
"""

try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass   

import feature_set as fs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from config import GBL_DEBUG
import matplotlib

def plot_figures(dirPath='new_saved_features_11_subjects'):
    '''
    
    
    '''
    
    params = {'backend': 'Agg',
               'axes.labelsize': 10,
               'font.size': 10 } # extend as needed
    matplotlib.rcParams.update(params)
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
        
    A = fs.AllShearersFeatures(shearers=None,dir_path=dirPath)
    
    #Need a feature list
    
    feature_list = ['emg_shannon_entropy_L3 Erector Spinae RIGHT',\
                    'env_quantile_05_L3 Erector Spinae RIGHT',\
                        'emg_shannon_entropy_L3 Erector Spinae LEFT',\
                    'env_quantile_05_L3 Erector Spinae LEFT',\
                        'mocap_quantile_05_jRightHip_z',\
                            'mocap_quantile_05_Pelvis_T8_z']
    
    
    #Need a subject list w/ harness vs. no-harness comparison
    subject_list_harness = [3,6,8,9,10]
    subject_list_no_harness = [0,1,2,4,5,7,11,12]
    
    #get feature list
    def get_feature(subject: int, feature: str):
        return A.get_subject_feature_list(subject,feature)
    
    #get feature time
    def get_feature_time(subject: int):
        return pd.to_datetime(A.get_subject_feature_list(subject,'emg_get_datetime'))
    
    def plot_feature(subject: int, feature: str):
        
        plt.figure()
        plt.plot(A.get_subject_feature_list(subject,feature),'x')
    
    initial_dataframe = pd.DataFrame()
    
    for subject in subject_list_harness:
        
        if subject in [3,8,9,10]:
            temp_datetime = get_feature_time(subject) - pd.Timedelta(minutes=30)
            subject_dataframe = pd.DataFrame(temp_datetime.time,columns=['time_of_day'])
        else:
            subject_dataframe = pd.DataFrame(get_feature_time(subject).time,columns=['time_of_day'])
        subject_dataframe['harness'] = [True for i in subject_dataframe.index]
        subject_dataframe['subject'] = [subject for i in subject_dataframe.index]
        for feature in feature_list:
            subject_dataframe[feature] = get_feature(subject,feature)
        initial_dataframe = initial_dataframe.append(subject_dataframe, ignore_index = True)
        
    for subject in subject_list_no_harness:
        
        if subject in [3,8,9,10]:
            temp_datetime = get_feature_time(subject) - pd.Timedelta(minutes=30)
            subject_dataframe = pd.DataFrame(temp_datetime.time,columns=['time_of_day'])
        else:
            subject_dataframe = pd.DataFrame(get_feature_time(subject).time,columns=['time_of_day'])
        subject_dataframe['harness'] = [False for i in subject_dataframe.index]
        subject_dataframe['subject'] = [subject for i in subject_dataframe.index]
        for feature in feature_list:
            subject_dataframe[feature] = get_feature(subject,feature)
        initial_dataframe = initial_dataframe.append(subject_dataframe, ignore_index = True)
    
    
    import datetime
    run_1_example = datetime.time(hour=7,minute=30)
    run_2_example = datetime.time(hour=10,minute=0)
    run_3_example = datetime.time(hour=15,minute=0)
    run_4_example = datetime.time(hour=15,minute=30)
    
    def get_run(test):
        if test < datetime.time(hour=9,minute=45):
            return 1
        elif (test > datetime.time(hour=9,minute=45)) & (test < datetime.time(hour=12,minute=15)):
            return 2
        elif (test > datetime.time(hour=12,minute=45)) & (test < datetime.time(hour=15,minute=15)):
            return 3
        elif (test > datetime.time(hour=15,minute=15)):
            return 4
        else:
            print(test)
            raise ValueError('Looks like there is an issue with the entered datetime')
    
    initial_dataframe['run'] = [get_run(i) for i in initial_dataframe['time_of_day']]
    initial_dataframe.columns = initial_dataframe.columns.str.replace(' ','_')
    initial_dataframe['ToD'] = initial_dataframe['run']
    initial_dataframe['subject_cat'] = initial_dataframe['subject'].astype('category')
     
    
    #Use LMM to evaluate the significance of feature set
    for feature in feature_list:
        if 'rms' in feature:
            temp = initial_dataframe[feature]
            temp = temp*len(temp)
            temp = np.square(temp)
            temp = temp/len(temp)
            temp = np.sqrt(temp)
            initial_dataframe[feature] = temp
       
        
        md = smf.mixedlm("{} ~ run + C(harness) + run*C(harness)".format(\
                         feature.replace(' ','_')),\
                         initial_dataframe,\
                         groups='subject',\
                         re_formula='~1',\
                         vc_formula={'run (category)' : '0 + C(run)'})
            
        nested_md = smf.mixedlm("{} ~ run".format(\
                     feature.replace(' ','_')),\
                     initial_dataframe,\
                     groups='subject',\
                     re_formula='~1',\
                     vc_formula={'run (category)' : '0 + C(run)'})
        
        
        mdf = md.fit()
        print(mdf.summary())
        
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
    
    
    
    # set figure size as a fraction of the columnwidth
    columnwidth = 245.71811
    #fig = plt.figure(figsize=get_figsize(columnwidth, wf=1.0, hf=0.3))
    # make the plot
     
    from matplotlib.ticker import FormatStrFormatter
    
    import matplotlib.font_manager as fm
    
    font = fm.FontProperties(
            family = 'Times New Roman', fname = 'C:\\Windows\\Fonts\\Times New Roman\\Times New Roman Regular.ttf')
    
    # graph_results = True
    # if graph_results:
    #     import seaborn as sns
    #     import matplotlib.pyplot as plt
        
    #     feature_list = ['emg_spectral_ind_ratio_L1 Erector Spinae RIGHT_1div0',\
    #                     'env_quantile_05_L1 Erector Spinae RIGHT',\
    #                     'mocap_quantile_05_Pelvis_T8_z',\
    #                 'emg_spectral_ind_ratio_L5 Multifidus RIGHT_1div0',\
    #                 'env_quantile_05_L5 Multifidus RIGHT',\
    #                     'mocap_rmsquare_Pelvis_T8_x'
    #                 ]
            
        
    # #Plot the subplots!
    # #2 x 2 with all sharing x
    fig, axes = plt.subplots(2,2,figsize=get_figsize(505.89, hf = 1, wf=1.0))
    
    def get_shearer_plot(subject_num):
        Q = initial_dataframe.loc[initial_dataframe['subject'] == subject_num]
        Q.rename({'subject':'Subject','run':'Run'}, axis='columns',inplace=True)
        Q['Sheep (Iteration)'] = np.arange(1,len(Q.index)+1)
        return Q
    
    
    plot_1_df = get_shearer_plot(1)
    plot_1_df = plot_1_df.loc[plot_1_df['env_quantile_05_L3_Erector_Spinae_RIGHT'] < 0.22]
    plot_2_df = get_shearer_plot(8)
    
    plot_1_df['Mean NEMG Per Sheep (%)'] = plot_1_df['env_quantile_05_L3_Erector_Spinae_RIGHT'] * 100
    plot_2_df['Mean NEMG Per Sheep (%)'] = plot_2_df['env_quantile_05_L3_Erector_Spinae_RIGHT'] * 100
    
    
    plot_1_df['Shannon Entropy (nats)'] = plot_1_df['emg_shannon_entropy_L3_Erector_Spinae_RIGHT']
    plot_2_df['Shannon Entropy (nats)'] = plot_2_df['emg_shannon_entropy_L3_Erector_Spinae_RIGHT']
    
    
    sns.scatterplot(data=plot_1_df,x='Sheep (Iteration)',\
                        y='Shannon Entropy (nats)',\
                      hue='Run',\
                  palette=sns.color_palette('colorblind',n_colors=4),\
                      ax=axes[1][0])
        
    
    sns.scatterplot(data=plot_2_df,x='Sheep (Iteration)',\
                        y='Shannon Entropy (nats)',\
                      hue='Run',\
                  palette=sns.color_palette('colorblind',n_colors=4),\
                      ax=axes[1][1])
        
    sns.scatterplot(data=plot_1_df,x='Sheep (Iteration)',\
                        y='Mean NEMG Per Sheep (%)',\
                      hue='Run',\
                  palette=sns.color_palette('colorblind',n_colors=4),\
                      ax=axes[0][0])
        
    
    sns.scatterplot(data=plot_2_df,x='Sheep (Iteration)',\
                        y='Mean NEMG Per Sheep (%)',\
                      hue='Run',\
                  palette=sns.color_palette('colorblind',n_colors=4),\
                      ax=axes[0][1])
    
    fig.tight_layout()
    #If want to save figures...    
    #fig.savefig("", format = 'pdf')
       
    
