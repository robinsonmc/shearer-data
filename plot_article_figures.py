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
from statsmodels.compat import lzip
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

    
#A = fs.AllShearersFeatures(shearers=None,dir_path='D:\Data\saved_features_with_time')
A = fs.AllShearersFeatures(shearers=None,dir_path='D:\\Data\\new_saved_features_11_subjects')

#Need a feature list
feature_list = [#'env_quantile_01_L3 Erector Spinae RIGHT',\
                #'env_quantile_025_L3 Erector Spinae RIGHT',\
                #'emg_shannon_entropy_L3 Erector Spinae RIGHT',\
                #'env_quantile_01_L5 Multifidus LEFT',\
                #'env_quantile_01_L1 Erector Spinae RIGHT',\
                'env_quantile_05_L1 Erector Spinae RIGHT',\
                'env_quantile_05_L3 Erector Spinae RIGHT',\
                'env_quantile_05_L5 Multifidus RIGHT',\
                #'env_quantile_01_L5 Multifidus RIGHT',\
                #'env_quantile_025_L5 Multifidus LEFT',\
                #'emg_shannon_entropy_L5 Multifidus LEFT',\
                #'emg_shannon_entropy_L5 Multifidus RIGHT',\
                'emg_spectral_ind_ratio_L1 Erector Spinae RIGHT_1div0',\
                #'emg_spectral_ind_ratio_L1 Erector Spinae LEFT_1div0',\
                #'mocap_angvel_rmsquare_Pelvis_T8_z',\
                #'emg_shannon_entropy_L1 Erector Spinae RIGHT',\
                'emg_spectral_ind_ratio_L3 Erector Spinae RIGHT_1div0',\
                #'emg_spectral_ind_ratio_L3 Erector Spinae LEFT_1div0',\
                'emg_spectral_ind_ratio_L5 Multifidus RIGHT_1div0',\
                #'emg_spectral_ind_ratio_L5 Multifidus LEFT_1div0',\
                'mocap_quantile_05_Pelvis_T8_z',\
                'mocap_rmsquare_Pelvis_T8_x',\
                'mocap_rmsquare_Pelvis_T8_y']
                #'env_quantile_05_L5 Multifidus RIGHT',\
                #'mocap_quantile_05_jLeftHip_z',\
                #'mocap_quantile_05_jRightHip_z']

#Delta = 0.15 final method
#feature_list = ['env_quantile_01_L3 Erector Spinae RIGHT',
#                     'env_quantile_01_L5 Multifidus LEFT',
#                     'emg_spectral_ind_ratio_L1 Erector Spinae RIGHT_1div0',
#                     'emg_shannon_entropy_L5 Multifidus RIGHT',
#                     'mocap_angvel_rmsquare_Pelvis_T8_z',
#                     'emg_spectral_ind_ratio_L3 Erector Spinae RIGHT_1div0',
#                     'env_quantile_01_Rectus Abdominis LEFT',
#                     'env_col_autocorr_Rectus Abdominis LEFT',
#                     'env_quantile_01_L1 Erector Spinae RIGHT',
#                     'mocap_col_autocorr_Pelvis_T8_z']

#Delta = 0.15 by relevance scores only
# feature_list = ['env_quantile_01_L3 Erector Spinae RIGHT',\
#                 'env_quantile_025_L3 Erector Spinae RIGHT',\
#                 'emg_shannon_entropy_L3 Erector Spinae RIGHT',\
#                 'env_quantile_01_L5 Multifidus LEFT',\
#                 'env_quantile_01_L1 Erector Spinae RIGHT',\
#                 'env_quantile_05_L3 Erector Spinae RIGHT',\
#                 'env_quantile_01_L5 Multifidus RIGHT',\
#                 'env_quantile_025_L5 Multifidus LEFT',\
#                 'emg_shannon_entropy_L5 Multifidus LEFT',\
#                 'emg_shannon_entropy_L5 Multifidus RIGHT']

#Delta = 0.1 by final scores TODO
feature_list = ['env_quantile_01_L3 Erector Spinae RIGHT']
              # 'emg_shannon_entropy_L5 Multifidus LEFT']
                #'emg_col_autocorr_L1 Erector Spinae RIGHT',\
                #'env_quantile_01_L5 Multifidus RIGHT',\
               # 'emg_spectral_ind_ratio_L3 Erector Spinae RIGHT_1div0',\
               # 'mocap_quantile_05_Pelvis_T8_z',\
               # 'emg_col_autocorr_Rectus Abdominis LEFT',\
            #    'env_quantile_01_L1 Erector Spinae RIGHT',\
           #     'emg_shannon_entropy_External Oblique RIGHT',\
      #          'mocap_col_autocorr_Pelvis_T8_z']    


#Delta = 0.1 by relevance scores only
feature_list = ['env_quantile_01_L3 Erector Spinae RIGHT',\
                'env_quantile_025_L3 Erector Spinae RIGHT',\
                'emg_shannon_entropy_L3 Erector Spinae RIGHT',\
                'env_quantile_01_L5 Multifidus LEFT',\
                'env_quantile_01_L1 Erector Spinae RIGHT',\
                'emg_shannon_entropy_L5 Multifidus LEFT',\
                'env_quantile_01_L5 Multifidus RIGHT',\
                'env_quantile_05_L3 Erector Spinae RIGHT',\
                'env_quantile_025_L5 Multifidus LEFT',\
                'emg_shannon_entropy_L5 Multifidus RIGHT']     

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

print("EXTRA SUBJECTS WOKRING?")
#initial_dataframe.to_csv('C:\\Users\\mrobinson2\\Documents\\gitub-repositories\\phd-data-analysis\\statistical_analysis\\flat_data.csv')
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
#initial_dataframe['combined_f'] = (initial_dataframe['mocap_quantile_05_jRightHip_z'] + initial_dataframe['mocap_quantile_05_jLeftHip_z'])/2 + initial_dataframe['mocap_quantile_05_Pelvis_T8_z'] 
#initial_dataframe.to_csv('C:\\Users\\mrobinson2\\Documents\\gitub-repositories\\phd-data-analysis\\statistical_analysis\\flat_data.csv')


initial_dataframe.columns = initial_dataframe.columns.str.replace(' ','_')
#initial_dataframe['run_ordered'] = initial_dataframe['run']
initial_dataframe['ToD'] = initial_dataframe['run']
initial_dataframe['subject_cat'] = initial_dataframe['subject'].astype('category')

# #calculate normalisation value
# value_for_norm = initial_dataframe.loc[(initial_dataframe['harness'] == False) & (initial_dataframe['run'] == 1)]['mocap_rmsquare_Pelvis_T8_y'].mean()
# initial_dataframe['mocap_rmsquare_Pelvis_T8_y'] = initial_dataframe['mocap_rmsquare_Pelvis_T8_y'] /value_for_norm

# value_for_norm_x = initial_dataframe.loc[(initial_dataframe['harness'] == False) & (initial_dataframe['run'] == 1)]['mocap_rmsquare_Pelvis_T8_x'].mean()
# initial_dataframe['mocap_rmsquare_Pelvis_T8_x'] = initial_dataframe['mocap_rmsquare_Pelvis_T8_x'] /value_for_norm_x



    



import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.tools.sm_exceptions import ConvergenceWarning

#feature_list.append('combined_f')

for feature in feature_list:
    if 'rms' in feature:
        temp = initial_dataframe[feature]
        temp = temp*len(temp)
        temp = np.square(temp)
        temp = temp/len(temp)
        temp = np.sqrt(temp)
        initial_dataframe[feature] = temp
    #initial_dataframe = initial_dataframe[(np.abs(stats.zscore(initial_dataframe[feature.replace(' ','_')])) < 3)]
    #initial_dataframe[feature.replace(' ','_')] = np.sqrt(initial_dataframe[feature.replace(' ','_')])
    
    #md = smf.mixedlm("{} ~ run + harness + run*harness".format(\
    #                 feature.replace(' ','_')),\
    #                 initial_dataframe,\
    #                 groups=initial_dataframe['subject'],\
    #                 re_formula='~1')
    
    #mdf = md.fit()
    #print(mdf.summary())
    
    #mdq = smf.mixedlm("{} ~ C(run) + C(harness) + C(run)*C(harness)".format(\
    #                 feature.replace(' ','_')),\
    #                 initial_dataframe,\
    #                 groups=initial_dataframe['subject'],\
    #                 re_formula='~1 + C(run)')
                     #vc_formula={'run' : '0 + C(run)'})
    
    #mdqf = mdq.fit()
    #print(mdqf.summary())
    
   
    
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
    
    ### NOT CORRECT
    #free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(8),
    #                                                                 np.eye(4))
    
    #mdqf = mdq.fit(free=free)
    #print(mdqf.summary())
    
    
    #free = sm.regression.mixed_linear_model.MixedLMParams.from_components(np.ones(8),
     #                                                                np.array([[1, 0, 0, 0],\
     #                                                                         [0, 1, 1, 1],\
     #                                                                         [0, 1, 1, 1,],\
     #                                                                         [0, 1, 1, 1]]))
    
    #mdqf = mdq.fit(free=free,method=['lbfgs'])
    #print(mdqf.summary())


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

test_normality = False
if test_normality:  
    fig = plt.figure(figsize = (16, 9))
    
    ax = sns.distplot(mdf.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    
    ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
    ax.set_xlabel("Residuals")
    
    ## Q-Q PLot
    
    fig = plt.figure(figsize = (16, 9))
    ax = fig.add_subplot(111)
    
    sm.qqplot(mdf.resid, dist = stats.norm, line = 's', ax = ax)
    
    ax.set_title("Q-Q Plot")
    
    name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
    test = sms.jarque_bera(mdf.resid)
    lzip(name, test)

import matplotlib
params = {'backend': 'Agg',
           'axes.labelsize': 10,
           'font.size': 10 } # extend as needed
matplotlib.rcParams.update(params)
plt.rcParams["font.family"] = "Times New Roman"
hfont = {'fontname':'Times New Roman Regular'}

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


# #Calc num_in_run and add column then select on this column using location



plot_1_df = get_shearer_plot(1)
plot_1_df = plot_1_df.loc[plot_1_df['env_quantile_05_L3_Erector_Spinae_RIGHT'] < 0.22]
plot_2_df = get_shearer_plot(8)

plot_1_df['Mean NEMG Per Sheep (%)'] = plot_1_df['env_quantile_05_L3_Erector_Spinae_RIGHT'] * 100
plot_2_df['Mean NEMG Per Sheep (%)'] = plot_2_df['env_quantile_05_L3_Erector_Spinae_RIGHT'] * 100


plot_1_df['Shannon Entropy (nats)'] = plot_1_df['emg_shannon_entropy_L3_Erector_Spinae_RIGHT']
plot_2_df['Shannon Entropy (nats)'] = plot_2_df['emg_shannon_entropy_L3_Erector_Spinae_RIGHT']


# plot_1_df = get_shearer_plot(11)
# plot_1_df = plot_1_df.loc[plot_1_df['env_quantile_05_L3_Erector_Spinae_RIGHT'] < 0.22]
# plot_2_df = get_shearer_plot(12)

# plot_1_df['Mean NEMG Per Sheep (%)'] = plot_1_df['env_quantile_05_L3_Erector_Spinae_RIGHT'] * 100
# plot_2_df['Mean NEMG Per Sheep (%)'] = plot_2_df['env_quantile_05_L3_Erector_Spinae_RIGHT'] * 100


# plot_1_df['Shannon Entropy (nats)'] = plot_1_df['emg_shannon_entropy_L3_Erector_Spinae_RIGHT']
# plot_2_df['Shannon Entropy (nats)'] = plot_2_df['emg_shannon_entropy_L3_Erector_Spinae_RIGHT']
    
    

# test_1_res = list(plot_1_df.groupby('Run', as_index=False).nth((0,1,2))['Sheep (Iteration)'].values)
# test_2_res = list(plot_2_df.groupby('Run', as_index=False).nth((0,1,2))['Sheep (Iteration)'].values)

# def test_1(sheep_num):
#     if sheep_num in test_1_res:
#         return 0
#     else:
#         return 1
    
# def test_2(sheep_num):
#     if sheep_num in test_2_res:
#         return 0
#     else:
#         return 1
    
# plot_1_df['to_plot'] = plot_1_df['Sheep (Iteration)'].apply(test_1) 
# plot_2_df['to_plot'] = plot_2_df['Sheep (Iteration)'].apply(test_2) 

# plot_1_df = plot_1_df.loc[plot_1_df['to_plot'] == 1] 
# plot_2_df = plot_2_df.loc[plot_2_df['to_plot'] == 1] 
# # plot_1_df['
# axes[0].set_ylim(ymin=0,ymax=90)
# #axes[1].set_ylim(ymin=0,ymax=130)

# #'mocap_quantile_05_jRightHip_z',\
# #                        'mocap_quantile_05_Pelvis_T8_z']

# sns.scatterplot(data=plot_1_df,x='Sheep (Iteration)',\
#                     y='mocap_quantile_05_Pelvis_T8_z',
#                     palette=sns.color_palette('Blues',n_colors=4),\
#                   hue='Run', ci=95,\
#                   ax=axes[0])
    
# sns.scatterplot(data=plot_1_df,x='Sheep (Iteration)',\
#                     y='mocap_quantile_05_jRightHip_z',\
#                         palette=sns.color_palette('Oranges',n_colors=4),\
#                   hue='Run', ci=95,\
#                   ax=axes[0])
    
# axes[0].legend(labels = ['Lumbar Flexion','Hip Flexion'])

# sns.scatterplot(data=plot_2_df,x='Sheep (Iteration)',\
#                     y='mocap_quantile_05_Pelvis_T8_z',\
#                         palette=sns.color_palette('Blues',n_colors=3),\
#                   hue='Run', ci=95,\
#                   ax=axes[1])
    
# sns.scatterplot(data=plot_2_df,x='Sheep (Iteration)',\
#                     y='mocap_quantile_05_jRightHip_z',\
#                          palette=sns.color_palette('Oranges',n_colors=3),\
#                   hue='Run', ci=95,\
#                   ax=axes[1])
    
# axes[1].legend(labels = ['Lumbar Flexion','Hip Flexion'])


                
# import matplotlib.lines as mlines
# from matplotlib.collections import PatchCollection

# blue_star = mlines.Line2D([], [], color=sns.color_palette('Blues',n_colors=4)[2], marker='o', linestyle='None',
#                           markersize=5, label='Lumbar Flexion')
# red_square = mlines.Line2D([], [], color=sns.color_palette('Oranges',n_colors=4)[2], marker='o', linestyle='None',
#                           markersize=5, label='Hip Flexion')

# #axes[0].legend(handles=[blue_star, red_square],numpoints=4)
# #axes[1].legend(handles=[blue_star, red_square],numpoints=3)


# # define an object that will be used by the legend
# class MulticolorPatch(object):
#     def __init__(self, colors):
#         self.colors = colors
        
# # define a handler for the MulticolorPatch object
# class MulticolorPatchHandler(object):
#     def legend_artist(self, legend, orig_handle, fontsize, handlebox):
#         width, height = handlebox.width, handlebox.height
#         patches = []
#         for i, c in enumerate(orig_handle.colors):
#             patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
#                                           -handlebox.ydescent],
#                            width / len(orig_handle.colors),
#                            height, 
#                            facecolor=c, 
#                            edgecolor='none'))

#         patch = PatchCollection(patches,match_original=True)

#         handlebox.add_artist(patch)
#         return patch


# # ------ choose some colors
# colors1 = sns.color_palette('Blues',n_colors=4)
# colors2 = sns.color_palette('Oranges',n_colors=4)

# # ------ get the legend-entries that are already attached to the axis
# h, l = axes[0].get_legend_handles_labels()
# h,l = ([],[])


# # ------ append the multicolor legend patches
# h.append(MulticolorPatch(colors1))
# l.append("Lumbar flexion across runs 1-4")

# h.append(MulticolorPatch(colors2))
# l.append("Hip flexion across runs 1-4")

# # ------ create the legend
# axes[0].legend(h, l, loc='lower left', 
#          handler_map={MulticolorPatch: MulticolorPatchHandler()})


# # ------ choose some colors
# colors1 = sns.color_palette('Blues',n_colors=3)
# colors2 = sns.color_palette('Oranges',n_colors=3)

# # ------ get the legend-entries that are already attached to the axis
# h, l = axes[1].get_legend_handles_labels()
# h,l = ([],[])


# # ------ append the multicolor legend patches
# h.append(MulticolorPatch(colors1))
# l.append("Lumbar flexion across runs 1-3")

# h.append(MulticolorPatch(colors2))
# l.append("Hip flexion across runs 1-3")

# # ------ create the legend
# axes[1].legend(h, l, loc='best', 
#          handler_map={MulticolorPatch: MulticolorPatchHandler()})

#plt.legend(title='Smoker', loc='upper left', labels=['Hell Yeh', 'Nah Bruh'])

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

    
    # for i in range(len(feature_list)):
    # #g = sns.FacetGrid(initial_dataframe,hue='harness')
    # #g.map_dataframe(sns.pointplot, x = 'run', y = 'emg_spectral_ind_ratio_L5 Multifidus RIGHT_1div0'.replace(' ','_'),estimator=np.mean,ci='sd',join=False)
    # #g.map_dataframe(sns.regplot, x = 'run', y = 'emg_spectral_ind_ratio_L5 Multifidus RIGHT_1div0'.replace(' ','_'),scatter=False)
    
    # #g = sns.lmplot(x='run', y='emg_spectral_ind_ratio_L5 Multifidus RIGHT_1div0'.replace(' ','_'), data=initial_dataframe,\
    #                 #  hue='harness',x_estimator=np.mean,x_ci='sd', fit_reg=True, ci=None);
    #     g = sns.regplot(x='run', y=feature_list[i].replace(' ','_'), scatter_kws={'s':12}, data=initial_dataframe.loc[initial_dataframe['harness'] == True],\
    #                       x_estimator=np.mean,x_ci='sd', fit_reg=True, ci=None,label='Harness',ax=axes[int(np.floor(i/3)),i%3]);
            
    #     g = sns.regplot(x='run', y=feature_list[i].replace(' ','_'), scatter_kws={'s':12}, data=initial_dataframe.loc[initial_dataframe['harness'] == False],\
    #                       x_estimator=np.mean,x_ci='sd', fit_reg=True, ci=None,label='No harness',ax=axes[int(np.floor(i/3)),i%3]);
            
    #     plt.setp(g.lines, linewidth=1, alpha=.75) 
    #     plt.setp(g.collections, alpha=.75) 
        
    #     y_labels = ['L1 ES $\mu$F (Hz)','L1 ES $\mu$A (N)', 'Lumbar flex (\N{DEGREE SIGN})','L5 MF $\mu$F (Hz)','L5 MF $\mu$A (N)','Lumbar twist (\N{DEGREE SIGN})']
    #     titles = ['(A)','(D)','(G)','(C)','(F)','(H)']
        
    #     plt.rcParams.update({'axes.titlesize': 'large'})
    #     #g.legend(loc="best",fontsize='x-large')
        
    #     if i == 0 or i == 3:
    #         pass
    #         g.set_ylim([50,163])
    #         g.set_yticks([60,80,100,120,140,160])
    #     elif i == 1 or i == 4:
    #         g.set_ylim([0,0.15])
    #         g.set_yticks([0.05,0.1,0.15])
    #     else:
    #         pass
    #         #g.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #         #g.set_yticks([2.5,5,7.5])
    #         #g.set_ylim(2.5,9)
            
    #     if i == 0: g.legend(loc="best")
    #     g.set_xlabel('',fontsize=10)
    #     g.set_ylabel('temp',fontsize=10)
    #     #Numbers
    #     g.tick_params(labelsize=10)
    #     #Legend size?
    #     g.set_xlim(0.9,4.1)
    #     g.set_xticks([1,2,3,4])
    #     if i > 2: 
    #         g.set(
    #         title=titles[i],
    #         xlabel = 'Run',
    #         ylabel = y_labels[i])
    #     else:
    #         g.set(
    #         title=titles[i],
    #         ylabel = y_labels[i])
    #     plt.tight_layout()
 
    
#fig.savefig("C:\\Users\\robin\\Documents\\lumbar_flex_angles.pdf", format = 'pdf')
       
    
