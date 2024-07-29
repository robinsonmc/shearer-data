# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:53:22 2019

File to read the Xsens xml data and return a pandas dataframe containing the
data and the starting datetime contained in a tuple.

@author: mrobinson2
"""

from lxml import etree
import pandas as pd
from io import StringIO
import xml.etree.ElementTree as ET

#DEBUG FLAG
debug = 1

try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass 

def read_xsens_xml(file_path):
    '''
    read_xsens(file_path)
    
    Input: XSens mvnx file-type
    Output: Tuple containing Pandas dataframe, and initial datetime
    
    BUG: If the xsens mvnx file has a comment about the file 
    (user entered at the finish of recording) the parsing will fail
    - don't know why
    
    '''
    try:
        parser = etree.XMLParser()
        tree = etree.parse(file_path, parser)
        
    except:
        print("in exception block! for reading tree")
        tree = ET.parse(file_path)
        
    finally:
        root = tree.getroot()
    
    version = root[0].attrib['version']
    subject = root[2]
    
    if version == '2022.0.0':
        joints = subject[3]
        ergojoints = subject[4]
        footcontact_defn = subject[5]
        sensors = subject[2]
    else:       
        joints = subject[2]
        ergojoints = subject[3]
    
    datetime_start = pd.Timestamp(subject.attrib['recDate'])
    
    #Store all of the joints (x,y,z) in order
    joint_labels = []
    for child in joints:
        current_joint = child.attrib['label']
        if debug == 1: print('Current Joint: {}'.format(current_joint))
        joint_labels.append(current_joint + '_x')
        joint_labels.append(current_joint + '_y')
        joint_labels.append(current_joint + '_z')

    #Store the ergo joints (x,y,z) in order
    ergojoint_labels = []
    for child in ergojoints:
        current_joint = child.attrib['label']
        if debug == 1: print('Current Ergo Joint: {}'.format(current_joint))
        ergojoint_labels.append(current_joint + '_x')
        ergojoint_labels.append(current_joint + '_y')
        ergojoint_labels.append(current_joint + '_z')
    
    #DataFrame columns
    if version == '2022.0.0':
        xsens_dataframe_columns = list(['time_ms'] + joint_labels + ergojoint_labels + ['CoG_1', 'CoG_2', 'CoG_3' ,'CoG_4', 'CoG_5', 'CoG_6','CoG_7', 'CoG_8', 'CoG_9' ] + ['timecode'])
    
    else:
        xsens_dataframe_columns = (['time_ms'] + joint_labels + ergojoint_labels
        + ['CoG_x', 'CoG_y', 'CoG_z'] + ['timecode'])
    if debug == 1: print('number of columns in the xml files is: {}'\
              .format(len(xsens_dataframe_columns)))
    
    #Need to get the frames
    if version == '2022.0.0':
        frames = subject[6]
    else:
        frames = subject[4]
    
    #Create dictionary key = index, value = time + list of joint angles
    #+ ergo joint angles + cog location
    xsens_data = {}
    
    
    
    #Index within the frames of the joint angles and the ergonomic joint angles
    if version == '2022.0.0':
        JA_index = 10
        EJA_index = 12
    else:
        JA_index = 2
        EJA_index = 4
    #THIS CHANGED IN NEW VERSION to 6 (orig 5??)
    #NEED TO SEARCH FOR THIS BY ITSELF
    #TODO
    if version == '2019.0.0':
        CoG_index = 5
    else:
        CoG_index = 6
        
    if version == '2022.0.0':
        CoG_index = 14
    
    #Only parse normal frames (type normal)
    count = 0;
    for child in frames:
        if child.attrib['type'] == 'normal':
            index = (pd.Timestamp(float(child.attrib['ms']),
                                     unit='ms',tz='Australia/Melbourne'))
            time = float(child.attrib['time'])
            all_joint_data_str = child[JA_index].text.split()
            all_joint_data = [float(i) for i in all_joint_data_str]
            del all_joint_data_str
            
            all_ergo_data_str = child[EJA_index].text.split()
            all_ergo_data = [float(i) for i in all_ergo_data_str]
            del all_ergo_data_str
            
            all_cog_data_str = child[CoG_index].text.split()
            all_cog_data = [float(i) for i in all_cog_data_str]
            del all_cog_data_str
            
            
            all_data = [time] + all_joint_data + all_ergo_data + all_cog_data + [child.attrib['tc']]
            if count == 0 and debug == 1: 
                print('length of the actual data: {}'.format(len(all_data)))
            #Add data to dictionary
            xsens_data[index] = all_data
            count += 1;
            
            
    print(all_data)
    print(len(all_data))
    print(xsens_dataframe_columns)   
    print(len(xsens_dataframe_columns))     
    assert(len(all_data)==len(xsens_dataframe_columns))
    #Create dataframe from the dictionary
    xsens_dataframe = pd.DataFrame.from_dict(xsens_data, orient='index',
                                          columns = xsens_dataframe_columns)
    
    return (xsens_dataframe,datetime_start)

if __name__ == '__main__':
    from time import time as times
    import sys
    filename = sys.argv[1]
    
    xsens_data, starttime = read_xsens_xml(filename)