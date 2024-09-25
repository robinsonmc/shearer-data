# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:28:45 2024

Run this to get example plots

@author: robin
"""

import extract_from_raw_data as efrd
from config import GBL_PATH_TO_DATA

if __name__ == '__main__':
    '''
    Run this to plot example features:
        - additional features can be plotted, a full list can
        be found in to ALL_FEATURES_LIST.py
    
    
    The following conditions can be set to true to extract from the raw data:
        - extract_metadata
        - segment and extract_features
        
    Note: this will take a lot longer to run
    '''
    
    efrd.get_the_feature_plots(extract_metadata = False, 
                          segment_and_extract_features = True,
                          data_path=GBL_PATH_TO_DATA)