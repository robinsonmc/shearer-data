# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:22:29 2024

set up the the raw data feature extraction

@author: robin
"""

import pickle
import get_all_metadata as gam



#Optional steps
#1
extract_metadata = True
#2
segment_data = True
#3
extract_features = True


if extract_metadata:

    a = gam.get_all_metadata("D:\Data_for_up")
    #a = gam.get_all_metadata("D:\Data_for_up\Week 4")
    
    with open('metadata_all_shearers.pickle','wb') as f:
        pickle.dump(a,f,protocol=pickle.HIGHEST_PROTOCOL)
        
if segment_data:
    pass
    #Segment the data steps here
    
