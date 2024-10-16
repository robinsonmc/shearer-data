# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:12:52 2024

@author: robinsonmc
"""

#Set this to 1 for verbose messages while running the code
GBL_DEBUG = 0

#Path to the downloaded data:
GBL_PATH_TO_DATA = "D:\Data_for_up"

#Path to a set of features extracted for previous analysis that can be loaded
#if the segment_and_extract_features flag is set to false.
GBL_PATH_TO_SAVED_FEATURES = "./new_saved_features_11_subjects"

#Binary file containing information linking filenames to subjects and runs
GBL_PATH_TO_METADATA_ORIG = 'metadata_all_shearers.pickle'

#Path where the extracted features are saved if the 
# segment_and_extract_features flag is set to false. NOTE THAT THIS
# DIRECTORY MUST ALREADY EXIST, it will not be created by the code.
GBL_EXTRACTED_FEATURES_SAVE_PATH = "saved_features_test_extract"

#If the extract_metadata flag is set to true, the metadata will be saved
#at the following path. This file will be generated if it does not 
#already exist, and it will overwrite if it does exist.
GBL_SAVEPATH_FOR_GENERATED_METADATA = 'metadata_all_shearers_generated.pickle'