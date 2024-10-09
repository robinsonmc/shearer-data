# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:01:47 2024

Some files were named in a way that differs from the expected name. Some of
the code in this repo relies on the expected name.

Search through the data directory recursively and rename files such that the
rest of the code can handle them.

@author: robin
"""

import os
from config import GBL_PATH_TO_DATA, GBL_DEBUG

def find_files(start_dir, prefix):
    for dirpath, dirnames, filenames in os.walk(start_dir):
        
        for dirname in dirnames:
            if dirname.startswith(prefix):
                
                #print('dirname: ' + dirname)
                #print(os.path.join(dirpath))
                
                new_dirname = dirname.replace('week4_shearer10','s10_week4_thursday')
                new_dirname = new_dirname.replace('week4_shearer11_friday','s11_week4_friday')
                
                if GBL_DEBUG == 1:
                    print('rename ' + dirname + ' with ' + new_dirname)
                try:
                    os.rename(os.path.join(dirpath, dirname),   os.path.join(dirpath,new_dirname))
                except FileExistsError:
                    pass
                    
                
                
                      
    for dirpath, dirnames, filenames in os.walk(start_dir):        
        for filename in filenames:
            if filename.startswith(prefix):
                
                #print(os.path.join(dirpath, filename))
                
                #Chain the replacements only one can do anything
                new_filename = filename.replace('week4_shearer10','s10_week4_thursday')
                new_filename = new_filename.replace('week4_shearer11_friday','s11_week4_friday')
                
                
                if GBL_DEBUG == 1:
                    print('rename ' + filename + ' with ' + new_filename)
                
                try:
                    os.rename(os.path.join(dirpath, filename),   os.path.join(dirpath,new_filename))
                except FileExistsError:
                    pass
#s10_week4_thursday_run1_part1

#week4_shearer10_run1_part1_emg

if __name__ == '__main__':
    
    test_dir = 'D:\Data_for_up\Week 4\s11_week4_friday_NEWTEST - Copy'
    
    find_files(test_dir, 'week')