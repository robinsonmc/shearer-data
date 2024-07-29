# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:34:56 2020

@author: mrobinson2
"""
import pathlib
import models_movie as mm

class StringNotFound(ValueError):
    pass

class MetaData():
    def __init__(self,dir_path):
        self.shearer  = MetaData.get_shearer_num(dir_path)
        self.day      = MetaData.get_day(dir_path)
        self.run      = MetaData.get_run(dir_path)
        self.part     = MetaData.get_part(dir_path)
        self.dir_path = dir_path
        self.length   = MetaData.get_length(dir_path)
        
        
    def get_shearer_num(dir_path):
        path = pathlib.Path(dir_path)
        last_section = str(path.parts[-1])
        shearer_num = MetaData.find_str(last_section,'s')
        return int(shearer_num)
    
    def get_day(dir_path):
        path = pathlib.Path(dir_path)
        last_section = str(path.parts[-1])
        
        days = ['monday','tuesday','wednesday','thursday','friday','test']
        for el in days:
            if el in last_section: return el
        
        raise ValueError('day not found in dir_path')

    
    def get_run(dir_path):
        path = pathlib.Path(dir_path)
        last_section = str(path.parts[-1])
        shearer_num = MetaData.find_str(last_section,'run')
        return int(shearer_num)
    
    def get_part(dir_path):
        path = pathlib.Path(dir_path)
        last_section = str(path.parts[-1])
        try:
            shearer_num = MetaData.find_str(last_section,'part')
        except StringNotFound:
            shearer_num = 1
        return int(shearer_num)
    
    def get_length(dir_path):
        myData = mm.DataFile(dir_path)
        return len(myData.xsens_data)
    
    def find_str(search_string, target):
        index_start = search_string.find(target)
        if index_start == -1:
            raise StringNotFound('String not found in search')
        
        index_end = search_string.find('_',index_start)
        if index_end == -1: 
            index_end = len(search_string)
            
        return search_string[index_start+len(target):index_end]
    
    
        
if __name__ == '__main__':
    pass
    #from run_on_all import run_on_all
    #all_metadata = run_on_all(MetaData)
    
    #import pickle
    
    #with open('metadata_test.pickle','wb') as file:
    #    pickle.dump(all_metadata,file)