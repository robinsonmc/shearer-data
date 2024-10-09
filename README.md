# How to use

## 1. Set the correct paths
Edit the config.py file with the right paths, (a) to the root directory where 
the data is, and (b) optionally a directory for the extracted features 
if running with the segment_and_extract_features arg set true.

## 2. Make sure all packages are installed
See the environment.yml file.

## 3a. Run from saved feature list
If you run Run.py it will call get_the_feature_plots with default args which
will generate the plots in the paper from the features previously extracted and stored
in ./new_saved_features_11_subjects.

## 3b. Run from raw data
You can easily modify Run.py to call the get_the_feature_plots function with the following
flags set True:
* extract_metadata
* segment_and_extract_features
    
If all the flags are set true, this will run everything on the raw data.
There is probably little need to re-extract the metadata unless new data is added,
but it  may be useful to re-segment and re-extract features as this can be used to 
validate the dataset. This code could be further modified to extract new or altered
features.

## Example output
![example_output](https://github.com/user-attachments/assets/6eb72d5a-96ba-46de-ac2c-6a83f7f4f874)

Executing Run.py should produce this figure.


## Investigating other features
The code can be changed to extract or investigate other features. A full list
of saved features is in ALL_FEATURES_LIST.py, and the functions to extract
features can be added/modified in get_feature_functions.py and can be called in
extract_from_reaw_data.py
