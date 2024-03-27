#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:19:37 2024

@author: ali
"""


import nibabel as nib
import numpy as np
from scipy.ndimage import morphology
from nibabel import load, save, Nifti1Image, processing
import os

path_T2 = '/mnt/munin2/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/'
#path_T2 = '/Volumes/Data/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/'

T2_list=os.listdir(path_T2)


T2_list = [s for s in T2_list if "T2.nii.gz" in s]


reorient_folder_path = '/mnt/munin2/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/reorient_work/'
#reorient_folder_path = '/Volumes/Data/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/reorient_work/'

if not os.path.isdir(reorient_folder_path): os.mkdir(reorient_folder_path)


for file in T2_list:
    
    input_path  = path_T2 + file
    reoriented_path = reorient_folder_path + "reoriented_" +  file 
    resampled_path = reorient_folder_path + "res_reor_" +  file 


    if not os.path.isfile(resampled_path): # check if mask exists 
    
        command = '/mnt/clustertmp/common/rja20_dev/matlab_execs_for_SAMBA//img_transform_executable/run_img_transform_exec.sh /mnt/clustertmp/common/rja20_dev/MATLAB2015b_runtime/v90 '+ input_path + ' ASR ALS '+reoriented_path 
        #print(command)
        os.system(command)
        refrence_image_path = "/mnt/munin2/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/A23060501_T1_ALS_to_ALS.nii.gz"

        command2 = 'antsApplyTransforms -d 3 -e 0 --float  -u float -i '+ reoriented_path +  ' -r '  + refrence_image_path + ' -o ' + resampled_path
        #antsApplyTransforms -d 3 -e 0 --float  -u float -i **** -r **** -o ***
        #print(command2)
        os.system(command2)