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

path = '/mnt/munin2/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/reorient_work/'
#path = '/Volumes/Data/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/reorient_work/'
T2_root = '/mnt/munin2/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/'
#T2_root = '/Volumes/Data/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/'

list0=os.listdir(path)


list0 = [s for s in list0 if "bin_pred_mask_res_reor_" in s]





for file in list0:
    
    
    input_path  = path + file
    resample_mask_path = path + "res_mask_" +  file.replace("bin_pred_mask_res_reor_","")
    
    reoriented_T2_file = path +  "reoriented_" + file.replace("bin_pred_mask_res_reor_","")

    hs_mask_rs_ro = path +  "res_reoirent_mask_" + file.replace("bin_pred_mask_res_reor_","")
    
    original_T2 = T2_root  +  file.replace("bin_pred_mask_res_reor_","")
    
    T1 = file.replace("bin_pred_mask_res_reor_","")
    T1 = T1.replace("T2","T1")

    T2_mask = path + 'pred_mask_' +T1
        
    if not os.path.isfile(T2_mask): # check if mask exists 
        command = 'antsApplyTransforms -v -d 3 -e 0 -u char -n NearestNeighbor -i ' + input_path + ' -o '+ resample_mask_path+ ' -r ' +  reoriented_T2_file
        os.system(command)
        #print(command)
        
        command2 = '/mnt/clustertmp/common/rja20_dev/matlab_execs_for_SAMBA//img_transform_executable/run_img_transform_exec.sh /mnt/clustertmp/common/rja20_dev/MATLAB2015b_runtime/v90 ' + resample_mask_path +' ALS ASR '+ hs_mask_rs_ro
        os.system(command2)
        #print(command2)

            
        command3 = '/mnt/clustertmp/common/rja20_dev/gunnies/nifti_header_splicer.bash '  + original_T2 + ' ' + hs_mask_rs_ro + ' ' +  T2_mask
        os.system(command3)
        #print(command3)
