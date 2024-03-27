#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:27:39 2022

@author: ali
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import morphology
from nibabel import load, save, Nifti1Image, processing
import os

'''
path='pred_mask_A19120504_T1.nii.gz'
file = nib.load(path)
file_data=file.get_fdata()
#thresh=np.median(fil_data)
thresh=np.quantile(file_data, 0.9)
thresh=0.00001
file_data[ file_data < thresh] = 0
file_data[ file_data >= thresh] = 1
sum(sum(sum(file_data))) 
path_out='bin_mask_A19120504_T1.nii.gz';

file_data=file_data.astype(int)#astype(int)

dtype=np.int
file_data = morphology.binary_erosion(file_data, iterations=1, structure=np.ones((5,5,5)))#, structure=np.ones((5,5)) .astype(a.dtype)
file_data = morphology.binary_dilation(file_data, iterations=1, structure=np.ones((5,5,5)))#, structure=np.ones((5,5)) .astype(a.dtype)

file_result= nib.Nifti1Image(file_data, file.affine, file.header)
nib.save(file_result, path_out)
'''



##### loop 
data_path= '/Volumes/Data/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/reorient_work/'
files_list=os.listdir(data_path)
files_list = [s for s in files_list if "pred_mask" in s ]
output = '/Volumes/Data/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/reorient_work/'

dtype=np.int
thresh=0.0001

for file in files_list:
    single_file=nib.load(data_path + file)
    file_data=single_file.get_fdata()
    file_data[ file_data < thresh ] = 0
    file_data[ file_data >= thresh] = 1
    file_data=file_data.astype(int)#astype(int)
    file_data = morphology.binary_erosion(file_data, iterations=5, structure=np.ones((4,4,4)))#, structure=np.ones((5,5)) .astype(a.dtype)
    file_data = morphology.binary_dilation(file_data, iterations=10, structure=np.ones((4,4,4)))#, structure=np.ones((5,5)) .astype(a.dtype)
    file_data = morphology.binary_erosion(file_data, iterations=5, structure=np.ones((4,4,4)))#, structure=np.ones((5,5)) .astype(a.dtype)
    #file_data = morphology.binary_dilation(file_data, iterations=10, structure=np.ones((2,2,2)))#, structure=np.ones((5,5)) .astype(a.dtype)
    #file_data = morphology.binary_erosion(file_data, iterations=10, structure=np.ones((2,2,2)))#, structure=np.ones((5,5)) .astype(a.dtype)
    file_result= nib.Nifti1Image(file_data, single_file.affine, single_file.header)
    nib.save(file_result, output+"bin_"+file)
    
   # smoothing 
data_path = output
files_list=os.listdir(data_path)   
files_list = [s for s in files_list if "bin_" in s ]
for file in files_list:
    single_file=nib.load(data_path + file)
    single_file = nib.processing.smooth_image(single_file, (.1,.1,.1), mode='constant')
    file_data=single_file.get_fdata()
    file_data[ file_data > 0 ] = 1
    file_result= nib.Nifti1Image(file_data, single_file.affine, single_file.header)
    nib.save(single_file, output+file)


#np.unique(file_data)







