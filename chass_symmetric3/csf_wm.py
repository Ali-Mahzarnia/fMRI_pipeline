#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:37:17 2023

@author: ali
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import morphology
from nibabel import load, save, Nifti1Image, squeeze_image
import os
import sys, string, os
import pandas as pd
import openpyxl

mypath= '/Users/ali/Desktop/Feb23/fmri_pipeline/'

#construct csf and wm mask
label_path= mypath+'chass_symmetric3/chass_symmetric3_labels_PLI.nii.gz'
label_nii=nib.load(label_path)
label_nii.shape
data_label=label_nii.get_fdata()
roi_list=np.unique(data_label)
roi_list = roi_list[1:]


path_atlas_legend = mypath+ '/chass_symmetric3/CHASSSYMM3AtlasLegends.xlsx'
legend  = pd.read_excel(path_atlas_legend)
index_csf = legend [ 'Subdivisions_7' ] == '8_CSF'
index_wm = legend [ 'Subdivisions_7' ] == '7_whitematter'

vol_index_csf = legend[index_csf]
vol_index_csf = vol_index_csf['index2']



vol_index_wm  = legend[index_wm]
vol_index_wm  = vol_index_wm['index2']




label_nii_csf_data =label_nii.get_fdata()*0

for csf in vol_index_csf:
    #print(csf)
    label_nii_csf_data[  data_label == int(csf)] = 1
    
    
    
file_result= nib.Nifti1Image(label_nii_csf_data, label_nii.affine, label_nii.header)
nib.save(file_result,mypath + 'chass_symmetric3/csf_mask.nii.gz'  )

label_path_res= mypath+'chass_symmetric3/chass_symmetric3_labels_PLI_res.nii.gz'
os.system('/Applications/ANTS/antsApplyTransforms -d 3 -e 0 --float  -u float -i ' +mypath +'chass_symmetric3/csf_mask.nii.gz -n NearestNeighbor -r '+label_path_res+" -o "+mypath +'chass_symmetric3/csf_mask_0p3.nii.gz') 






label_nii_wm_data =label_nii.get_fdata()*0

for wm in vol_index_wm:
    #print(csf)
    label_nii_wm_data[  data_label == int(wm)] = 1
    
    
    
file_result= nib.Nifti1Image(label_nii_wm_data, label_nii.affine, label_nii.header)
nib.save(file_result,mypath + 'chass_symmetric3/wm_mask.nii.gz'  )
os.system('/Applications/ANTS/antsApplyTransforms -d 3 -e 0 --float  -u float -i ' +mypath +'chass_symmetric3/wm_mask.nii.gz -n NearestNeighbor -r '+label_path_res+" -o "+mypath +'chass_symmetric3/wm_mask_0p3.nii.gz') 


########### making a mask out of labels

label_path= mypath +'/chass_symmetric3/chass_symmetric3_labels_PLI_res.nii.gz'
label_nii=nib.load(label_path)
mask_labels_data = label_nii.get_fdata()
mask_labels = np.unique(mask_labels_data)
mask_labels=np.delete(mask_labels, 0)
mask_of_label =label_nii.get_fdata()*0

for vol in mask_labels:
    mask_of_label[  mask_labels_data == int(vol)] = 1
    
file_result= nib.Nifti1Image(mask_of_label, label_nii.affine, label_nii.header)
nib.save(file_result,mypath + 'chass_symmetric3/mask_of_label.nii.gz'  )    
    









###############


path ='/Volumes/dusom_mousebrains/All_Staff/Nariman_fmri_pipeline/chass_symmetric3/chass_symmetric3_labels_PLI_res.nii.gz'
nii=nib.load(path)
nii.shape
data =nii.get_fdata()
roi=np.unique(data)
roi = roi[1:]



