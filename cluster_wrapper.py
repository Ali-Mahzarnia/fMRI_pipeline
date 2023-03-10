#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:30:59 2023

@author: ali
"""

import os , glob
import sys
#import nibabel as nib

try :
    BD = os.environ['BIGGUS_DISKUS']
#os.environ['GIT_PAGER']
except KeyError:  
    print('BD not found locally')
    BD = '***/mouse'    
    #BD ='***/example'
else:
    print("BD is found locally.")
#create sbatch folder
job_descrp =  "fmri_coreg"
sbatch_folder_path = BD+"/fmri_pipeline/"+job_descrp + '_sbatch/'

if not os.path.exists(sbatch_folder_path):
    os.system(f"mkdir -p {sbatch_folder_path}" )
    #os.makedirs(sbatch_folder_path)
GD = '***/gunnies/'

list_fmir_folders_path ='***/fmri_raw_files/'
list_fmri_folders = os.listdir(list_fmir_folders_path)
list_of_subjs_long = [i for i in list_fmri_folders if 'T1' in i]

list_of_subjs = [i.partition('_T1.nii.gz')[0] for i in list_of_subjs_long]
#list_fmri_folders.remove(".DS_Store")


for subj in list_of_subjs:
    #print(subj)
    #fmri_file = list_fmir_folders_path +subj + "/ses-1/func/" + subj +"_ses-1_bold.nii.gz" 
    #nib.load(fmri_file)
    python_command = "python ***/fmri_prep.py "+subj
    job_name = job_descrp + "_"+ subj
    command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " "+ job_name + " 0 0 '"+ python_command+"'"   
    os.system(command)

