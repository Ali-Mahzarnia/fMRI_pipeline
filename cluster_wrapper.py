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
    BD = '/mnt/munin2/Badea/Lab/mouse'    
    #BD ='/Volumes/data/Badea/Lab/mouse'
else:
    print("BD is found locally.")
#create sbatch folder
job_descrp =  "fmri_coreg"
sbatch_folder_path = BD+"/Nariman_fmri_pipeline/"+job_descrp + '_sbatch/'

if not os.path.exists(sbatch_folder_path):
    os.system(f"mkdir -p {sbatch_folder_path}" )
    #os.makedirs(sbatch_folder_path)
GD = '/mnt/clustertmp/common/rja20_dev/gunnies/'

list_fmir_folders_path ='/mnt/munin2/Badea/Lab/mouse/fMRI_data_packages_for_Nariman/' #add input path + subj + ... to have the path of functional data
#list_fmir_folders_path = '/Volumes/Data/Badea/Lab/mouse/missing_fmri_files_raw_Nariman_Jul20th/' #add input path + subj + ... to have the path of functional data 

list_fmri_folders = os.listdir(list_fmir_folders_path)
list_of_subjs_long = [i for i in list_fmri_folders if '_fMRI' in i]

list_of_subjs = [i.partition('_fMRI.nii.gz')[0] for i in list_of_subjs_long]
#list_fmri_folders.remove(".DS_Store")

connectome_folder = BD+"/Nariman_fmri_pipeline/time_ser/" 

for subj in list_of_subjs:
    #print(subj)
    #fmri_file = list_fmir_folders_path +subj + "/ses-1/func/" + subj +"_ses-1_bold.nii.gz" 
    #nib.load(fmri_file)
    if not os.path.isfile( connectome_folder + "FC_"+subj + ".csv" ):
        python_command = "python /mnt/munin2/Badea/Lab/mouse/Nariman_fmri_pipeline/fmri_prep.py "+subj
        job_name = job_descrp + "_"+ subj
        command = GD + "submit_sge_cluster_job.bash " + sbatch_folder_path + " "+ job_name + " 0 0 '"+ python_command+"'"   
        os.system(command)

