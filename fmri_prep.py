#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:04:03 2023

@author: ali
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import morphology
from nibabel import load, save, Nifti1Image, squeeze_image
import os
import sys, string, os
import pandas as pd
#import openpyxl



#subj = 'A22040401' #reads subj number with s... from input of python file 
subj = sys.argv[1] #reads subj number with s... from input of python file 

mypath= '/mnt/munin2/Badea/Lab/mouse/fmri_pipeline/' # root 
input_path = mypath+'/fmri_raw_files/' #add input path + subj + ... to have the path of functional data 


#add input path + subj + ... to have the path of functional data 
fmri_file_path=input_path+ subj +'_fMRI.nii.gz' 
bold=nib.load(fmri_file_path) # read the data of this functional file as nib object
bold_data=bold.get_fdata() #read data as array 

#start masking T1 rare
T1_file_path=input_path + subj + '_T1.nii.gz' # path of T1
T1=nib.load(T1_file_path) # read the data of this anatomical file as nib object
T1_data=T1.get_fdata()#read data as array 




#read and get mask data for the subject
mask_file_path = mypath+'T1_mask_binary/pred_mask_'+ subj + '_T1.nii.gz' #path of mask
mask=nib.load(mask_file_path) # read mask file
mask_data=mask.get_fdata() # get data of mask

#set output for T1 masked
output_T1_masked = mypath+'T1_masked/'
if not os.path.isdir(output_T1_masked) : os.mkdir(output_T1_masked)

#mask the T1 image by multlipication
T1_masked_data=mask_data*T1_data

T1_masked_nii=nib.Nifti1Image(T1_masked_data, T1.affine, T1.header)
nib.save(T1_masked_nii, output_T1_masked + subj + '.nii.gz')
T1_masked_path=output_T1_masked + subj + '.nii.gz'
#end masking T1 rare and saving it


T1_atlas_reg_dir = mypath+'T1_atlas_reg/' # make directory for T1 masked
if not os.path.isdir(T1_atlas_reg_dir) : os.mkdir(T1_atlas_reg_dir)

out_T1_atlas_reg = T1_atlas_reg_dir +subj +"_" 
Atlas_T1_path = mypath+'/chass_symmetric3/chass_symmetric3_T1_PLI_0p1.nii.gz'

#reorient T1 call it out_T1_atlas_reg+"RAI.nii.gz"
os.system("c3d "+T1_masked_path+" -orient RAI -o "+out_T1_atlas_reg+"RAI.nii.gz")
#register reoiriented T1 to Atlas T1/ diffusion  results are in out_T1_atlas_reg/0Generic...mat
os.system("antsRegistration -v 1 -d 3 -m Mattes["+Atlas_T1_path+" ,"+out_T1_atlas_reg+"RAI.nii.gz,1,32,None] -r ["+Atlas_T1_path+" ,"+out_T1_atlas_reg+"RAI.nii.gz,1] -t affine[0.1] -c [300x300x0x0,1e-8,20] -s 4x2x1x0.5vox -f 6x4x2x1 -u 1 -z 1 -l 1 -o "+out_T1_atlas_reg+ " >/dev/null 2>&1")  
#appply previous transform and resample to T1 original
#os.system("antsApplyTransforms -d 3 -e 0 --float  -u float -i "+out_T1_atlas_reg+"RAI.nii.gz -r "+Atlas_T1_path+" -o "+out_T1_atlas_reg+"regs.nii.gz -t "+out_T1_atlas_reg+"0GenericAffine.mat"+ " >/dev/null 2>&1") 
#os.system("ResampleImageBySpacing  3 " +out_T1_atlas_reg+"regs.nii.gz " +out_T1_atlas_reg+"regs.nii.gz 0.1 0.1 0.1 0 0 0 >/dev/null 2>&1") 

os.system("antsApplyTransforms -d 3 -e 0 --float  -u float -i "+out_T1_atlas_reg+"RAI.nii.gz -r "+Atlas_T1_path+" -o "+out_T1_atlas_reg+"regs.nii.gz -t "+out_T1_atlas_reg+"0GenericAffine.mat"+ " >/dev/null 2>&1") 
#new register T1_rare to atlas PLI after reorienting to RAI as well as resampling



#####
#new dirs for volumes registered to atlas
vol_dir = mypath + 'vol_atlas_reg/'
if not os.path.isdir(vol_dir) : os.mkdir(vol_dir)
out_vol_atlas_reg = vol_dir + subj+'_'
Atlas_T1_path = mypath+'chass_symmetric3/chass_symmetric3_T1_PLI_0p3.nii.gz'
##

#new path of b0 field map and its output for cropping before use after making a b0_after folder
b0_path= input_path + subj +'_B0_Map.nii.gz'
b0_after_folder = mypath + '/b0_after/'
if not os.path.isdir(b0_after_folder) : os.mkdir(b0_after_folder)
b0_path_after= b0_after_folder + subj +'_after_B0Map.nii.gz'
#os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ' # only for spider

#set path for squuezed and coreg fmri
squeezed_data_path= mypath+'squeezed/'
if not os.path.isdir(squeezed_data_path) : os.mkdir(squeezed_data_path)

fmri_coreg_path =  mypath + 'fmri_coreg/'
if not os.path.isdir(fmri_coreg_path) : os.mkdir(fmri_coreg_path)

out_masked_slice= mypath+'masked_sliced/' #volume sliced masked path
if not os.path.isdir(out_masked_slice) : os.mkdir(out_masked_slice)


#loop over volumes of subj fmri every multiple of 3 (3 echos)
begin_volume = 4 #begin the time points volume
for i in range(begin_volume,int(bold_data.shape[3]/3)):
#for i in range(begin_volume,200):     #only for test otherwise uncomment last line
    #print(i)
    bold_data_3d=bold_data[:,:,:,(3*i)] #read volume
    
    squuezed=squeeze_image(nib.Nifti1Image(bold_data_3d,bold.affine)) #squeeze the last dimension

   
    nib.save(squuezed, squeezed_data_path + subj + '_'+str(i)+'.nii.gz') #save the squeezed fmri 3*i th volume as ith squeezed image
    squuezed_path=squeezed_data_path + subj + '_'+str(i)+'.nii.gz'  #to read and use the squeezed ith volume save its path
    squuezed_path_1=squeezed_data_path + subj + '_'+str(begin_volume)+'.nii.gz'  #to read and use the squeezed 1st volume save its path
    squuezed_path_b0 = squeezed_data_path + subj + '_'+str(i)+'b0.nii.gz'   # squuezed path of volume i after applied fugue
    
    
    # for the b0 register ot to the beginer volume   save in b0_path_after
    if i==begin_volume : os.system("antsApplyTransforms -d 3 -e 0 -i "+b0_path+" -r "+squuezed_path+" -u float -o "+b0_path_after+" >/dev/null 2>&1") #>/dev/null 2>&1"       
    #apply the registered b0 map to all volume to unwarp
    os.system("fugue -i "+squuezed_path+" --dwell=0.00139 --loadfmap="+b0_path_after+" -s 0.3 --unwarpdir=x- -v -u "+ squuezed_path_b0 +" >/dev/null 2>&1")#" >/dev/null 2>&1" 
    #path of squuezed volume ith and 1sth after nbias corrections
    squuezed_path_b0_nbias = squeezed_data_path + subj + '_'+str(i)+'corrected.nii.gz' 
    squuezed_path_b0_nbias_1 = squeezed_data_path + subj + '_'+str(begin_volume)+'corrected.nii.gz' 
    #save the bias just in case
    bias_bold = squeezed_data_path + subj +'_bias_bold.nii.gz'
    #bias correction call it squuezed_path_b0_nbias
    os.system('N4BiasFieldCorrection -d 3 -v 1 -s 2 -b [ 2x2x2 , 3 ] -c [ 100 x 50, 1e-6 ] -i '+ squuezed_path_b0 +' -o  [' + squuezed_path_b0_nbias + ' , ' + bias_bold + ' ]' + " >/dev/null 2>&1")

    outnib = fmri_coreg_path + subj + '_'+str(i)+'.nii.gz'
    outmat = fmri_coreg_path + subj 

    out_nib_masked= out_masked_slice+ subj + '_'+str(i)+".nii.gz"
  


    #registering squeezed ith vlume to T1 masked :1- affine computation 2- apply computed affine transform
    #1-affine  mat output in outmat folder (fmri_core + subj) as: subj+0GenericAffine
    if i== begin_volume : os.system("antsRegistration -v 1 -d 3 -m Mattes["+T1_masked_path+" ,"+squuezed_path_b0_nbias+",1,32,None] -t affine[0.1] -c [3000x3000x300x0,1e-8,20] -s 4x2x1x0.5vox -f 6x4x2x1 -u 1 -z 1 -l 1 -o "+outmat +">/dev/null 2>&1")  
    if not i==begin_volume : os.system("antsRegistration -v 1 -d 3 -m Mattes["+squuezed_path_b0_nbias_1+" ,"+squuezed_path_b0_nbias+",1,32,None] -t affine[0.1] -c [3000x3000x300x0,1e-8,20] -s 4x2x1x0.5vox -f 6x4x2x1 -u 1 -z 1 -l 1 -o "+outmat+"21" +" >/dev/null 2>&1")  

    #2-apply affine in previous step and save in outnib folder  (fmri_core + subj) 
    if i==begin_volume: os.system("antsApplyTransforms -d 3 -e 0 -i "+squuezed_path_b0_nbias+" -r "+T1_masked_path+" -u float -o "+outnib+" -t "+outmat+"0GenericAffine.mat >/dev/null 2>&1") #" >/dev/null 2>&1"
    if not i==begin_volume: os.system("antsApplyTransforms -d 3 -e 0 -i "+squuezed_path_b0_nbias+" -r "+T1_masked_path+" -u float -o "+outnib+" -t "+outmat+"0GenericAffine.mat " +outmat+"210GenericAffine.mat") #" >/dev/null 2>&1"
    # multiply the registered squeezed ith volume to the masked T1 by the mask  save in masked_sliced folder + ith+ subj
    #os.system("/Users/ali/Downloads/ANTsR/install/bin/ImageMath 3 "+out_nib_masked+" m "+mask_file_path+" " +outnib + ">/dev/null 2>&1") 
    
    # path of registered ith volume
    out_vol_atlas_reg_ith = out_vol_atlas_reg +str(i)

    #if outnib output instead of outnib masked the fmri are unmasked, next line
    #os.system("c3d "+out_nib_masked+" -orient RAI -o "+out_vol_atlas_reg_ith+"regs.nii.gz")
    os.system("c3d "+outnib+" -orient RAI -o "+out_vol_atlas_reg_ith+"regs.nii.gz")
  #  os.system("/Applications/ANTS/antsRegistration -v 1 -d 3 -m Mattes["+Atlas_T1_path+" ,"+out_vol_atlas_reg_ith+"RAI.nii.gz,1,32,None] -r ["+Atlas_T1_path+" ,"+out_vol_atlas_reg_ith+"RAI.nii.gz,1] -t affine[0.1] -c [300x300x0x0,1e-8,20] -s 4x2x1x0.5vox -f 6x4x2x1 -u 1 -z 1 -l 1 -o "+out_vol_atlas_reg_ith+" >/dev/null 2>&1")  
  
  #  os.system("antsApplyTransforms -d 3 -e 0 --float  -u float -i "+out_vol_atlas_reg_ith+"regs.nii.gz -r "+Atlas_T1_path+" -o "+out_vol_atlas_reg_ith+"regs.nii.gz -t "+out_T1_atlas_reg+"0GenericAffine.mat"+ ">/dev/null 2>&1") 
  #  os.system("ResampleImageBySpacing  3 " +out_vol_atlas_reg_ith+"regs.nii.gz " +out_vol_atlas_reg_ith+"regs.nii.gz 0.3 0.3 0.3 0 0 0 >/dev/null 2>&1") 
 
    os.system("antsApplyTransforms -d 3 -e 0 --float  -u float -i "+out_vol_atlas_reg_ith+"regs.nii.gz -r "+Atlas_T1_path+" -o "+out_vol_atlas_reg_ith+"regs.nii.gz -t "+out_T1_atlas_reg+"0GenericAffine.mat"+ ">/dev/null 2>&1") 


  

#TimeSeriesAssemble : Outputs a 4D time-series image from a list of 3D volumes.

# from masked_sliced folder for that paricular subj concatenating all time series abd saving in  output folder
output_path =  mypath + 'output/'
if not os.path.isdir(output_path) : os.mkdir(output_path)
 
outpath4D = output_path + subj + "_4D.nii.gz"
os.system(f'ImageMath 4 {outpath4D} TimeSeriesAssemble 1 0 {out_vol_atlas_reg}*.nii.gz') # concatenate volumes of fmri

#despike
prefix =  output_path + subj + '_despike.nii.gz' 
#if os.path.isfile(prefix): os.remove(prefix)
os.system('3dDespike -prefix '+ prefix+' '+ outpath4D+" -overwrite >/dev/null 2>&1") # smoothing


#slice time correction
outpath4D_tc=output_path+subj+"_4D_tc.nii.gz"
#os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ' # only for spider
os.system('slicetimer -i '+prefix +' -o '+outpath4D_tc + ' --down -r 2.25 -d 3 >/dev/null 2>&1' ) # time slice correction
#os.system('/Users/ali/abin/3dTshift -prefix '+outpath4D_tc +' -TR 2.25s -tpattern seq-z '+prefix+ ' -overwrite' ) # time slice correction


# detrend
prefix_detrend =  output_path + subj + '_detrend.nii.gz' 
#if os.path.isfile(prefix_detrend): os.remove(prefix_detrend)
os.system('3dDetrend -polort 9 -prefix '+ prefix_detrend+' '+ outpath4D_tc + ' -overwrite >/dev/null 2>&1') # smoothing






#new_mask_dir = mypath +'mask_binary_after/'
#if not os.path.isdir(new_mask_dir) : os.mkdir(new_mask_dir)
#mask_file_path_after = new_mask_dir + subj + '_T1_after.nii.gz'
mask_file_path_after = mypath +'chass_symmetric3/mask_of_label.nii.gz' # using the mask made by the atlas label

#out_vol_atlas_reg_1st = out_vol_atlas_reg +str(begin_volume)
#os.system("/Applications/Convert3DGUI.app/Contents/bin/c3d "+mask_file_path+" -orient RAI -o "+mask_file_path_after)
#os.system("/Applications/ANTS/antsApplyTransforms -n NearestNeighbor -d 3 -e 0 --float  -u float -i "+mask_file_path_after+" -r "+Atlas_T1_path+" -o "+mask_file_path_after+" -t "+out_T1_atlas_reg+"0GenericAffine.mat"+ " >/dev/null 2>&1") 
#os.system("/Applications/ANTS/antsApplyTransforms -n NearestNeighbor -d 3 -e 0 --float  -u float -i "+mask_file_path_after+" -r "+out_vol_atlas_reg_1st+"regs.nii.gz  -o "+mask_file_path_after+"  >/dev/null 2>&1") 



fmri_masked = output_path + subj + '_masked_detrend.nii.gz'
#skip detrend and mask the fmri
os.system("fslmaths "+outpath4D_tc+" -mul "+mask_file_path_after+" " + fmri_masked + ">/dev/null 2>&1") 





#scale: first mean
prefix_mean =  output_path + subj + '_mean.nii.gz' 
#if os.path.isfile(prefix_mean): os.remove(prefix_mean)
os.system('3dTstat -prefix '+prefix_mean+ '  '+fmri_masked + ' -overwrite >/dev/null 2>&1' ) #






#scaling
prefix_scaled =  output_path + subj + '_scaled.nii.gz' 
#if os.path.isfile(prefix_scaled): os.remove(prefix_scaled) 
os.system("3dcalc -a "+outpath4D_tc+ " -b "+ prefix_mean + " -c "+ mask_file_path_after + " -expr  'c*min(200,a/b*100)*step(a)*step(b)' -prefix "+ prefix_scaled + ' -overwrite >/dev/null 2>&1') # 





# confound corr
CSF_1D = output_path + subj + '_CSF.1D' 
csf_mask= mypath +'chass_symmetric3/csf_mask_0p3.nii.gz'
os.system('fslmeants -i '+prefix_scaled + ' -o ' + CSF_1D + ' -m ' + csf_mask  + ' >/dev/null 2>&1'  )

WM_1D = output_path + subj + '_WM.1D' 
wm_mask= mypath +'chass_symmetric3/wm_mask_0p3.nii.gz'
os.system('fslmeants -i '+prefix_scaled + ' -o ' + WM_1D + ' -m ' + wm_mask  + ' >/dev/null 2>&1'  )

 #create motion files
 #mfile=motion_te1.1D
 #/Users/ali/abin/3dcalc/1d_tool.py -infile $mfile -set_nruns 1  -derivative  -collapse_cols euclidean_norm   -write motion_enorm.1D
 #1d_tool.py -infile $mfile -set_nruns 1  -demean -write motion_demean.1D
 #1d_tool.py -infile $mfile -set_nruns 1 -derivative -demean -write motion_deriv.1D

 # create bandpass regressors
#bandpass_rall_1D = output_path + subj + '_bandpass_rall.1D' 
#os.system('/Users/ali/abin/1dBport -nodata ' + str(int(bold_data.shape[3]/3)-begin_volume)+' ' + str(2.25) +' -band 0.01 0.1 -invert -nozero > ' + bandpass_rall_1D)

X_xmat_1D = output_path + subj + '_X.xmat.1D' 
X_jpg = output_path + subj +'_X.jpg'
fitts_subj  = output_path + subj +'_fitts.'+subj
errts_subj =  output_path + subj +'_errts.'+subj
stats_subj = output_path + subj +'_stats.' +subj

#os.system('3dDeconvolve -input ' + prefix_scaled +' -ortvec ' +  bandpass_rall_1D + '  bandpass ' +  -ortvec ' + CSF_1D +' css '+ '-ortvec ' + WM_1D + ' wm ' + '-polort 5 -float  -num_stimts 0  -fout -tout -x1D ' + X_xmat_1D+ ' -xjpeg ' + X_jpg +  ' -fitts '+  fitts_subj  + ' -errts ' + errts_subj + '-x1D_stop ' + ' -bucket ' + stats_subj )              
#os.system('3dDeconvolve -input ' + prefix_scaled +' -ortvec ' +  bandpass_rall_1D + '  '+ str(int(bold_data.shape[3]/3)-1) +' -ortvec ' + CSF_1D +' css '+ '-ortvec ' + WM_1D + ' wm ' + '-polort 5 -float  -num_stimts 0  -fout -tout -x1D ' + X_xmat_1D+ ' -xjpeg ' + X_jpg +  ' -fitts '+  fitts_subj  + ' -errts ' + errts_subj + '-x1D_stop ' + ' -bucket ' + stats_subj )              
os.system('3dDeconvolve -input ' + prefix_scaled +' -ortvec ' + CSF_1D +  ' ' + csf_mask + ' -ortvec ' + WM_1D + ' ' + wm_mask + ' -polort 5 -float  -num_stimts 0  -fout -tout -x1D ' + X_xmat_1D+ ' -xjpeg ' + X_jpg +  ' -fitts '+  fitts_subj  + ' -errts ' + errts_subj + '-x1D_stop ' + ' -bucket ' + stats_subj + ' -overwrite >/dev/null 2>&1' )              

errts_nii_gz = output_path + subj +'_errts.nii.gz'
os.system('3dTproject -polort 0  -input ' + prefix_scaled + ' -ort ' + X_xmat_1D + ' -passband 0.01 0.1 -prefix '  + errts_nii_gz + ' -overwrite >/dev/null 2>&1' )



#time series
def parcellated_matrix(sub_timeseries, atlas_idx, roi_list):
    timeseries_dict = {}
    for i in roi_list:
        roi_mask = np.asarray(atlas_idx == i, dtype=bool)
        timeseries_dict[i] = sub_timeseries[roi_mask].mean(axis=0)
        #print (i)
    roi_labels = list(timeseries_dict.keys())
    sub_timeseries_mean = []
    for roi in roi_labels:
        sub_timeseries_mean.append(timeseries_dict[roi])
        #print(sum(sub_timeseries_mean[int(roi)]==0))
    #corr_matrix = np.corrcoef(sub_timeseries_mean)
    return sub_timeseries_mean









label_path= mypath+'chass_symmetric3/chass_symmetric3_labels_PLI_res.nii.gz'
label_nii=nib.load(label_path)
label_nii.shape
data_label=label_nii.get_fdata()
roi_list=np.unique(data_label)
roi_list = roi_list[1:]
#atlas_idx = data_label

time_ser_path= mypath + 'time_ser/'
if not os.path.isdir(time_ser_path) : os.mkdir(time_ser_path)
file_data=nib.load(errts_nii_gz)
sub_timeseries=file_data.get_fdata()
result=parcellated_matrix(sub_timeseries, data_label, roi_list)
if os.path.isfile(time_ser_path + 'ts_' + subj +  '.csv'): os.remove(time_ser_path + 'ts_' + subj +  '.csv')
np.savetxt( time_ser_path + 'ts_' + subj +  '.csv'  , result, delimiter=',', fmt='%s')





# FCs


def parcellated_FC_matrix(sub_timeseries, atlas_idx, roi_list):
    timeseries_dict = {}
    for i in roi_list:
        roi_mask = np.asarray(atlas_idx == i, dtype=bool)
        timeseries_dict[i] = sub_timeseries[roi_mask].mean(axis=0)
        #print (i)
    roi_labels = list(timeseries_dict.keys())
    sub_timeseries_mean = []
    for roi in roi_labels:
        sub_timeseries_mean.append(timeseries_dict[roi])
        #print(sum(sub_timeseries_mean[int(roi)]==0))
    corr_matrix = np.corrcoef(sub_timeseries_mean)
    return corr_matrix

# if more than 298 time series limit the time series to 299 tp 
if sub_timeseries.shape[3] >298 : sub_timeseries = sub_timeseries[:,:,:,:299]
#
resultFC=parcellated_FC_matrix(sub_timeseries, data_label, roi_list)
if os.path.isfile(time_ser_path + 'FC_' + subj +  '.csv'): os.remove(time_ser_path + 'FC_' + subj +  '.csv')
np.savetxt( time_ser_path + 'FC_' + subj +  '.csv'  , resultFC, delimiter=',', fmt='%s')






