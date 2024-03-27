# fMRI_pipeline
1.	With a trained deep-learning model found at (https://zenodo.org/records/10805906), probability masks are made via T1_RARE_mask_maker.py based on T1 images.
2.	The masks are eroded and dilated before binarizing them with a threshold using the binarizer.py. 
# binarizer
  Description:
  This Python script performs preprocessing and smoothing on MRI image masks using the Nibabel and Scipy libraries. The main goal is to generate binary masks      from MRI image data and subsequently apply erosion, dilation, and smoothing operations to enhance the quality of the masks.

  The script includes the following functionalities:

  Generate Binary Masks from MRI Images:

  Load an MRI image from a given path.
  Threshold the image data to create a binary mask, where values above a certain threshold are set to 1 (foreground) and values below the threshold are set to 0   (background).
  Perform binary erosion and dilation operations on the mask to remove small noise and fill gaps, enhancing the mask quality.
  Save the binary mask as a NIfTI file.
  Loop over Multiple MRI Image Masks:

  Load a set of MRI image masks from a specified directory path.
  Apply the same binary thresholding, erosion, and dilation operations to each mask in the set.
  Save the processed masks in an output directory.
  Smoothing of the Binary Masks:

  Load the processed binary masks from the output directory.
  Apply a smoothing operation to the binary masks using a specified smoothing kernel size.
  Convert smoothed image data to binary (1 for values above 0 and 0 for values equal to or below 0).
  Save the smoothed binary masks as NIfTI files.
  This script provides a simple yet effective way to preprocess and enhance MRI image masks, making them suitable for further analysis, such as brain region       segmentation or connectivity studies.
  
# if subjects have T2 instead of T1:
  The appropriate binarizer should be used based on the name of the code file (binarizerT1.py vs. binarizerT2.py). In addition if subject has T2 we need more       rotation using a)T2_reorient_for_DL_masking.py b)deep learning mask maker c)binarizerT2.py d)reorient_T2masks_toT2.py respectively. 

# fmri_prep.py description: 
We run fmri_prep.py via cluster_wrapper.py for all subjects.

3.	T1 are masked 
4.	Masked T1 are re-oriented RAI by c3d
5.	Using antsRegistration these T1 are registered to T1 of Atlas
6.	The registration is applied to these T1
7.	The T1 are resampled to have the same voxel dimensions to atlas
8.	The raw fmri files are read and separated into 3d images a loop  (every 3  iterations bc of 3 echos while throwing away 4 first images):
   
  a.	The b0 map is registered to the volumes before applied to them via fugue (--dwell=0.00139)

  b.	The result 3d images are gone through N4BiasFieldCorrection
  
  c.	For the beginning volume it is registered to T1 masked (of step 4)
  
  d.	For the rest of volumes, they are registered to the first volume
  
  e.	For the first volume the registration of 8.c is applied to it

  f.	For the rest of the volumes oth the regsiteration of 8.c and 8.d are applied to it

  g.	The results are re-oriented RAI by c3d

  h.	The registration of 5 is applied to these volumes	

  i.	These volumes are resampled to have the same voxel dimensions to atlas

10.	These volumes are concatenated after the above loop in 8. ImageMath
11.	 The results are despiked. 3dDespike
12.	Slice time correction: slicetimer -r 2.25 –down
13.	Deterning (skiped) : 3dDetrend -polort 9
14.	A mask is made based on the Atlas labels
15.	The results are masked using the results of 13 fslmaths
16.	3dTstat finds the mean of the fmri
17.	3dcalc find the scaled fmri c*min(200,a/b*100)*step(a)*step(b) where c is masked a is masked fmri and b is mean from 15.
18.	Wm and csf masks are made and scaled result of 16 are masked using them separately
19.	3dDeconvolve does the confound correction using 16 and 17 and (-polort 5 -float  -num_stimts 0).
20.	3dTproject -polort 0 and -passband 0.01 0.1 does bandpass filtering 
21.	TS and FC are computed.


![image](https://github.com/Ali-Mahzarnia/fMRI_pipeline/assets/69542146/9b859843-8d1e-4178-96a1-8f3e7880abd2)




