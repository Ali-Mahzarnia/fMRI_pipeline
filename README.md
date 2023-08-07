# fMRI_pipeline

# binarizer
Description:
This Python script performs preprocessing and smoothing on MRI image masks using the Nibabel and Scipy libraries. The main goal is to generate binary masks from MRI image data and subsequently apply erosion, dilation, and smoothing operations to enhance the quality of the masks.

The script includes the following functionalities:

Generate Binary Masks from MRI Images:

Load an MRI image from a given path.
Threshold the image data to create a binary mask, where values above a certain threshold are set to 1 (foreground) and values below the threshold are set to 0 (background).
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
This script provides a simple yet effective way to preprocess and enhance MRI image masks, making them suitable for further analysis, such as brain region segmentation or connectivity studies.

Note: The script assumes that the MRI images are in NIfTI format and that the filenames and directory paths are appropriately configured for the data used. Additionally, this description is based on the code provided, and further context or details may be available in the full repository.




