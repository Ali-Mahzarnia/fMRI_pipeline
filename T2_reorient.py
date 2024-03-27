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

