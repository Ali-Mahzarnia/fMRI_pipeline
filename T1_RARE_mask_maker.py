#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Modified: 09/03/21
@author: hsm
"""

import numpy as np
#import keras
import os
import fnmatch
import sys
from fnmatch import fnmatch
import tensorflow as tf
import scipy.io as sio
import tensorflow.keras as keras
from keras import backend as K

from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Dropout, BatchNormalization, concatenate, Add, ReLU, Softmax, LeakyReLU
from keras.initializers import Constant
from keras.models import load_model
import h5py
import nibabel as nib
from tensorflow.python.client import device_lib 


# GPU Configuration ----------------------------------------------------------------
# 4 GPUs

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])

print("Number of Devices : {}".format(strategy.num_replicas_in_sync))


# Custom Loss and Accuracy functions ----------------------------------------------------------------
# Dice coefficients are calculated again separately after training and testing
def intensity_normalizer(image):

    img_normalized = np.copy(image)



    img_tmp_norm = (img_normalized-np.mean(img_normalized))/np.std(img_normalized)


    return img_tmp_norm

def dice_metric(y_true, y_pred):


    threshold = 0.3

    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)
    mask = y_true > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.multiply(y_true, mask)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    hard_dice = (2. * inse) / (l + r)
    mean_dice = tf.reduce_mean(hard_dice)
    # tf.debugging.check_numerics(mean_dice, 'NaN found', name=None)
    value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(mean_dice)), dtype=tf.float32)
    mean_dice_no_nan = tf.math.multiply_no_nan(mean_dice, value_not_nan)
    return mean_dice_no_nan



# Data Pre-processing ----------------------------------------------------------------

def contrast_clipper(image):
    img_clipped = np.copy(image)
    for i in range(0, len(image)):
        img_tmp = image[i,:,:,:]
        img_tmp[img_tmp > 0.8] = 0.8
        img_tmp[img_tmp < 0] = 0
        img_clipped[i,:,:,:] = img_tmp
    return img_clipped


def intensity_normalizer(image):
    img_tmp = np.copy(image)
    img_tmp_norm = (img_tmp-np.mean(img_tmp))/np.std(img_tmp)
    img_normalized = np.copy(img_tmp_norm)
    return img_normalized


def train_test_index(kfold, nth_fold, data_img):
    data_length = len(data_img)
    test_num = list(range(int(nth_fold)-1,data_length,kfold))
    train_num_pre = list(range(0,data_length))

    for i in range(0, len(test_num)):
        train_num_pre[test_num[i]] = []
    train_num = [ele for ele in train_num_pre if ele != []]
    return test_num, train_num

classification_threshold = 0.5




# CNN Network ----------------------------------------------------------------

def cnn(fliter_num, kernel_size, kfold, nth_fold, trial_str):
    with strategy.scope():
        my_metrics = [
              dice_metric,
  #            tf.keras.metrics.Precision(thresholds=classification_threshold,
  #                                       name='precision'
  #                                       ),
  #            tf.keras.metrics.Recall(thresholds=classification_threshold,
  #                                    name="recall"),
  #            tf.keras.metrics.AUC(name='auc')
        ]
        
        input_layer = keras.layers.Input(shape=(180, 100, 1))
        conv1a = keras.layers.Conv2D(filters=fliter_num, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(input_layer)
        conv1b = keras.layers.Conv2D(filters=fliter_num, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv1a)
        pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv1b)
        conv2a = keras.layers.Conv2D(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(pool1)
        conv2b = keras.layers.Conv2D(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv2a)
        pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2b)
        conv3a = keras.layers.Conv2D(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(pool2)
        conv3b = keras.layers.Conv2D(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv3a)

        dconv3a = keras.layers.Conv2DTranspose(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), padding='same')(conv3b)
        dconv3b = keras.layers.Conv2DTranspose(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), padding='same')(dconv3a)
        unpool2 = keras.layers.UpSampling2D(size=(2, 2))(dconv3b)
        cat2 = keras.layers.concatenate([conv2b, unpool2])
        dconv2a = keras.layers.Conv2DTranspose(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), padding='same')(cat2)
        dconv2b = keras.layers.Conv2DTranspose(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), padding='same')(dconv2a)
        unpool1 = keras.layers.UpSampling2D(size=(2, 2))(dconv2b)
        cat1 = keras.layers.concatenate([conv1b, unpool1])
        dconv1a = keras.layers.Conv2DTranspose(filters=fliter_num, kernel_size=(kernel_size, kernel_size), padding='same')(cat1)
        dconv1b = keras.layers.Conv2DTranspose(filters=fliter_num, kernel_size=(kernel_size, kernel_size), padding='same')(dconv1a)

        output = keras.layers.Conv2D(filters=1, kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')(dconv1b)

        model = keras.models.Model(inputs=input_layer, outputs=output)

##


def RARE_mask_pred(image_dir, output_dir):


    imglist_temp = os.listdir(image_dir)
    imglist=[s for s in imglist_temp if "T2" in s]

    for img_num in range(0,len(imglist)):

        img_name = imglist[img_num]

        image = nib.load(image_dir + '/'+ img_name)
        data_img = image.get_fdata()
        data_img = intensity_normalizer(data_img)
        affine_RARE = image.affine  

    ################## if unc data then flip image as follows: #######################

    #    data_img = np.rot90(data_img,2)
    #    data_img = np.flip(data_img,1)
    #    data_img = np.flip(data_img,2)





        print(np.shape(data_img))

        if (np.size(data_img,0) == 180 and np.size(data_img,1) == 200):

            data_clipped = intensity_normalizer(data_img)
            data_norm = intensity_normalizer(data_clipped)
            #data_norm = np.copy(data_img)

            #est_num, train_num = train_test_index(kfold, nth_fold, data_norm)


            test_img_rot = np.rot90(data_norm,axes=(0,1))
            print(np.shape(test_img_rot))

            test_img = np.reshape(test_img_rot, [200,180,100,1])

            print('1')


            model_path = '/Users/ali/Desktop/Mar24/fmri_masking_T2/model_2d_contrast_full_2.h5'


            with strategy.scope():
                my_metrics = [
                    dice_metric,
              #      tf.keras.metrics.Precision(thresholds=classification_threshold,
              #                               name='precision'
              #                               ),
              #      tf.keras.metrics.Recall(thresholds=classification_threshold,
              #                            name="recall"),
              #      tf.keras.metrics.AUC(name='auc')
                ]
                model = tf.keras.models.load_model(
                    model_path, custom_objects={'loss': 'binary_crossentropy', 'dice_metric': dice_metric, 'my_metrics': my_metrics}, compile=True, options=None
                )



            #np.save('/home/alex/Stroke_multi/results/history_3d_fold_' + str(nth_fold) + '_trial1_1.npy',history.history)
            #model.save('/home/alex/Stroke_multi/results/model_3d_fold_' + str(nth_fold) + '_trial1_1.h5')

            #nif_test = nib.Nifti1Image(y_val, affine=np.eye(4))
            #nib.save(nif_test, '/home/alex/Stroke_multi/results/mask_test_3d_fold_' + str(nth_fold) + '_trial1_1.nii')
                pred_pre = np.empty([200,180,100,1])
                for i in range(0,200):
                    pred_slice = np.reshape(test_img[i,:,:,:], [1,180,100,1])
                    pred_pre[i,:,:,:] = model.predict(pred_slice)


                pred_full_rot = np.rot90(pred_pre,axes=(1,0))
                pred = pred_full_rot[:,:,:,0]
                

                nif_img = nib.Nifti1Image(pred, affine=affine_RARE)
                #nif_img_raw = nib.Nifti1Image(pred_full, affine=affine_RARE)
                #nif_test_img = nib.Nifti1Image(test_img_rot, affine=affine_RARE)
                nib.save(nif_img, output_dir + '/pred_mask_' + img_name)
                #nib.save(nif_img_raw, '/home/alex/hsm/Invivo/RARE/pred/sub-22040410_ses-1_T1w_pred_mask_raw.nii')
                #nib.save(nif_test_img, '/home/alex/hsm/Invivo/RARE/pred/sub-22040410_ses-1_T1w_test_img.nii')
                print('Prediction Finished.')
                print(np.shape(pred))



if __name__=='__main__':
    image_dir = input("\n\n\nEnter image directory: ")
    output_dir = input("\nEnter mask output directory: ")
    RARE_mask_pred(image_dir, output_dir)





