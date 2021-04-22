"""
File:     deep-fus/src/losses.py
Author:   Tommaso Di Ianni (todiian@stanford.edu)

Copyright 2021 Tommaso Di Ianni

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf

# CUSTOM loss
def custom_loss(beta=0.5):
    """
    This function only works for batch size of 1. 
    """
    
    def loss(y_true, y_pred):
        # Calculate MAE loss
        mae_loss = tf.keras.backend.mean(tf.keras.backend.abs(y_true-y_pred))
        
        # Reshape tensors
        y_true = tf.image.convert_image_dtype(tf.reshape(y_true,[1,96,96,1]), tf.float32)
        y_pred = tf.image.convert_image_dtype(tf.reshape(y_pred,[1,96,96,1]), tf.float32)
        
        # Calculate SSIM loss
        ssim_loss = 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1, filter_size=3))
    
        loss_value = beta*ssim_loss + (1-beta)*mae_loss
        
        return loss_value
    
    return loss


# SSIM loss
def ssim_loss(y_true, y_pred):
    """
    This function only works for batch size of 1. 
    """
    
    # Reshape tensors
    y_true = tf.image.convert_image_dtype(tf.reshape(y_true,[1,96,96,1]), tf.float32)
    y_pred = tf.image.convert_image_dtype(tf.reshape(y_pred,[1,96,96,1]), tf.float32)
    
    loss_value = 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1, filter_size=3))
    
    return loss_value


# SSIM metric
def ssim(y_true, y_pred):
    """
    This function only works for batch size of 1. 
    """
    
    # Reshape tensors
    y_true = tf.image.convert_image_dtype(tf.reshape(y_true,[1,96,96,1]), tf.float32)
    y_pred = tf.image.convert_image_dtype(tf.reshape(y_pred,[1,96,96,1]), tf.float32)
    
    metric_value = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1, filter_size=3))
    
    return metric_value


# PSNR metric 
def psnr(y_true, y_pred):
    """
    This function only works for batch size of 1. 
    """
    
    # Reshape tensors    
    y_true = tf.image.convert_image_dtype(tf.reshape(y_true,[1,96,96,1]), tf.float32)
    y_pred = tf.image.convert_image_dtype(tf.reshape(y_pred,[1,96,96,1]), tf.float32)
    
    metric_value = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1))
    
    return metric_value

# NMSE metric
def nmse(y_true, y_pred):
    
    metric_value = tf.keras.backend.mean(tf.keras.backend.square(y_pred-y_true))/tf.keras.backend.mean(tf.keras.backend.square(y_true))
    
    return metric_value

# nRMSE metric
def nrmse(y_true, y_pred):
    """
    This function only works for batch size of 1. 
    """
    
    # Reshape tensors    
    y_true = tf.image.convert_image_dtype(tf.reshape(y_true,[1,96,96,1]), tf.float32)
    y_pred = tf.image.convert_image_dtype(tf.reshape(y_pred,[1,96,96,1]), tf.float32)
    
    metric_value = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred-y_true)))/(tf.keras.backend.max(y_true)-tf.keras.backend.min(y_true))
    
    return metric_value