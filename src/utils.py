"""
File:     deep-fus/src/utils.py
Author:   Tommaso Di Ianni (todiian@stanford.edu)

<<<<<<< HEAD
Copyright 2020 Tommaso Di Ianni
=======
Copyright Tommaso Di Ianni 2020
>>>>>>> d4262e445e964cb51fc6f9e5cbfeb78c8b15c99a

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

# Import packages
import numpy as np
import tensorflow as tf
import json
import scipy.io as sio
import random
import os
import matplotlib.pyplot as plt

def load_dataset(dataset, n_img=64, m='all'):
    """
    This function is used to load the training, validation, and test datasets.
    
    Arguments:
    dataset -- string for dataset. Accept: 'train', 'dev' or 'test'. Require set to be in the 
               datasets folder
    n_img -- number of compounded RF images 
    m -- number of sets to load. Select m sets after random permutation
    
    Returns:
    set_x, set_y -- pairs of features (compounded RF) and labels (power Doppler image) for each  
                    dataset    
    """
    
    # The network was tested with images of 96x96 pixels. If this parameter is changed, the dimensions of train and dev examples must be changed accordingly
    n_pix = 96
    
    # Load set list
    with open('../datasets/' +dataset +'/datasets_list.txt', 'r') as f:
        data_list = f.read().splitlines()
    
    # Number of available examples
    m_avail = len(data_list)
    
    if m=='all':
        m = m_avail
    
    print('Loading ' +str(m) +' ' +dataset +' examples.')
    
    # Initialize output arrays
    set_x = np.zeros((m, n_pix, n_pix, n_img))
    set_y = np.zeros((m, n_pix, n_pix))
    
    # Shuffle set list
    random.Random(4).shuffle(data_list)
    
    # Select random subset of m sets from data_list
    data_list = data_list[:m]
    
    for idx, val in enumerate(data_list):
        
        # Load dataset
        data_dir = '../datasets/' +dataset +'/' +val
        mat_contents = sio.loadmat(data_dir)
        
        set_x[idx] = mat_contents['x'][:,:,:n_img]
        set_y[idx] = mat_contents['y']
    
    print('Done loading ' +str(m) +' ' +dataset +' examples.')
    
    return set_x, set_y


def plot_and_stats(Yhat, Y, model_dir):
    """
    This function is used for plotting the original and predicted frame, and their difference. 
    The function also calculates the following metrics:
    -- NMSE
    -- SSIM
    -- PSNR
    
    Arguments:
    Yhat -- Predicted examples
    Y -- Original examples (ground truth)
    model_dir -- path to the folder containing the file 'my_model.h5'
    
    Returns:
    --
                    
    """
    
    # Dynamic range [dB]
    dr = 40
    
    loc_dir = model_dir +'/plot_and_stats'
    if not os.path.exists(loc_dir):
        os.mkdir(loc_dir)
    
    nmse = []
    ssim = []
    psnr = []
    
    # Create dict to store metrics
    metrics = {};
    
    for idx in range(Yhat.shape[0]):
        # Convert Y to dB scale
        Y_dB = 10*np.log10(Y[idx]/np.amax(Y[idx]))
        
        # Clip to dynamic range
        Y_dB[np.where(Y_dB<=-dr)] = -dr
        Y_dB[np.isnan(Y_dB)] = -dr
        
        # Convert Yhat to dB scale
        Yhat_dB = 10*np.log10(Yhat[idx]/np.amax(Y[idx]))
        
        # Clip to dynamic range
        Yhat_dB[np.where(Yhat_dB<=-dr)] = -dr
        Yhat_dB[np.isnan(Yhat_dB)] = -dr
        
        # PLot Y
        fig, ax = plt.subplots()
        cs = ax.imshow(Y_dB, vmin=-dr, vmax=0, cmap='bone')
        cbar = fig.colorbar(cs)
        plt.show()
        plt.title('Original ' +str(idx))
        plt.savefig(loc_dir +'/orig' +str(idx) +'.png')
        plt.close(fig)

        # Plot Yhat
        fig, ax = plt.subplots()
        cs = ax.imshow(Yhat_dB, vmin=-dr, vmax=0, cmap='bone')
        cbar = fig.colorbar(cs)
        plt.show()
        plt.title('Predicted ' +str(idx))
        plt.savefig(loc_dir +'/pred' +str(idx) +'.png')
        plt.close(fig)
        
        # Plot difference
        img_diff = Yhat_dB-Y_dB
        fig, ax = plt.subplots()
        cs = ax.imshow(img_diff, cmap='bone')
        cbar = fig.colorbar(cs)
        plt.show()
        plt.title('Difference ' +str(idx))
        plt.savefig(loc_dir +'/diff' +str(idx) +'.png')
        plt.close(fig)
        
        # NMSE
        nmse_tmp = tf.keras.backend.mean(tf.keras.backend.square(Yhat[idx]-Y[idx]))/tf.keras.backend.mean(tf.keras.backend.square(Y[idx]))
        nmse.append(nmse_tmp)
        
        # Prep for SSIM calc
        y_true = tf.convert_to_tensor(Y[idx])
        y_pred = tf.convert_to_tensor(Yhat[idx])
        
        y_true = tf.image.convert_image_dtype(tf.reshape(y_true,[1,96,96,1]), tf.float32)
        y_pred = tf.image.convert_image_dtype(tf.reshape(y_pred,[1,96,96,1]), tf.float32)
        
        # SSIM
        ssim_tmp = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1, filter_size=3))
        ssim.append(np.float_(ssim_tmp))
        
        # Prep for PSNR calc
        y_pred = tf.divide(y_pred,tf.reduce_max(y_true))    # Normalize y_pred [0 1]        
        y_pred = tf.clip_by_value(y_pred, np.power(10,-dr/10), 1)         # Clip to dynamic range
        y_pred = tf.multiply(tf.divide(tf.math.log(y_pred), tf.math.log(tf.constant(10, dtype=y_true.dtype))), 10)   
        y_pred = (y_pred+dr)/dr
        
        y_true = tf.divide(y_true,tf.reduce_max(y_true))    # Normalize y_true [0 1]
        y_true = tf.clip_by_value(y_true, np.power(10,-dr/10), 1)          # Clip to dynamic range
        y_true = tf.multiply(tf.divide(tf.math.log(y_true), tf.math.log(tf.constant(10, dtype=y_true.dtype))), 10)
        y_true = (y_true+dr)/dr
        
        # PSNR
        psnr_tmp = tf.image.psnr(y_true, y_pred, max_val=1)
        psnr.append(np.float_(psnr_tmp))        
    
    metrics["nmse"] = list(np.float_(nmse))
    metrics["nmse_mean"] = np.float_(np.mean(nmse))
    metrics["nmse_std"] = np.float_(np.std(nmse))
    
    metrics["ssim"] = list(np.float_(ssim))
    metrics["ssim_mean"] = np.float_(np.mean(ssim))
    metrics["ssim_std"] = np.float_(np.std(ssim))
    
    metrics["psnr"] = list(np.float_(psnr))
    metrics["psnr_mean"] = np.float_(np.mean(psnr))
    metrics["psnr_std"] = np.float_(np.std(psnr))
    
    with open(loc_dir +'/metrics', 'w') as file:
        json.dump(metrics, file)
        
    return