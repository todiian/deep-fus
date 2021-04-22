"""
File:     deep-fus/src/utils.py
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

# Import packages
import numpy as np
import tensorflow as tf
import json
import scipy.io as sio
import random
import os
import matplotlib.pyplot as plt

def load_dataset(dataset, n_img, m):
    """
    This function is used to load the training, validation, and test datasets.
    
    Arguments:
    dataset -- string for dataset. Accept: 'train', 'dev' or 'test'. Require set to be in the data folder
    n_img -- number of compounded RF images 
    m -- number of sets to load. Select m sets after random permutation
    
    Returns:
    set_x, set_y -- pairs of features (compounded RF) and labels (power Doppler image) for each  
                    dataset    
    """
    
    # The network was tested with images of 96x96 pixels. If this parameter is changed, the dimensions of train and dev examples must be changed accordingly
    n_pix = 96
    
    print('Loading ' +str(m) +' ' +dataset +' examples.')
    
    # Initialize output arrays
    set_x = np.zeros((m, n_pix, n_pix, n_img))
    set_y = np.zeros((m, n_pix, n_pix))
    
    data_list = [i for i in range(m)]
    
    # Shuffle set list
    np.random.seed(1)
    np.random.shuffle(data_list)
    
    for k in range(m):
        # Load dataset
        data_dir = '../data/' +dataset +'/fr' +str(k+1) +'.mat'
        mat_contents = sio.loadmat(data_dir)
        
        idx = data_list[k]
                
        set_x[idx] = mat_contents['x'][:,:,:n_img]
        set_y[idx] = mat_contents['y']
    
    print('    Done loading ' +str(m) +' ' +dataset +' examples.')
    
    return set_x, set_y



def plot_and_stats(Yhat, Y, model_dir):
    """
    This function is used for plotting the original and predicted frame, and their difference. 
    The function also calculates the following metrics:
    -- NMSE
    -- nRMSE
    -- SSIM
    -- PSNR
    
    Arguments:
    Yhat -- Predicted examples
    Y -- Original examples (ground truth)
    model_dir -- Path to the folder containing the file 'my_model.h5'
    
    Returns:
    --
                    
    """
    
    # Dynamic range [dB]
    dr = 40
    
    loc_dir = model_dir +'/plot_and_stats'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    
    nmse  = []
    nrmse = []
    ssim  = []
    psnr  = []
    
    # Create dict to store metrics
    metrics = {};
    
    for idx in range(np.minimum(Yhat.shape[0],50)):
        
        ###################
        # CALCULATE METRICS
        ###################
        
        # Prep for metric calc
        y_true = tf.convert_to_tensor(Y[idx])
        y_pred = tf.convert_to_tensor(Yhat[idx])
        
        y_true = tf.image.convert_image_dtype(tf.reshape(y_true,[1,96,96,1]), tf.float32)
        y_pred = tf.image.convert_image_dtype(tf.reshape(y_pred,[1,96,96,1]), tf.float32)
        
        # NMSE
        nmse_tmp = tf.keras.backend.mean(tf.keras.backend.square(y_pred-y_true))/tf.keras.backend.mean(tf.keras.backend.square(y_true))
        nmse_tmp = np.float_(nmse_tmp)
        nmse.append(nmse_tmp)
        
        # nRMSE
        nrmse_tmp = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred-y_true)))/(tf.keras.backend.max(y_true)-tf.keras.backend.min(y_true))
        nrmse.append(np.float_(nrmse_tmp))
        
        # SSIM
        ssim_tmp = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1, filter_size=3))
        ssim_tmp = np.float_(ssim_tmp)
        ssim.append(ssim_tmp)
        
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
        psnr_tmp = np.float_(psnr_tmp)
        psnr.append(psnr_tmp)
        
        ###########################
        # PLOT ORIG AND PRED FRAMES
        ###########################
        
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
        plt.title('Pred ' +str(idx) +' - SSIM: ' +'{:.03f}'.format(ssim_tmp) +' - PSNR: ' +'{:.03f}'.format(psnr_tmp) +' - NMSE: ' +'{:.03f}'.format(nmse_tmp) +' - NRMSE: ' +'{:.03f}'.format(nrmse_tmp) )
        plt.savefig(loc_dir +'/pred' +str(idx) +'.png')
        plt.close(fig)
        
        # Plot difference
        img_diff = np.abs(Yhat_dB-Y_dB)
        fig, ax = plt.subplots()
        cs = ax.imshow(img_diff, cmap='bone')
        cbar = fig.colorbar(cs)
        plt.show()
        plt.title('Difference ' +str(idx))
        plt.savefig(loc_dir +'/diff' +str(idx) +'.png')
        plt.close(fig)
        
        # Scatter plot
        y1 = np.copy(Y_dB)
        y2 = np.copy(Yhat_dB)
        fig, ax = plt.subplots()
        plt.scatter(y1.flatten(), y2.flatten(), marker='o', color='black')
        x = np.linspace(-40, 0, 41)
        plt.plot(x, x);
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.show()
        plt.savefig(loc_dir +'/scatt' +str(idx) +'.png')
        plt.close(fig)
        
    ######################
    # SAVE METRICS TO FILE
    ######################
    
    metrics["nmse"] = list(np.float_(nmse))
    metrics["nmse_mean"] = np.float_(np.mean(nmse))
    metrics["nmse_std"] = np.float_(np.std(nmse))
    
    metrics["nrmse"] = list(np.float_(nrmse))
    metrics["nrmse_mean"] = np.float_(np.mean(nrmse))
    metrics["nrmse_std"] = np.float_(np.std(nrmse))
    
    metrics["ssim"] = list(np.float_(ssim))
    metrics["ssim_mean"] = np.float_(np.mean(ssim))
    metrics["ssim_std"] = np.float_(np.std(ssim))
    
    metrics["psnr"] = list(np.float_(psnr))
    metrics["psnr_mean"] = np.float_(np.mean(psnr))
    metrics["psnr_std"] = np.float_(np.std(psnr))
    
    with open(loc_dir +'/metrics', 'w') as file:
        json.dump(metrics, file)
        
    return


def load_dataset_postproc(dataset, n_img, m):
    """
    This function is used to load the training, validation, and test datasets for the experiment
    using pre-processed power Doppler images.
    
    Arguments:
    dataset -- string for dataset. Accept: 'train', 'dev' or 'test'. Require set to be in the data folder
    n_img -- number of compounded RF images 
    m -- number of sets to load. Select m sets after random permutation
    
    Returns:
    set_x, set_y -- pairs of features (compounded RF) and labels (power Doppler image) for each  
                    dataset    
    """
    
    # The network was tested with images of 96x96 pixels. If this parameter is changed, the dimensions of train and dev examples must be changed accordingly
    n_pix = 96
    
    print('Loading ' +str(m) +' ' +dataset +' examples.')
    
    # Initialize output arrays
    set_x = np.zeros((m, n_pix, n_pix))
    set_y = np.zeros((m, n_pix, n_pix))
    
    data_list = [i for i in range(m)]
    
    # Shuffle set list
    np.random.seed(1)
    np.random.shuffle(data_list)
    
    for k in range(m):
        # Load dataset
        data_dir = '../data/' +dataset +'_process/' +str(n_img) +'img/fr' +str(k+1) +'.mat'
        mat_contents = sio.loadmat(data_dir)
        
        idx = data_list[k]
                
        set_x[idx] = mat_contents['x']
        set_y[idx] = mat_contents['y']
    
    print('    Done loading ' +str(m) +' ' +dataset +' examples.')
    
    return set_x, set_y