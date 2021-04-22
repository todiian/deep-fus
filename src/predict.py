"""
File:     deep-fus/src/predict.py
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
import tensorflow as tf
import json
import os
import scipy.io as sio
from utils import *
from losses import *

###################
# PRE-TRAINED MODEL
###################

model_dir = '../pretrained_models/ResUNet53D_125'
n_img = 125

#####################
# LOAD MODEL AND DATA
#####################

m = 40

# Load model 
model = tf.keras.models.load_model(model_dir +'/my_model.h5', custom_objects={'loss': custom_loss(beta=0.1), 'ssim': ssim, 'psnr': psnr, 'nmse': nmse, 'nrmse': nrmse})
    
# Load TEST examples
X_test, Y_test = load_dataset('test', n_img, m)

# Standardize X data - use mean and standard deviation of training set
Xmean = -0.5237595494149918
Xstd = 131526.6016974602

X_test = (X_test-Xmean) / Xstd

##################
# PREDICT AND PLOT
##################

# Predict TEST examples
Yhat_test = model.predict(X_test, verbose=0)
    
# Plot original and predicted TEST examples
plot_and_stats(Yhat_test, Y_test, model_dir)

#################
# SAVE .MAT FILES
#################

mat_dir = model_dir +'/mat_files'
if not os.path.exists(mat_dir):
    os.mkdir(mat_dir)

fr_data = {}
for idx in range(m):
    fr_data['y_true'] = Y_test[idx]
    fr_data['y_pred'] = Yhat_test[idx]
    
    # Save dataset
    fr_str = 'fr' +str(idx) +'.mat'
    data_dir = os.path.join(os.getcwd(), mat_dir, fr_str)
    sio.savemat(data_dir, fr_data)