"""
File:     deep-fus/src/train.py
Author:   Tommaso Di Ianni (todiian@stanford.edu)

Copyright 2020 Tommaso Di Ianni

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
import os

"""
# Initialize random seeds for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)
tf.random.set_seed(1)
"""

# Import plot packages
import matplotlib.pyplot as plt

# Import other packages
from utils import *
from models import *
from losses import *

def train(res_dir, n_img=125, learn_rate=5e-6, loss_func='custom', reg_fact=0.5, train_epochs=20000):
    
    ######
    # INIT
    ######
    
    # Create dict to store training parameters and losses
    train_log = {}
    
    # Append training parameters to training log
    train_log["n_img"] = n_img
    train_log["learn_rate"] = learn_rate
    train_log["loss"] = loss_func
    train_log["epochs"] = train_epochs
    train_log["reg_fact"] = reg_fact
    
    # Create results directory if not already existing
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
        print("Directory " , res_dir ,  " created ")
    else:    
        print("Directory " , res_dir ,  " already existing")
        
    ##########
    # DATASETS
    ##########
    
    # Load TRAIN and DEV datasets
    X_train, Y_train = load_dataset('train', n_img)
    X_dev, Y_dev = load_dataset('dev', n_img)

    m, n_x, n_z, n_t = X_train.shape

    # Standardize X data
    Xmean = np.mean(X_train)
    Xstd = np.std(X_train)
    
    X_train = (X_train-Xmean) / Xstd
    X_dev = (X_dev-Xmean) / Xstd

    ##############
    # CREATE MODEL 
    ##############
    
    # Build model graph
    model = deepfUS_5(input_shape = (n_x, n_z, n_t))
    
    print(model.summary())
    
    # Adam optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')

    # Define the metrics
    mtrc = [ssim, psnr, nmse]
    
    # Compile model
    if loss_func=='custom':
        model.compile(optimizer=opt, loss=custom_loss, metrics=mtrc)
    elif loss_func=='ssim':
        model.compile(optimizer=opt, loss=ssim_loss, metrics=mtrc)
    elif loss_func=='mse':
        model.compile(optimizer=opt, loss=mse, metrics=mtrc)
    elif loss_func=='mae':
        model.compile(optimizer=opt, loss=mae, metrics=mtrc)
    
    #############
    # TRAIN MODEL
    #############
    
    results = model.fit(X_train, Y_train, validation_data=(X_dev,Y_dev), epochs=train_epochs, batch_size=1)
    
    # Save trained model
    model.save(res_dir +'/my_model.h5') 
    
    # Log train and val losses and metrics
    train_log["train_loss"] = list(np.float_(results.history["loss"]))
    train_log["train_ssim"] = list(np.float_(results.history["ssim"]))
    train_log["train_psnr"] = list(np.float_(results.history["psnr"]))
    train_log["train_nmse"] = list(np.float_(results.history["nmse"]))
    
    train_log["val_loss"] = list(np.float_(results.history["val_loss"]))
    train_log["val_ssim"] = list(np.float_(results.history["val_ssim"]))
    train_log["val_psnr"] = list(np.float_(results.history["val_psnr"]))
    train_log["val_nmse"] = list(np.float_(results.history["val_nmse"]))
        
    with open(res_dir +'/train_history', 'w') as file:
        json.dump(train_log, file)

    # Plot loss and defined metric
    plt.figure()
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.grid()
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper right')
    plt.savefig(res_dir +'/train_loss.png')
    
    # Clear graph
    tf.keras.backend.clear_session()    
        
    return