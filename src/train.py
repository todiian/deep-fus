"""
File:     deep-fus/src/train.py
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
import os

# Initialize random seeds for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)
tf.random.set_seed(1)

# Import plot packages
import matplotlib.pyplot as plt

# Import time packages
from datetime import datetime

# Import my packages
from utils import *
from models import *
from losses import *

# Create and compile model
def build_model(n_img):
    
    # Build model graph
    model = ResUNet53D(
        input_shape = (96, 96, n_img),
        do_rate=0.2,
        filt0_k12=3,
        filt0_k3=16
    )
    
    print(model.summary())
    
    # Adam optimizer
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.0005507175513654356,
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-07, 
        amsgrad=False, 
        name='Adam'
    )
    
    # Define the metrics
    mtrc = [ssim, psnr, nmse]
    
    # Compile model
    model.compile(
        optimizer=opt, 
        loss=custom_loss(beta=0.1),
        metrics=mtrc
    ) 
    
    return model

def main(n_img=125):
    
    print('Training ResUNet model with ' +str(n_img) +' images!')
    
    n_epochs = 2500
    
    ############
    # INITIALIZE 
    ############
    
    # Create unique results folder
    today = datetime.now()
    res_dir = '../results/train_ResUNet53D_' +str(n_img) +'/' +today.strftime('%Y%m%d') +'_' +today.strftime('%H%M%S')
    
    # Create results directory if not already existing
    os.makedirs(res_dir)
    print('Directory ' +res_dir +' succesfully created!')
    
    # Create dict to store training parameters
    train_log = {}
    
    # Append training parameters to training log
    train_log["n_img"] = n_img
    train_log["epochs"] = n_epochs
    
    ##########
    # DATASETS
    ##########
    
    # Load TRAIN and DEV datasets
    X_train, Y_train = load_dataset('train', n_img, 740)
    X_dev, Y_dev = load_dataset('dev', n_img, 40)
    
    # Standardize X data
    Xmean = np.mean(X_train)
    Xstd = np.std(X_train)
    
    X_train = (X_train-Xmean) / Xstd
    X_dev = (X_dev-Xmean) / Xstd
    
    ###########################
    # INITIALIZE CHECKPT SAVING
    ###########################
    
    checkpoint_path = res_dir +'/training/checkpoint'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_ssim',
        mode='max',
        save_best_only=True)
    
    #############
    # TRAIN MODEL
    #############
    
    # Create model build
    model = build_model(n_img)
    
    # Train
    results = model.fit(X_train, 
                        Y_train, 
                        validation_data=(X_dev,Y_dev), 
                        epochs=n_epochs, 
                        batch_size=1,
                        callbacks=[cp_callback])
    
    # Load weights from best performance checkpoint
    model.load_weights(checkpoint_path)
    
    # Save model
    model.save(res_dir +'/my_model.h5')
    
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
    # _ = plt.xscale('log')
    plt.grid()
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper right')
    plt.savefig(res_dir +'/train_loss.png')
    
    ###########################
    # PREDICT AND PLOT TEST SET
    ###########################
    
    # Load TEST examples
    X_test, Y_test = load_dataset('test', n_img, 40)
    X_test = (X_test-Xmean) / Xstd
    
    # Predict TEST examples
    Yhat_test = model.predict(X_test, verbose=0)
    
    # Plot original and predicted TEST examples
    plot_and_stats(Yhat_test, Y_test, res_dir)
    
    return

######################
# CALL MAIN TO EXECUTE
######################

main(125)