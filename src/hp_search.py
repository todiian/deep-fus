"""
File:     deep-fus/src/hp_search.py
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
import kerastuner as kt
from kerastuner import HyperModel
import json
import os
import IPython
import sys

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

# Logger class
class Logger(object):
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = open(file, "w")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass
    
    def close(self):
        self.log.close()

# Create and compile model
class myHyperModel(HyperModel):
    def __init__(self, n_img):
        self.n_img = n_img
        
    def build(self, hp):
    
        # Build model graph
        model = ResUNet53D(
            input_shape = (96, 96, self.n_img),
            do_rate=hp.Float('dropout_rate', 0, 0.9, step=0.1),
            filt0_k12=hp.Choice('filt0_k12', [1, 3]),
            filt0_k3=hp.Choice('filt0_k3', [4, 8, 16])
        )
    
        print(model.summary())
    
        # Adam optimizer
        opt = tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=1e-3, sampling='log'),
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
            loss=custom_loss(beta=hp.Float('beta', 0, 1, step=0.1)),
            metrics=mtrc
        ) 
    
        return model

def main(n_img):
    
    ############
    # INITIALIZE 
    ############
    
    # Create unique results folder
    today = datetime.now()
    res_dir = '../results/hp_search_ResUNet53D_' +str(n_img) +'/' +today.strftime('%Y%m%d') +'_' +today.strftime('%H%M%S')
    
    # Create results directory if not already existing
    os.makedirs(res_dir)
    print('Directory ' +res_dir +' succesfully created!')
    
    # Logger file
    log_file = res_dir +'/log_ResUNet53D_' +str(n_img) +'.txt'
    sys.stdout = Logger(log_file)
    
    # Print start processing time
    print('Start processing: ' +today.strftime('%Y%m%d') +' ' +today.strftime('%H%M%S'))
    
    n_epochs = 2500
    
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
    
    ###################
    # INSTANTIATE TUNER
    ###################
    hypermodel = myHyperModel(n_img=n_img)
    
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective=kt.Objective('val_ssim', direction='max'),
        num_initial_points=15,
        max_trials=15,
        executions_per_trial=1,
        seed=1,
        directory=res_dir,
        project_name='hp_opt'
    )
    
    # Print summary of search space
    tuner.search_space_summary()
    
    ################
    # EXECUTE SEARCH
    ################
    
    class ClearTrainingOutput(tf.keras.callbacks.Callback):
        def on_train_end(*args, **kwargs):
            IPython.display.clear_output(wait = True)
    
    results = tuner.search(
        X_train,
        Y_train,
        validation_data=(X_dev,Y_dev),
        epochs=n_epochs,
        batch_size=1,
    )
    # callbacks=[escb]
    
    # Print end processing time
    today = datetime.now()
    print('End processing: ' +today.strftime('%Y%m%d') +' ' +today.strftime('%H%M%S'))
    
    # Print summary of results
    tuner.results_summary()
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print('Hyperparameter search completed!')
    print('Optimal learning rate: ' +str(best_hps.get('learning_rate')))
    print('Optimal dropout rate: ' +str(best_hps.get('dropout_rate')))
    print('Optimal beta: ' +str(best_hps.get('beta')))
    print('Optimal filt0_k12: ' +str(best_hps.get('filt0_k12')))
    print('Optimal filt0_k3: ' + str(best_hps.get('filt0_k3')))
    
    # Log train and val losses and metrics
    train_log["learning_rate"] = best_hps.get('learning_rate')
    train_log["dropout_rate"] = best_hps.get('dropout_rate')
    train_log["beta"] = best_hps.get('beta')
    train_log["filt0_k12"] = best_hps.get('filt0_k12')
    train_log["filt0_k3"] = best_hps.get('filt0_k3')
    
    # Select best model
    model = tuner.get_best_models(num_models=1)[0]
    
    # Store search and train history to file
    with open(res_dir +'/search_train_history', 'w') as file:
        json.dump(train_log, file)
    
    # Save trained model
    model.save(res_dir +'/my_model.h5') 
    
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
    
    # Close logger
    sys.stdout.close()
    
    return

######################
# CALL MAIN TO EXECUTE
######################

main(125)