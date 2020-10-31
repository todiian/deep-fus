"""
File:     deep-fus/src/predict.py
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
import tensorflow as tf
from utils import *
from losses import *

<<<<<<< HEAD
# 125 COMPOUND FRAMES
model_dir = '../pretrained_models/deepfUS5_125'
n_img = 125

# # 63 COMPOUND FRAMES
# model_dir = '../pretrained_models/deepfUS5_63'
# n_img = 63

# # 25 COMPOUND FRAMES
# model_dir = '../pretrained_models/deepfUS5_25'
# n_img = 25

# # 13 COMPOUND FRAMES
# model_dir = '../pretrained_models/deepfUS5_13'
# n_img = 13

=======
model_dir = '../results/deepfUS5_125'
n_img = 125

>>>>>>> d4262e445e964cb51fc6f9e5cbfeb78c8b15c99a
# Load model 
model = tf.keras.models.load_model(model_dir +'/my_model.h5', custom_objects={'custom_loss': custom_loss, 'ssim': ssim, 'psnr': psnr, 'nmse': nmse})
    
# Load TRAIN examples
X_train, Y_train = load_dataset('train', n_img)
    
# Load TEST examples
<<<<<<< HEAD
X_test, Y_test = load_dataset('test', n_img)
=======
X_test, Y_test = load_dataset('train', n_img)
>>>>>>> d4262e445e964cb51fc6f9e5cbfeb78c8b15c99a

# Standardize X data
Xmean = np.mean(X_train)
Xstd = np.std(X_train)

X_test = (X_test-Xmean) / Xstd

# Predict prop TEST examples
Yhat_test = model.predict(X_test, verbose=0)
    
# Plot original and predicted TEST examples
plot_and_stats(Yhat_test, Y_test, model_dir)
