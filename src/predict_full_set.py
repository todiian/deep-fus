"""
File:     deep-fus/src/predict_full_set.py
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
import tensorflow as tf
from utils import *
from losses import *

# Timer
from timeit import default_timer as timer

######################
# SELECT TRAINED MODEL
######################

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

################
# SELECT DATASET
################

# # VISUAL-EVOKED EXPERIMENT
# test_set = '../data/test_full_sets/visual_evoked'
# fr_start = 1
# fr_stop = 390
# n_frames = fr_stop-fr_start+1

# AWAKE FUS EXPERIMENT
test_set = '../data/test_full_sets/awake_fUS'
fr_start = 1
fr_stop = 499
n_frames = fr_stop-fr_start+1

###################
# RESULTS DIRECTORY
###################

res_dir = '../results/predict_fullset_' +str(n_img)

# Create results directory if not already existing
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
    print("Directory " , res_dir ,  " created ")
else:    
    print("Directory " , res_dir ,  " already existing")

#####################
# LOAD MODEL AND DATA
#####################

# Load model 
model = tf.keras.models.load_model(model_dir +'/my_model.h5', custom_objects={'custom_loss': custom_loss, 'ssim': ssim, 'psnr': psnr, 'nmse': nmse})
    
# Load TRAIN examples
X_train, Y_train = load_dataset('train', n_img)

print('Loading ' +str(n_frames) +' test examples.')

# Initialize output arrays
n_pix = 96
X_test = np.zeros((n_frames, n_pix, n_pix, n_img))
Y_test = np.zeros((n_frames, n_pix, n_pix))

for k in range(n_frames):
    
    # Load dataset
    data_dir = test_set +'/fr' +str(k+fr_start) +'.mat'
    mat_contents = sio.loadmat(data_dir)
    
    X_test[k] = mat_contents['x'][:,:,:n_img]
    Y_test[k] = mat_contents['y']
    
print('Done loading ' +str(n_frames) +' test examples.')

# Standardize X data
Xmean = np.mean(X_train)
Xstd = np.std(X_train)

X_test = (X_test-Xmean) / Xstd

#########
# PREDICT
#########

# Start timer
start = timer()

# End timer and print execution time
end = timer()
exec_time = end - start
print(exec_time)

# Predict TEST examples
Yhat_test = model.predict(X_test, verbose=0)

######
# PLOT
######

# Dynamic range [dB]
dr = 40

nmse = []
ssim = []
psnr = []
    
# Create dict to store metrics
metrics = {};
    
for idx in range(Yhat_test.shape[0]):        
    # Convert Yhat to dB scale
    Yhat_dB = 10*np.log10(Yhat_test[idx]/np.amax(Yhat_test[idx]))
        
    # Clip to dynamic range
    Yhat_dB[np.where(Yhat_dB<=-dr)] = -dr
    Yhat_dB[np.isnan(Yhat_dB)] = -dr

    # Plot Yhat
    fig, ax = plt.subplots()
    cs = ax.imshow(Yhat_dB, vmin=-dr, vmax=0, cmap='bone')
    cbar = fig.colorbar(cs)
    plt.show()
    plt.title('Predicted ' +str(idx))
    plt.savefig(res_dir +'/pred' +str(idx) +'.png')
    plt.close(fig)
        
    # NMSE
    nmse_tmp = tf.keras.backend.mean(tf.keras.backend.square(Yhat_test[idx]-Y_test[idx]))/tf.keras.backend.mean(tf.keras.backend.square(Y_test[idx]))
    nmse.append(nmse_tmp)
    
    # Prep for SSIM calc
    y_true = tf.convert_to_tensor(Y_test[idx])
    y_pred = tf.convert_to_tensor(Yhat_test[idx])
    
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

with open(res_dir +'/metrics', 'w') as file:
    json.dump(metrics, file)