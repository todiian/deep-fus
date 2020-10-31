"""
File:     deep-fus/src/models.py
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

import numpy as np
import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Concatenate, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import he_uniform


def deepfUS_5(input_shape = (96, 96, 250), reg_factor=0.5):
    """
    Implementation of a 5-layer U-Net model for reconstruction of functional ultrasound images from 
    sparse compounded data. 
    
    Convolutional blocks made of Conv2D + ReLU activation. Downsampling implemented with   
    MaxPooling2D. Upsampling implemented with Conv2DTranspose. Dropout used on convolutional blocks 
    of encoder L5 and on all the decoder layers.
    
    Arguments:
    input_shape -- dimensions of compounded dataset
    reg_factor -- regularization factor for dropout layers    
    
    Returns:
    model -- a Model() instance in Keras
    
    """
    
    # Filter kernel size used for convolutional layers
    filter_kernel = 3
    
    # Number of filters for respective layers
    F1 = 32
    F2 = 64
    F3 = 128
    F4 = 256
    F5 = 512
    
    # Input layer
    X_input = Input(input_shape, dtype=tf.float32, name="input")
    
    # Encoder L1
    X = Conv2D(filters=F1, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X_input)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F1, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    X_skip1 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L2
    X = Conv2D(filters=F2, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F2, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    X_skip2 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L3
    X = Conv2D(filters=F3, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F3, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    X_skip3 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L4
    X = Conv2D(filters=F4, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=F4, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    X_skip4 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L5
    X = Conv2D(filters=F5, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    X = Conv2D(filters=F5, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    X = Conv2DTranspose(filters=F4, kernel_size=(filter_kernel,filter_kernel), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L4
    X = Concatenate(axis=3)([X,X_skip4])
   
    X = Conv2D(filters=F4, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    X = Conv2D(filters=F4, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    X = Conv2DTranspose(filters=F3, kernel_size=(filter_kernel,filter_kernel), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L3
    X = Concatenate(axis=3)([X,X_skip3])
   
    X = Conv2D(filters=F3, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    X = Conv2D(filters=F3, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    X = Conv2DTranspose(filters=F2, kernel_size=(filter_kernel,filter_kernel), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L2
    X = Concatenate(axis=3)([X,X_skip2])
   
    X = Conv2D(filters=F2, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    X = Conv2D(filters=F2, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    X = Conv2DTranspose(filters=F1, kernel_size=(filter_kernel,filter_kernel), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L1    
    X = Concatenate(axis=3)([X,X_skip1])
   
    X = Conv2D(filters=F1, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    X = Conv2D(filters=F1, kernel_size=(filter_kernel,filter_kernel), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dropout(reg_factor)(X)
    
    # Output layer
    X = Conv2D(filters=1, kernel_size=(1,1), strides=1, padding='same')(X)
    
    # Reshape output
    X = Reshape((input_shape[0],input_shape[1]))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='deepfUS_5')
    
    return model