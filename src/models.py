"""
File:     deep-fus/src/models.py
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

import numpy as np
import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Activation, MaxPooling2D, Dropout, Concatenate, Conv2DTranspose, Reshape, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import he_uniform


def UNet5(input_shape = (96, 96, 125), do_rate=0.5):
    """
    Implementation of a 5-layer U-Net model for the reconstruction of power Doppler  images 
    from sparse compound data.
    
    Convolutional blocks are made of Conv2D + ReLU activations. 
    Downsampling is implemented with MaxPooling2D. 
    Upsampling is implemented with Conv2DTranspose. 
    Dropout used on all convolutional blocks.
    
    Arguments:
    input_shape -- dimensions of compounded dataset
    do_rate -- dropout factor   
    
    Returns:
    model -- a Model() instance in Keras
    
    """    
    
    def conv_block(X_in, nf, k, dr):
    
        X = Conv2D(filters=nf, kernel_size=(k,k), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X_in)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
    
        X = Conv2D(filters=nf, kernel_size=(k,k), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
    
        return X
    
    # Filter kernel size used for convolutional layers
    filt_k = 3
    
    # Number of filters for respective layers
    F1 = 32
    F2 = 64
    F3 = 128
    F4 = 256
    F5 = 512
    
    # Input layer
    X_input = Input(input_shape, dtype=tf.float32, name="input")
    
    # Encoder L1
    X = conv_block(X_input, F1, filt_k, do_rate)
    
    X_skip1 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L2
    X = conv_block(X, F2, filt_k, do_rate)
    
    X_skip2 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L3
    X = conv_block(X, F3, filt_k, do_rate)
    
    X_skip3 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L4
    X = conv_block(X, F4, filt_k, do_rate)
    
    X_skip4 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L5
    X = conv_block(X, F5, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F4, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L4
    X = Concatenate(axis=3)([X,X_skip4])
   
    X = conv_block(X, F4, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F3, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L3
    X = Concatenate(axis=3)([X,X_skip3])
   
    X = conv_block(X, F3, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F2, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L2
    X = Concatenate(axis=3)([X,X_skip2])
   
    X = conv_block(X, F2, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F1, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L1    
    X = Concatenate(axis=3)([X,X_skip1])
   
    X = conv_block(X, F1, filt_k, do_rate)
    
    # Output layer
    X = Conv2D(filters=1, kernel_size=(1,1), strides=1, padding='same')(X)
    
    # Reshape output
    X = Reshape((input_shape[0],input_shape[1]))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='UNet5')
    
    return model



def UNet53D(input_shape = (96, 96, 125), do_rate=0.5, filt0_k12=3, filt0_k3=8):
    """
    Implementation of a 5-layer U-Net model for the reconstruction of power Doppler images 
    from sparse compound data. This model includes an initial Conv3D block that extracts 
    spatiotemporal features.
    
    Convolutional blocks are made of Conv2D + ReLU activations. 
    Downsampling is implemented with MaxPooling2D. 
    Upsampling is implemented with Conv2DTranspose. 
    Dropout used on all convolutional blocks.
    
    Arguments:
    input_shape -- dimensions of compound dataset
    do_rate -- dropout factor
    filt0_k12 -- first and second kernel dimensions for first Conv3D layer
    filt0_k3 -- third kernel dimension of first Conv3D layer
    
    Returns:
    model -- a Model() instance in Keras
    
    """
    
    def conv_block(X_in, nf, k, dr):
    
        X = Conv2D(filters=nf, kernel_size=(k,k), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X_in)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
    
        X = Conv2D(filters=nf, kernel_size=(k,k), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
    
        return X
    
    # Filter kernel size used for convolutional layers
    filt_k = 3
    
    # Number of filters for respective layers
    F0 = 4
    
    F1 = 32
    F2 = 64
    F3 = 128
    F4 = 256
    F5 = 512
    
    # Input layer
    X_input = Input(input_shape, dtype=tf.float32, name="input")
    
    # Reshape input to 4-D
    X = Reshape((input_shape[0],input_shape[1],input_shape[2],1))(X_input)
    
    # Conv3D layer
    X = Conv3D(filters=F0, kernel_size=(filt0_k12,filt0_k12,filt0_k3), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    # Reshape input to 3-D
    X = Reshape((input_shape[0],input_shape[1],input_shape[2]*F0))(X)
    
    # Encoder L1
    X = conv_block(X, F1, filt_k, do_rate)
    
    X_skip1 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L2
    X = conv_block(X, F2, filt_k, do_rate)
    
    X_skip2 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L3
    X = conv_block(X, F3, filt_k, do_rate)
    
    X_skip3 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L4
    X = conv_block(X, F4, filt_k, do_rate)
    
    X_skip4 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L5
    X = conv_block(X, F5, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F4, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L4
    X = Concatenate(axis=3)([X,X_skip4])
   
    X = conv_block(X, F4, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F3, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L3
    X = Concatenate(axis=3)([X,X_skip3])
   
    X = conv_block(X, F3, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F2, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L2
    X = Concatenate(axis=3)([X,X_skip2])
   
    X = conv_block(X, F2, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F1, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L1    
    X = Concatenate(axis=3)([X,X_skip1])
   
    X = conv_block(X, F1, filt_k, do_rate)
    
    # Output layer
    X = Conv2D(filters=1, kernel_size=(1,1), strides=1, padding='same')(X)
    
    # Reshape output
    X = Reshape((input_shape[0],input_shape[1]))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='UNet53D')
    
    return model



def ResUNet53D(input_shape = (96, 96, 125), do_rate=0.5, filt0_k12=3, filt0_k3=8):
    """
    Implementation of a 5-layer U-Net model with residual blocks for the reconstruction of power 
    Doppler images from sparse compound data. This model includes an initial Conv3D block that 
    extracts spatiotemporal features.
    
    Residual blocks are made of Conv2D + ReLU activations. 
    Downsampling is implemented with MaxPooling2D. 
    Upsampling is implemented with Conv2DTranspose. 
    Dropout used on all residual blocks.
    
    Arguments:
    input_shape -- dimensions of compound dataset
    do_rate -- dropout factor
    filt0_k12 -- first and second kernel dimensions for first Conv3D layer
    filt0_k3 -- third kernel dimension of first Conv3D layer
    
    Returns:
    model -- a Model() instance in Keras
    
    """
    
    def res_block(X_in, nf, k, dr):
    
        X = Conv2D(filters=nf, kernel_size=(k,k), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X_in)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
    
        X = Conv2D(filters=nf, kernel_size=(k,k), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
        
        X_out = Add()([X, X_in])
    
        return X_out
    
    # Filter kernel size used for convolutional layers
    filt_k = 3
    
    # Number of filters for respective layers
    F0 = 4
    
    F1 = 32
    F2 = 64
    F3 = 128
    F4 = 256
    F5 = 512
    
    # Input layer
    X_input = Input(input_shape, dtype=tf.float32, name="input")
    
    # Reshape input to 4-D
    X = Reshape((input_shape[0],input_shape[1],input_shape[2],1))(X_input)
    
    # Conv3D layer
    X = Conv3D(filters=F0, kernel_size=(filt0_k12,filt0_k12,filt0_k3), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    X = Activation('relu')(X)
    
    # Reshape input to 3-D
    X = Reshape((input_shape[0],input_shape[1],input_shape[2]*F0))(X)
    
    # Encoder L1
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    
    X = res_block(X, F1, filt_k, do_rate)
    
    X_skip1 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L2
    X = Conv2D(filters=F2, kernel_size=(1,1), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    
    X = res_block(X, F2, filt_k, do_rate)
    
    X_skip2 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L3
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    
    X = res_block(X, F3, filt_k, do_rate)
    
    X_skip3 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L4
    X = Conv2D(filters=F4, kernel_size=(1,1), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    
    X = res_block(X, F4, filt_k, do_rate)
    
    X_skip4 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L5
    X = Conv2D(filters=F5, kernel_size=(1,1), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    
    X = res_block(X, F5, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F4, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L4
    X = Concatenate(axis=3)([X,X_skip4])
   
    X = Conv2D(filters=F4, kernel_size=(1,1), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    
    X = res_block(X, F4, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F3, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L3
    X = Concatenate(axis=3)([X,X_skip3])
   
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    
    X = res_block(X, F3, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F2, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L2
    X = Concatenate(axis=3)([X,X_skip2])
   
    X = Conv2D(filters=F2, kernel_size=(1,1), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    
    X = res_block(X, F2, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F1, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L1    
    X = Concatenate(axis=3)([X,X_skip1])
   
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
    
    X = res_block(X, F1, filt_k, do_rate)
    
    # Output layer
    X = Conv2D(filters=1, kernel_size=(1,1), strides=1, padding='same')(X)
    
    # Reshape output
    X = Reshape((input_shape[0],input_shape[1]))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResUNet53D')
    
    return model



def UNet5_postproc(input_shape = (96, 96), do_rate=0.5):
    """
    Implementation of a 5-layer U-Net model for the post-processing of power Doppler images 
    reconstructed using sparse compound data.
    
    Convolutional blocks are made of Conv2D + ReLU activations. 
    Downsampling is implemented with MaxPooling2D. 
    Upsampling is implemented with Conv2DTranspose. 
    Dropout used on all convolutional blocks.
    
    Arguments:
    input_shape -- dimensions of compounded dataset
    do_rate -- dropout factor   
    
    Returns:
    model -- a Model() instance in Keras
    
    """    
    
    def conv_block(X_in, nf, k, dr):
    
        X = Conv2D(filters=nf, kernel_size=(k,k), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X_in)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
    
        X = Conv2D(filters=nf, kernel_size=(k,k), strides=1, padding='same', kernel_initializer=he_uniform(seed=0))(X)
        X = Activation('relu')(X)
        X = Dropout(dr)(X)
    
        return X
    
    # Filter kernel size used for convolutional layers
    filt_k = 3
    
    # Number of filters for respective layers
    F1 = 32
    F2 = 64
    F3 = 128
    F4 = 256
    F5 = 512
    
    # Input layer
    X_input = Input(input_shape, dtype=tf.float32, name="input")
    
    # Reshape input to 4-D
    X = Reshape((input_shape[0],input_shape[1],1))(X_input)
    
    # Encoder L1
    X = conv_block(X, F1, filt_k, do_rate)
    
    X_skip1 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L2
    X = conv_block(X, F2, filt_k, do_rate)
    
    X_skip2 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L3
    X = conv_block(X, F3, filt_k, do_rate)
    
    X_skip3 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L4
    X = conv_block(X, F4, filt_k, do_rate)
    
    X_skip4 = X
    
    X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)
    
    # Encoder L5
    X = conv_block(X, F5, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F4, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L4
    X = Concatenate(axis=3)([X,X_skip4])
   
    X = conv_block(X, F4, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F3, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L3
    X = Concatenate(axis=3)([X,X_skip3])
   
    X = conv_block(X, F3, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F2, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L2
    X = Concatenate(axis=3)([X,X_skip2])
   
    X = conv_block(X, F2, filt_k, do_rate)
    
    X = Conv2DTranspose(filters=F1, kernel_size=(filt_k,filt_k), strides=(2,2), padding='same')(X)
    X = Activation('relu')(X)
    
    # Decoder L1    
    X = Concatenate(axis=3)([X,X_skip1])
   
    X = conv_block(X, F1, filt_k, do_rate)
    
    # Output layer
    X = Conv2D(filters=1, kernel_size=(1,1), strides=1, padding='same')(X)
    
    # Reshape output
    X = Reshape((input_shape[0],input_shape[1]))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='UNet5_postproc')
    
    return model