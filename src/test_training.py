"""
File:     deep-fus/src/test_training.py
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

import os
from train import *

N_images = 125
N_epochs = 2
learning_rate = 5e-6
loss = 'custom'
dropout = 0.5

res_dir = '../results/trained_deepfUS5_' +str(N_images)
    
train(res_dir, n_img=N_images, learn_rate=learning_rate, loss_func=loss, reg_fact=dropout, train_epochs=N_epochs)