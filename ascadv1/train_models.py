# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:10:35 2021

@author: martho
"""


import argparse
import os
import numpy as np
import pickle
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Softmax, Dense, Input,Conv1D, AveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from multiprocessing import Process


# import dataset paths and variables
from utility import  METRICS_FOLDER , MODEL_FOLDER

# import custom layers
from utility import XorLayer , PoolingCrop

from utility import load_dataset, load_dataset_multi 
from tqdm import tqdm


seed = 42


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

###########################################################################


def model_single_task_xor(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length=1000, learning_rate=0.0001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    branch = input_layer_creation(inputs,input_length)
    inputs_dict['traces'] = inputs   
    
    branch = input_layer_creation(inputs,input_length)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)

    outputs = {}
      
    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True)
    intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False)
    
    xor = XorLayer(name ='xor' )([mask_branch,intermediate_branch]) 
    outputs['output'] = Softmax(name ='output')(xor)
    
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single_task_xor_{}'.format(name))

    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary() 
    return model


def model_single_task_twin(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length=1000, learning_rate=0.0001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    branch = input_layer_creation(inputs,input_length)
    inputs_dict['traces'] = inputs   
    
    
    mask_branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)
    mask_branch = dense_core(mask_branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True)
        
    intermediate_branch= cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)
    intermediate_branch = dense_core(intermediate_branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False)
    
    outputs = {}
    xor = XorLayer(name ='xor' )([mask_branch,intermediate_branch]) 
    outputs['output'] = Softmax(name ='output')(xor)
    
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single_task_xor_{}'.format(name))

    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary() 
    return model

def model_multi_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length=1000, learning_rate=0.0001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)

    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True)


    outputs = {} 

    for byte in range(2,16):
        
        intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False)
        
        xor = XorLayer(name ='xor_{}'.format(byte) )([intermediate_branch,mask_branch])        
        outputs['output_t_{}'.format(byte)] = Softmax(name = 'output_t_{}'.format(byte))(xor)
        
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary()
    return model   



def input_layer_creation(inputs,input_length,target_size = 25000,name = ''):

    size = input_length
    
    iteration  = 0
    crop = inputs
    
    while size > target_size:
        crop = PoolingCrop(input_dim = size,name = name)(crop)
        iteration += 1
        size = math.ceil(size/2)

    x = crop  
    return x



### Cnn for shared layers and mask/permutations single task models.

def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size):
    x = inputs_core
    for block in range(convolution_blocks):
        x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same')(x)    
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size = pooling_size)(x)
    
    output_layer = Flatten()(x) 

    return output_layer

def dense_core(inputs_core,dense_blocks,dense_units,batch_norm = False,activated = False):
    x = inputs_core    
    for block in range(dense_blocks):
        x = Dense(dense_units, activation='selu')(x)
        if batch_norm:
           x = BatchNormalization()(x)
    if activated:
        output_layer = Dense(256,activation ='softmax' )(x)  
    else:
        output_layer = Dense(256)(x)   
    return output_layer    


#### Training high level function
def train_model(training_type,byte,convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units):
    epochs = 100 
    batch_size = 250
    n_traces = 50000
    
    if 'single_task' in training_type:
        X_profiling , validation_data = load_dataset(byte,n_traces = n_traces,dataset = 'training')
        model_t = 'model_{}'.format(training_type) 
    else:
        X_profiling , validation_data = load_dataset_multi(n_traces = n_traces,dataset = 'training') 
        model_t = 'model_multi_task'
    
    window =  X_profiling.element_spec[0]['traces'].shape[0]
    monitor = 'val_accuracy'
    mode = 'max'

    if model_t == 'model_single_task_xor':
        model = model_single_task_xor(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = window)         
    elif model_t == 'model_single_task_twin':

        model = model_single_task_twin(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = window)         

    elif model_t == 'model_multi_task':
        
        model = model_multi_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = window)                  
        monitor = 'val_loss'   
        mode = 'min'        
    else:
        print('Some error here')

    
    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
    validation_data = validation_data.batch(batch_size)
    id_model = 'cb{}ks{}f{}s{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units)
    file_name = '{}_{}_{}'.format(model_t,byte if not model_t == 'model_multi_task' else 'all',id_model) 
    callbacks = tf.keras.callbacks.ModelCheckpoint(
                                filepath= MODEL_FOLDER+ file_name+'.h5',
                                save_weights_only=True,
                                monitor=monitor,
                                mode=mode,
                                save_best_only=True)

    

    
    history = model.fit(X_profiling, batch_size=batch_size, verbose = 1, epochs=epochs, validation_data=validation_data,callbacks =callbacks)
    print('Saved model {} ! '.format(file_name))
 
    file = open(METRICS_FOLDER+'history_training_'+(file_name ),'wb')
    pickle.dump(history.history,file)
    file.close()

    
    
    return model






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--SINGLE_TASK_TWIN', action="store_true", dest="SINGLE_TASK_TWIN",
                        help='Classical training of the intermediates', default=False)
    parser.add_argument('--SINGLE_TASK_XOR',   action="store_true", dest="SINGLE_TASK_XOR", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='Adding the masks to the labels', default=False)
    args            = parser.parse_args()
  

    SINGLE_TASK_TWIN        = args.SINGLE_TASK_TWIN
    SINGLE_TASK_XOR        = args.SINGLE_TASK_XOR
    MULTI = args.MULTI

    ALL = args.ALL

    TARGETS = {}
    if SINGLE_TASK_XOR:   
       training_types = ['single_task_xor']
    elif SINGLE_TASK_TWIN:
        training_types = ['single_task_twin']

    elif MULTI:
        training_types = ['multi_task']
    elif ALL:
        training_types = ['single_task_xor', 'single_task_twin' , 'multi_task']
    else:
        print('No training mode selected')

    for model_random in tqdm(range(50)):

        convolution_blocks = np.random.randint(1,3)
        kernel_size = sorted(np.random.randint(16,64,size = convolution_blocks))       
        filters = np.random.randint(3,16)
        strides = np.random.randint(2,30)
        pooling_size = np.random.randint(2,5)
        dense_blocks = np.random.randint(1,5)
        dense_units = (np.random.randint(64,512)//14) * 14 

        for training_type in training_types:
            if not training_type == 'multi_task':
                for byte in range(6,7):
                    process_eval = Process(target=train_model, args=(training_type,byte,convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units))
                    process_eval.start()
                    process_eval.join()
            else:
                process_eval = Process(target=train_model, args=(training_type,'all',convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units))
                process_eval.start()
                process_eval.join()                                    


    print("$ Done !")
            
        
        
