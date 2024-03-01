import os
import pickle
import h5py


import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow.experimental.numpy as tnp

tnp.experimental_enable_numpy_behavior()

from utils.generate_intermediate_values import multGF256

from gmpy2 import f_divmod_2exp

# Opening dataset specific variables 

file = open('utils/dataset_parameters','rb')
parameters = pickle.load(file)
file.close()


DATASET_FOLDER  = parameters['DATASET_FOLDER']
METRICS_FOLDER = DATASET_FOLDER + 'metrics_aihws/' 
MODEL_FOLDER = DATASET_FOLDER + 'models_aihws/' 
TRACES_FOLDER = DATASET_FOLDER + 'traces/'
REALVALUES_FOLDER = DATASET_FOLDER + 'real_values/'
POWERVALUES_FOLDER = DATASET_FOLDER + 'powervalues/'
TIMEPOINTS_FOLDER = DATASET_FOLDER + 'timepoints/'
KEY_FIXED = parameters['KEY']
FILE_DATASET = parameters['FILE_DATASET']
MASKS = parameters['MASKS']
ONE_MASKS = parameters['ONE_MASKS']
INTERMEDIATES = parameters['INTERMEDIATES']
VARIABLE_LIST = parameters['VARIABLE_LIST']
MASK_INTERMEDIATES = parameters['MASK_INTERMEDIATES']
MASKED_INTERMEDIATES = parameters['MASKED_INTERMEDIATES']




shift_rows_s = list([
    0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11
    ])



class PoolingCrop(tf.keras.layers.Layer):
    def __init__(self, input_dim=1, use_dropout = True,name = ''):
        if name == '':
            name = 'Crop_'+str(np.random.randint(0,high = 99999))
        super(PoolingCrop, self).__init__(name = name )
        self.w = self.add_weight(shape=(input_dim,1), dtype="float32",
                                  trainable=True,name = 'weights'+name,  regularizer = tf.keras.regularizers.L1L2(0.001)
                                  
        )
        self.input_dim = input_dim
        self.pooling = tf.keras.layers.AveragePooling1D(pool_size = 2,strides = 2,padding = 'same')
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = tf.keras.layers.AlphaDropout(0.001)
        self.bn = tf.keras.layers.BatchNormalization()
        
        
    
        
    def call(self, inputs):
        kernel = tf.multiply(self.w, inputs)       
        pooling = self.pooling(kernel)
        output = self.bn(pooling)
        if self.use_dropout:
            output = self.dropout(output)
        return output
    
    def get_config(self):
        config = {'w':self.w,
                  'input_dim' : self.input_dim,
                  'pooling' : self.pooling,
                  'dropout' : self.dropout
                  }
        base_config = super(PoolingCrop,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
class XorLayer(tf.keras.layers.Layer):
  def __init__(self,classes =256 ,name = ''):
    super(XorLayer, self).__init__(name = name)
    all_maps = np.zeros((classes,classes),dtype =np.uint8)
    for i in range(classes):
        for j in range(classes):
            all_maps[i, j ] = i^j    
    self.mapping2 = all_maps
    self.classes = classes
    
  def call(self, inputs):  
 
    pred1 = inputs[0]
    pred2 = tnp.asarray(inputs[1])
    
    p1 = pred1 
    p2 = pred2[:,self.mapping2]

    res = tf.einsum('ij,ijk->ik',p1,p2)
    return res

    
log_table=[ 0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
    100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
    125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
    101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
    150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
    102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
    126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
    43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
    175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
    44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
    127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
    204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
    151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
    83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
    68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
    103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7 ]

alog_table =[1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
    95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
    229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
    83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
    76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
    131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
    181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
    254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
    251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
    195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
    159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
    155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
    252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
    69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
    18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
    57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1 ]

# Multiplication function in GF(2^8)
def multGF256(a,b):
    if (a==0) or (b==0):
        return 0
    else:
        return alog_table[(log_table[a]+log_table[b]) %255]
    
class MultiLayer(tf.keras.layers.Layer):
    def __init__(self,classes = 256 ,name = ''):
        super(MultiLayer, self).__init__(name = name)
        all_maps = np.load('utils/mult_mapping.npy')
        mapping1 = []
        mapping2 = []
        for classe in range(classes):
            mapped = np.where(all_maps[classe] == 1)
            mapping1.append(mapped[0])
            mapping2.append(mapped[1])
        self.mapping1 = np.array(mapping1)
        self.mapping2 = np.array(mapping2)
        self.classes = classes
    
    def call(self, inputs):  
 
        pred1 = tnp.asarray(inputs[0])
        pred2 = tnp.asarray(inputs[1])
        p1 = pred1[:,self.mapping1]
        p2 = pred2[:,self.mapping2]
    
        res = tf.reduce_sum(tf.multiply(p1,p2),axis =2)   
        return res

    def get_config(self):
        config = {'mapping':self.mapping,
                  'classes':self.classes}
        base_config = super(MultiLayer,self).get_config()
        base_config.update(config)
        return base_config



def get_pow_rank(x):
    if x == 2 :
        return 1
    if x == 1 :
        return 0
    n = 1
    q  , r = f_divmod_2exp(x , n)    
    while q > 0:
        q  , r = f_divmod_2exp(x , n)
        n += 1
    return n

def get_rank(result,true_value):
    key_probabilities_sorted = np.argsort(result)[::-1]
    key_ranking_good_key = list(key_probabilities_sorted).index(true_value) + 1
    return key_ranking_good_key

def load_model_from_name(structure , name):
    model_file  = MODEL_FOLDER  + name
    print('Loading model {}'.format(model_file))
    structure.load_weights(model_file)
    return structure   



def adapt_plaintexts(p,k,fake_key):
    new_p = np.empty(p.shape,dtype = np.uint8)
    for trace  in range(len(p)):
        plaintext = int.from_bytes(p[trace],byteorder ='big')
        key = int.from_bytes(k[trace],byteorder ='big')        
        pxork = plaintext ^ key
        new_p[trace] = bytearray((pxork ^ fake_key).to_bytes(16,byteorder ='big'))
        assert int.from_bytes(new_p[trace],byteorder ='big') ^ fake_key ^ plaintext == key
    return new_p


def read_from_h5_file(n_traces = 1000,dataset = 'training',load_plaintexts = False):   
    
    f = h5py.File(DATASET_FOLDER + FILE_DATASET,'r')[dataset]  
    labels_dict = f['labels']
    if load_plaintexts:
        data =  {'keys':f['keys']   ,'plaintexts':f['plaintexts']}
        return  f['traces'][:n_traces] , labels_dict, data
    else:
        return  f['traces'][:n_traces] , labels_dict
def get_byte(i):
    for b in range(17,1,-1):
        if str(b) in i:
            return b
    return -1
    


def to_matrix(text):

    matrix = []
    for i in range(4):
        matrix.append([0,0,0,0])
    for i in range(len(text)):
        elem = text[i]
        matrix[i%4][i//4] =elem
    return matrix    








    

def load_dataset(byte,flat = False,whole = False,n_traces = None,dataset = 'training',encoded_labels = True,print_logs = True):    
    target = 't1'
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels for {}'.format(target)
        print(str_targets)
        
    traces , labels_dict  = read_from_h5_file(n_traces=n_traces,dataset = dataset)
    traces = np.expand_dims(traces,2)
    X_profiling_dict = {}  
    
    alpha = np.array(labels_dict['alpha'],dtype = np.uint8)[:n_traces]
    X_profiling_dict['alpha'] = get_hot_encode(alpha)
    if  whole:
        X_profiling_dict['traces'] = traces[:,4088:4088+1605+93*16]
        
    elif flat :
        intermediate_points = traces[:,4088+1605 :4088+1605 + 93*16]
        X_profiling_dict['inputs_intermediate'] = intermediate_points  
        X_profiling_dict['inputs_rin'] = traces[:,4088:4088+1605 ]                                 
    else:
        X_profiling_dict['inputs_intermediate'] = traces[:,4088+1605 + 93*byte:4088+1605 + 93*(byte+1) ] 
        X_profiling_dict['inputs_rin'] = traces[:,4088:4088+1605 ] 



    
    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=10000,dataset = 'validation')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  
        
        alpha = np.array(labels_dict_val['alpha'],dtype = np.uint8)[:10000]
        X_validation_dict['alpha'] = get_hot_encode(alpha)               
        if  whole:
            X_validation_dict['traces'] = traces_val[:,4088:4088+1605+93*16]
            
        elif flat :
            X_validation_dict['inputs_intermediate'] = traces_val[:,4088+1605:4088+1605 + 93*16 ] 
            X_validation_dict['inputs_rin'] = traces_val[:,4088:4088+1605 ]                                  
        else:
            X_validation_dict['inputs_intermediate'] = traces_val[:,4088+1605 + 93*byte:4088+1605 + 93*(byte+1) ] 
            X_validation_dict['inputs_rin'] = traces_val[:,4088:4088+1605 ] 

        

    Y_profiling_dict = {}

    permutations = np.array(labels_dict['p'],np.uint8)[:n_traces,byte]
    real_values_t1_temp = np.array(labels_dict[target],dtype = np.uint8)[:n_traces]
    real_values_t1  = np.array([real_values_t1_temp[i,permutations[i]] for i in range(len(real_values_t1_temp))])
    Y_profiling_dict['output'] = get_hot_encode(real_values_t1,classes = 256) if encoded_labels else  real_values_t1 
   
    if training:
        Y_validation_dict = {}
        permutations_val = np.array(labels_dict_val['p'],np.uint8)[:10000,byte]
        real_values_t1_temp_val = np.array(labels_dict_val[target],dtype = np.uint8)[:10000]
        real_values_t1_val  = np.array([real_values_t1_temp_val[i,permutations_val[i]] for i in range(len(real_values_t1_temp_val))])
        Y_validation_dict['output'] = get_hot_encode(real_values_t1_val,classes = 256 )   
        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)   
       

def load_dataset_second_order(byte,flat = False,whole = False,n_traces = None,dataset = 'training',encoded_labels = True,print_logs = True):    
    target = 't1'
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels for {}'.format(target)
        print(str_targets)
        
    traces , labels_dict  = read_from_h5_file(n_traces=n_traces,dataset = dataset)
    traces = np.expand_dims(traces,2)
    X_profiling_dict = {}  
    
    if  whole:
        X_profiling_dict['traces'] = traces
    elif flat:
        intermediate_points = traces[:,4088+1605 :4088+1605 + 93*16]
        X_profiling_dict['inputs_intermediate'] = intermediate_points
        X_profiling_dict['inputs_rin'] = traces[:,4088:4088+1605 ] 
        X_profiling_dict['inputs_alpha'] = traces[:,:4088 ] 
    else: 
        X_profiling_dict['inputs_intermediate'] = traces[:,4088+1605 + 93*byte:4088+1605 + 93*(byte+1) ] 
        X_profiling_dict['inputs_rin'] = traces[:,4088:4088+1605 ]   
        X_profiling_dict['inputs_alpha'] = traces[:,:4088 ] 
    
    
    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=10000,dataset = 'validation')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}              
        if  whole:
            X_validation_dict['traces'] = traces_val
            
        elif flat :
            X_validation_dict['inputs_intermediate'] = traces_val[:,4088+1605:4088+1605 + 93*16 ] 
            X_validation_dict['inputs_rin'] = traces_val[:,4088:4088+1605 ]
            X_validation_dict['inputs_alpha'] = traces_val[:,:4088 ]                                   
        else:
            X_validation_dict['inputs_intermediate'] = traces_val[:,4088+1605 + 93*byte:4088+1605 + 93*(byte+1) ] 
            X_validation_dict['inputs_rin'] = traces_val[:,4088:4088+1605 ] 
            X_validation_dict['inputs_alpha'] = traces_val[:,:4088 ] 

        

    Y_profiling_dict = {}

    permutations = np.array(labels_dict['p'],np.uint8)[:n_traces,byte]
    real_values_t1_temp = np.array(labels_dict[target],dtype = np.uint8)[:n_traces]
    real_values_t1  = np.array([real_values_t1_temp[i,permutations[i]] for i in range(len(real_values_t1_temp))])
    Y_profiling_dict['output'] = get_hot_encode(real_values_t1,classes = 256) if encoded_labels else  real_values_t1 
   
    if training:
        Y_validation_dict = {}
        permutations_val = np.array(labels_dict_val['p'],np.uint8)[:10000,byte]
        real_values_t1_temp_val = np.array(labels_dict_val[target],dtype = np.uint8)[:10000]
        real_values_t1_val  = np.array([real_values_t1_temp_val[i,permutations_val[i]] for i in range(len(real_values_t1_temp_val))])
        Y_validation_dict['output'] = get_hot_encode(real_values_t1_val,classes = 256 )   
        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)   
    

def load_dataset_multi(n_traces = None,flat = False,noperm = False,whole = False,dataset = 'training',encoded_labels = True,print_logs = True):
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels in order to train the multi-task model'
        print(str_targets)
        
    traces , labels_dict = read_from_h5_file(n_traces=n_traces,dataset = dataset)
    traces = np.expand_dims(traces,2)
    X_profiling_dict = {}  
    

    
    alpha = np.array(labels_dict['alpha'],dtype = np.uint8)[:n_traces]
    X_profiling_dict['alpha'] = get_hot_encode(alpha)
    if  whole:
        X_profiling_dict['traces'] = traces[:,4088:4088+1605+93*16]
    elif flat:
        intermediate_points = traces[:,4088+1605 :4088+1605 + 93*16]
        X_profiling_dict['inputs_intermediate'] = intermediate_points
        X_profiling_dict['inputs_rin'] = traces[:,4088:4088+1605 ] 
    else: 
        intermediate_points = np.empty((n_traces,93,16,1),dtype = np.int8)
        for byte in range(16):
            intermediate_points[:,:,byte] =traces[:,4088+1605 + 93*byte:4088+1605 + 93*(byte+1) ] 
        X_profiling_dict['inputs_intermediate'] = intermediate_points
        X_profiling_dict['inputs_rin'] = traces[:,4088:4088+1605 ]   
    
    all_permutations = np.array(labels_dict['p'],dtype = np.uint8)[:n_traces]

    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=10000,dataset = 'validation')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  

        if  whole:
            X_validation_dict['traces'] = traces_val[:,4088:4088+1605+93*16]
        elif flat:
            intermediate_points_val = traces_val[:,4088+1605 :4088+1605 + 93*16]
            X_validation_dict['inputs_intermediate'] = intermediate_points_val
            X_validation_dict['inputs_rin'] = traces_val[:,4088:4088+1605 ] 
        else: 
            intermediate_points_val = np.empty((10000,93,16,1),dtype = np.int8)
            for byte in range(16):
                intermediate_points_val[:,:,byte] =  traces_val[:,4088+1605 + 93*byte:4088+1605 + 93*(byte+1) ] 
            X_validation_dict['inputs_intermediate'] = intermediate_points
            X_validation_dict['inputs_rin'] = traces_val[:,4088:4088+1605 ]          
        
        alpha = np.array(labels_dict_val['alpha'],dtype = np.uint8)[:10000]
        X_validation_dict['alpha'] = get_hot_encode(alpha)
        

            
        
        all_permutations_val = np.array(labels_dict_val['p'],dtype = np.uint8)[:10000]
      




    if print_logs :
        print('Loaded inputs')    
        

    Y_profiling_dict = {}
    

    real_values_t1_temp = np.array(labels_dict['t1'],dtype = np.uint8)[:n_traces]
    for byte in range(16):
        permutations = all_permutations[:,byte]
        
        real_values_t1  = np.array([real_values_t1_temp[i,permutations[i]] for i in range(len(real_values_t1_temp))])
        Y_profiling_dict['output_t_{}'.format(byte)] = get_hot_encode(real_values_t1 if not noperm else real_values_t1_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_t1 
                                        
    if training:
        Y_validation_dict = {}
        real_values_t1_temp_val = np.array(labels_dict_val['t1'],dtype = np.uint8)[:10000]
        for byte in range(16):
            permutations_val = all_permutations_val[:,byte]
            
            real_values_t1_val  = np.array([real_values_t1_temp_val[i,permutations_val[i]] for i in range(len(real_values_t1_temp_val))])
            Y_validation_dict['output_t_{}'.format(byte)] = get_hot_encode(real_values_t1_val,classes = 256 )   if not noperm else get_hot_encode(real_values_t1_temp_val[:,byte],classes = 256 )




        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)    


def load_dataset_multi_second_order(n_traces = None,flat = False,noperm = False,whole = False,dataset = 'training',encoded_labels = True,print_logs = True):
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels in order to train the multi-task model'
        print(str_targets)
        
    traces , labels_dict = read_from_h5_file(n_traces=n_traces,dataset = dataset)
    traces = np.expand_dims(traces,2)
    print(traces.shape)
    X_profiling_dict = {}  
    

    

    if  whole:
        X_profiling_dict['traces'] = traces
    elif flat:
        intermediate_points = traces[:,4088+1605 :4088+1605 + 93*16]
        X_profiling_dict['inputs_intermediate'] = intermediate_points
        X_profiling_dict['inputs_rin'] = traces[:,4088:4088+1605 ] 
        X_profiling_dict['inputs_alpha'] = traces[:,:4088 ] 
    else: 
        intermediate_points = np.empty((n_traces,93,16,1),dtype = np.int8)
        inputs_permutations = np.empty((n_traces,93,16,1),dtype = np.int8) 
        for byte in range(16):
            intermediate_points[:,:,byte] =traces[:,4088+1605 + 93*byte:4088+1605 + 93*(byte+1) ]
            inputs_permutations[:,:,byte] =traces[:,4088+1605 +93*16+ 93*byte:4088+1605 +93*16+ 93*(byte+1) ] 
        X_profiling_dict['inputs_intermediate'] = intermediate_points
        X_profiling_dict['inputs_permutations'] = inputs_permutations
        X_profiling_dict['inputs_rin'] = traces[:,4088:4088+1605 ]   
        X_profiling_dict['inputs_alpha'] = traces[:,:4088 ] 
    all_permutations = np.array(labels_dict['p'],dtype = np.uint8)[:n_traces]

    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=20000,dataset = 'validation')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  

        if  whole:
            X_validation_dict['traces'] = traces_val
        elif flat:
            intermediate_points_val = traces_val[:,4088+1605 :4088+1605 + 93*16]
            X_validation_dict['inputs_intermediate'] = intermediate_points_val
            X_validation_dict['inputs_rin'] = traces_val[:,4088:4088+1605 ] 
            X_validation_dict['inputs_alpha'] = traces_val[:,:4088 ] 
        else: 
            intermediate_points_val = np.empty((20000,93,16,1),dtype = np.int8)
            inputs_permutations_val = np.empty((20000,93,16,1),dtype = np.int8)
            for byte in range(16):
                intermediate_points_val[:,:,byte] =  traces_val[:,4088+1605 + 93*byte:4088+1605 + 93*(byte+1) ] 
                inputs_permutations_val[:,:,byte] =  traces_val[:,4088+1605 +93*16+ 93*byte:4088+1605+93*16 + 93*(byte+1) ] 
            X_validation_dict['inputs_permutations'] = inputs_permutations_val
            X_validation_dict['inputs_intermediate'] = intermediate_points_val
            X_validation_dict['inputs_rin'] = traces_val[:,4088:4088+1605 ]   
            X_validation_dict['inputs_alpha'] = traces_val[:,:4088 ] 
        

        

            
        
        all_permutations_val = np.array(labels_dict_val['p'],dtype = np.uint8)[:20000]
      




    if print_logs :
        print('Loaded inputs')    
        

    Y_profiling_dict = {}
    

    real_values_t1_temp = np.array(labels_dict['t1'],dtype = np.uint8)[:n_traces]
    for byte in range(16):
        permutations = all_permutations[:,byte]
        
        real_values_t1  = np.array([real_values_t1_temp[i,permutations[i]] for i in range(len(real_values_t1_temp))])
        Y_profiling_dict['output_t_{}'.format(byte)] = get_hot_encode(real_values_t1_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_t1 
        #Y_profiling_dict['p_{}'.format(byte)] = get_hot_encode(permutations,classes = 16)
    if training:
        Y_validation_dict = {}
        real_values_t1_temp_val = np.array(labels_dict_val['t1'],dtype = np.uint8)[:20000]
        for byte in range(16):
            permutations_val = all_permutations_val[:,byte]
            
            
            Y_validation_dict['output_t_{}'.format(byte)] =  get_hot_encode(real_values_t1_temp_val[:,byte],classes = 256 )
            #Y_validation_dict['p_{}'.format(byte)] = get_hot_encode(permutations_val,classes = 16)



        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)   


def get_hw(k):
    hw = 0
    for _ in range(8):
        hw += k & 1
        k = k >> 1
    return hw 

def convert_to_binary(e):
    return [1 if e & (1 << (7-n)) else 0 for n in range(8)]   




def get_rank_list_from_prob_dist(probdist,l):
    res =  []
    accuracy = 0
    size = len(l)
    res_score = []
    accuracy_top5 = 0
    for i in range(size):
        rank = get_rank(probdist[i],l[i])
        res.append(rank)
        res_score.append(probdist[i][l[i]])
        accuracy += 1 if rank == 1 else 0
        accuracy_top5 += 1 if rank <= 5 else 0
    return res,(accuracy/size)*100,res_score , (accuracy_top5/size)*100


def get_hot_encode(label_set,classes = 256):    
    return np.eye(classes)[label_set]

