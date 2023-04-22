from utility import read_from_h5_file , adapt_plaintexts , get_hot_encode , load_model_from_name , get_rank , get_pow_rank
from utility import XorLayer 
from utility import METRICS_FOLDER , MODEL_FOLDER
from gmpy2 import mpz,mul

from train_models_third_order import   model_single_task_xor_whole , model_single_task_xor_extracted , model_single_task_xor_flat   , model_multi_task_whole, model_multi_task_extracted, model_multi_task_flat

import argparse , parse
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pickle 
import os

class Attack:
    def __init__(self,training_type,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units,n_experiments = 1000,n_traces = 50000,model_type = 'multi_task'):
        
        self.models = {}
        self.n_experiments = n_experiments
        self.n_traces = n_traces
        self.powervalues = {}

        traces , labels_dict, metadata  = read_from_h5_file(n_traces = self.n_traces,dataset = 'attack',load_plaintexts = True)
        traces = np.expand_dims(traces,2)

        self.correct_guesses = {}
        self.history_score = {}
        self.traces_per_exp = 1000
        self.predictions = np.zeros((16,self.n_traces,256),dtype =np.float32)
        
        
        plaintexts = np.array(metadata['plaintexts'],dtype = np.uint8)[:self.n_traces]
        keys =  np.array(metadata['keys'],dtype = np.uint8)[:self.n_traces]
        self.key = 0x00112233445566778899AABBCCDDEEFF
        master_key =[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                      0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF ]  
          
        
        self.permutations = get_hot_encode(np.array(labels_dict['p'],dtype = np.uint8)[:self.n_traces],classes = 16)
        self.permutations = np.swapaxes(self.permutations,1,0)
        
        self.plaintexts = get_hot_encode(adapt_plaintexts(plaintexts,keys,self.key))
        batch_size = self.n_traces//10
  

        predictions_non_permuted= np.zeros((16,self.n_traces,256),dtype =np.float32)
        

        id_model  = 'cb{}ks{}f{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units)
        if model_type == 'multi_task':
            multi_name = '{}_{}_{}_third.h5'.format('model_{}_{}'.format(model_type,training_type), 'all',id_model) 
            X_multi = {}
            if training_type == 'extracted':
                model_struct = model_multi_task_extracted(convolution_blocks ,dense_blocks, kernel_size,filters , pooling_size,dense_units,summary = False)
                intermediate_points= np.empty((self.n_traces ,93,16,1),dtype = np.int8)
                for byte in range(16):
                    intermediate_points[:,:,byte] = traces[:,4088+1605 +93*byte:4088+1605 +93*(byte+1)]
                X_multi['inputs_intermediate'] = intermediate_points
                X_multi['inputs_alpha'] = traces[:,:4088]
                X_multi['inputs_rin'] = traces[:,4088:4088+1605]
            elif training_type == 'flat':
                model_struct = model_multi_task_flat(convolution_blocks ,dense_blocks, kernel_size,filters , pooling_size,dense_units,summary = False)
                X_multi['inputs_intermediate'] = traces[:,4088+1605:4088+1605 +93*16]
                X_multi['alpha'] =self.alpha
                X_multi['inputs_rin'] = traces[:,4088:4088+1605]
            elif training_type == 'whole':
                model_struct = model_multi_task_whole(convolution_blocks ,dense_blocks, kernel_size,filters , pooling_size,dense_units,summary = False)
                X_multi['traces'] = traces[:,4088:4088+1605 +93*16]
                X_multi['alpha'] =self.alpha
            else:
                print('error')
            self.models['all'] = load_model_from_name(model_struct,multi_name)             
            all_predictions = {}         
            all_predictions = self.models['all'].predict(X_multi,verbose=1 ,batch_size = 1000)   
            for byte in range(16):
                predictions_non_permuted[byte] = all_predictions['output_t_{}'.format(byte)]
        else:
            for byte in range(16):
                name = '{}_{}_{}_third.h5'.format('model_{}_{}'.format(model_type,training_type),byte ,id_model) 
                X_single = {} 

                if training_type == 'extracted':
                    model_struct = model_single_task_xor_extracted(convolution_blocks ,dense_blocks, kernel_size,filters , pooling_size,dense_units,summary = False)
                    X_single['inputs_intermediate'] = traces[:,4088+1605 +93*byte:4088+1605 +93*(byte+1)]        
                    X_single['inputs_alpha'] = traces[:,:4088] 
                    X_single['inputs_rin'] = traces[:,4088:4088+1605]                   

                elif training_type == 'flat':
                      model_struct = model_single_task_xor_flat(convolution_blocks ,dense_blocks, kernel_size,filters , pooling_size,dense_units,summary = False) 
                      X_single['inputs_intermediate'] = traces[:,4088+1605 :4088+1605 +93*16]        
                      X_single['alpha'] = self.alpha
                      X_single['inputs_rin'] = traces[:,4088:4088+1605] 
                elif training_type == 'whole':
                    model_struct = model_single_task_xor_whole(convolution_blocks ,dense_blocks, kernel_size,filters , pooling_size,dense_units,summary = False) 
                    X_single['alpha'] = self.alpha
                    X_single['traces'] = traces[:,4088:4088+1605+93*16] 
                
                else:
                    print('error')

                self.models[byte] = load_model_from_name(model_struct,name)             
                predictions_non_permuted[byte] = self.models[byte].predict(X_single,verbose=1 ,batch_size = 1000)['output']   

     
                
        for batch in tqdm(range(self.n_traces// batch_size)):        
            for byte in range(16):
                for byte_perm in range(16):
                    self.predictions[byte_perm][batch_size*batch:batch_size*(batch +1)] = tf.add(self.predictions[byte_perm,batch_size*batch:batch_size*(batch +1)], tf.expand_dims(tf.cast(self.permutations[byte,batch_size*batch:batch_size*(batch +1),byte_perm],tf.float32),1) * predictions_non_permuted[byte,batch_size*batch:batch_size*(batch +1)] ) 
                               
        for batch in tqdm(range(self.n_traces// batch_size)):
            for byte in range(16):                   
        
                self.predictions[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([self.predictions[byte,batch_size*batch:batch_size*(batch +1)],self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])

        master_key = np.array(master_key,dtype = np.int32)
        self.subkeys = master_key
        
        
        


        
    def run(self,typ,id_model,print_logs = False):
       history_score = {}
       for experiment in tqdm(range(self.n_experiments)):
           if print_logs:
               print('====================')
               print('Experiment {} '.format(experiment))
           history_score[experiment] = {}
           history_score[experiment]['total_rank'] =  [] 
           subkeys_guess = {}
           for i in range(16):
               subkeys_guess[i] = np.zeros(256,)            
           
               history_score[experiment][i] = []
           traces_order = np.random.permutation(self.n_traces)[:self.traces_per_exp] 
           count_trace = 1
           
           for trace in traces_order:
               
               
               
               recovered  = {}
               all_recovered = True
               ranks = {}
               if print_logs:
                   print('========= Trace {} ========='.format(count_trace))
               rank_string = ""
               total_rank = mpz(1)
               for byte in range(16):
                   subkeys_guess[byte] += np.log(self.predictions[byte][trace] + 1e-36)
                  
                   ranks[byte] = get_rank(subkeys_guess[byte],self.subkeys[byte])
                   history_score[experiment][byte].append(ranks[byte])
                   total_rank = mul(total_rank,mpz(ranks[byte]))
                   rank_string += "| rank for byte {} : {} | \n".format(byte,ranks[byte])
                   if np.argmax(subkeys_guess[byte]) == self.subkeys[byte]:
                       recovered[byte] = True                        
                   else:
                       recovered[byte] = False
                       all_recovered = False                
              
               history_score[experiment]['total_rank'].append(get_pow_rank(total_rank))
               if print_logs:
                   print(rank_string)
                   print('Total rank 2^{}'.format( history_score[experiment]['total_rank'][-1]))
                   print('\n')
               if all_recovered:  
                   if print_logs:
                       
                       print('All bytes Recovered at trace {}'.format(count_trace))
                   
                   for elem in range(count_trace,self.traces_per_exp):
                       for i in range(16):
                           history_score[experiment][byte].append(ranks[byte])
                       history_score[experiment]['total_rank'].append(1)
                   break
                   count_trace += 1
               else:
                   count_trace += 1
               if print_logs:
                   print('\n')
           if not all_recovered:
                 for fake_experiment in range(self.n_experiments):
                     history_score[fake_experiment] = {}
                     history_score[fake_experiment]['total_rank'] =  [] 
                     for byte in range(16):
                         history_score[fake_experiment][byte] = []
                     for elem in range(self.traces_per_exp):
                         for byte in range(16):
                             history_score[fake_experiment][byte].append(128)
                         history_score[fake_experiment]['total_rank'].append(128)
                 break
       array_total_rank = np.empty((self.n_experiments,self.traces_per_exp))
       for i in range(self.n_experiments):
           for j in range(self.traces_per_exp):
               array_total_rank[i][j] =  history_score[i]['total_rank'][j] 
       whe = np.where(np.mean(array_total_rank,axis=0) < 2)[0]
       print(typ)
       print('GE < 2 : ',(np.min(whe) if whe.shape[0] >= 1 else self.traces_per_exp))        

       file = open(METRICS_FOLDER + 'history_attack_experiments_{}_{}_{}_third'.format(typ,id_model,self.n_experiments),'wb')
       pickle.dump(history_score,file)
       file.close()


                
   
def run_attack(training_type,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units,model_type):                
    attack = Attack(training_type,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units,model_type = model_type)
    id_model  = 'cb{}ks{}f{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units)
    attack.run('{}'.format(model_type),id_model)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--SINGLE_TASK', action="store_true", dest="SINGLE_TASK",
                        help='Single task models', default=False)
    parser.add_argument('--SINGLE_TASK_XOR',   action="store_true", dest="SINGLE_TASK_XOR", help='Single task xor mdoels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Multi learning models', default=False)
    parser.add_argument('-scenario',   action="store", dest="TRAINING_TYPE", help='choose the input scenario', default='extracted')
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='All model types', default=False)
        
    args            = parser.parse_args()
  

    SINGLE_TASK        = args.SINGLE_TASK
    SINGLE_TASK_XOR        = args.SINGLE_TASK_XOR
    MULTI = args.MULTI
    ALL = args.ALL
    TRAINING_TYPE= args.TRAINING_TYPE
    print(TRAINING_TYPE)

    TARGETS = {}


    if SINGLE_TASK:
        MODEL_TYPE = ['single_task']
    elif SINGLE_TASK_XOR:
        MODEL_TYPE = ['single_task_xor']
    elif MULTI:
        MODEL_TYPE = ['multi_task']
    elif ALL:
        MODEL_TYPE = ['single_task','single_task_xor','multi_task']
    else:
        print('No training mode selected')

    for model_type in MODEL_TYPE:
        for model_name in os.listdir(MODEL_FOLDER):
            byte = 'all' if 'multi' in model_name else '0'
            multi_task =  'model_{}_{}_{}'.format(model_type,TRAINING_TYPE,byte) 

            if not (multi_task in model_name):
                continue
            if not 'third' in model_name:
                continue
            print(model_name)
            print(multi_task)
            format_string = multi_task + '_cb{}ks{}f{}ps{}db{}du{}_third.h5'
            parsed = parse.parse(format_string,model_name)
            convolution_blocks = int(parsed[0])
            kernel_size_list = parsed[1][1:-1]
            kernel_size_list = kernel_size_list.split(',')   
            kernel_size = [int(elem) for elem in kernel_size_list]
            filters = int(parsed[2])
            pooling_size = int(parsed[3])
            dense_blocks = int(parsed[4])
            dense_units = int(parsed[5])
            
    
            process_eval = Process(target=run_attack, args=(TRAINING_TYPE,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units,model_type))
            process_eval.start()
            process_eval.join()
                            
            
            
    
    
            
            
        
        