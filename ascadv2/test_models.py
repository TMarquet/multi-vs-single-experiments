import argparse
import parse
import os
import numpy as np



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from multiprocessing import Process


from train_models import  model_multi_task_flat , model_single_task_xor_flat , model_single_task_xor_extracted , model_multi_task_extracted 

# import dataset paths and variables
from utility import   MODEL_FOLDER

# import custom layers

from utility import load_dataset, load_dataset_multi ,load_model_from_name , get_rank_list_from_prob_dist


#### Training high level function
def test_model(training_type,byte,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units,threat):
    n_traces = 5000

    if  'single_task' in training_type:
        flat = training_type == 'single_task_xor_flat'
        channel = training_type  == 'single_task_xor_channel'
        X_profiling , validation_data = load_dataset(byte,flat = flat,channel = channel,n_traces = n_traces,dataset = 'attack',encoded_labels=False)
        model_t = 'model_single_task_xor_{}'.format(training_type)
    else:
        flat = training_type == 'multi_task_flat'
        X_profiling , validation_data = load_dataset_multi(flat = flat,n_traces = n_traces,dataset = 'attack',encoded_labels=False) 
        model_t = 'model_multi_task_{}'.format(training_type)
    

  
    if model_t == 'model_single_task_xor_extracted':
        structure = model_single_task_xor_extracted(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units,summary = False)          
    elif model_t == 'model_single_task_xor_flat':
        structure = model_single_task_xor_flat(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units,summary = False)         
    elif model_t == 'model_multi_task_extracted':
        structure = model_multi_task_extracted(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units,summary = False)                  
           

        #all_callbacks.append(CombinedMetric())    
    elif model_t == 'model_multi_task_flat':
        structure = model_multi_task_flat(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units,summary = False)                  

    else:
        print('Some error here')

    name =  '{}_{}_cb{}ks{}f{}ps{}db{}du{}.h5'.format(model_t,byte,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units)
    model = load_model_from_name(structure,name)
    
    
    predictions =  model.predict(X_profiling, batch_size=256)
    for k , v in predictions.items():
        labels = validation_data[k]
    
        ranks , acc, scores , acc_5 = get_rank_list_from_prob_dist(v,labels)
        print('=========================')
        print(name)
        print('Mean rank {}, Median rank {}, Mean score {}, Median score {}, Accuracy {}'.format(np.mean(ranks), np.median(ranks),np.mean(scores),np.median(scores),acc))
    # np.save(METRICS_FOLDER + 'results_{}'.format(name),np.array([np.mean(ranks), np.median(ranks),np.mean(scores),np.median(scores),acc]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--SINGLE_TASK', action="store_true", dest="SINGLE_TASK",
                        help='Classical training of the intermediates', default=False)
    parser.add_argument('--SINGLE_TASK_XOR',   action="store_true", dest="SINGLE_TASK_XOR", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=True)
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='Adding the masks to the labels', default=False)
        
    args            = parser.parse_args()
  

    SINGLE_TASK        = args.SINGLE_TASK
    SINGLE_TASK_XOR        = args.SINGLE_TASK_XOR
    MULTI = args.MULTI
    ALL = args.ALL

    TARGETS = {}
    if SINGLE_TASK:   
       training_types = ['single_task']
    elif SINGLE_TASK_XOR:
        training_types = ['single_task_xor']

    elif MULTI:
        training_types = ['multi_task']
    elif ALL:
        training_types = ['single_task_xor','multi_task']
    else:
        print('No training mode selected')


    for model_name in os.listdir(MODEL_FOLDER):
        if 'first_batch' in model_name :
            continue

        multi_task =  'multi_task_extracted' in model_name and (not 'shared' in model_name)
        if not multi_task:
            continue
        print(model_name)
        
        format_string = 'model_multi_task_{}_{}_cb{}ks{}f{}ps{}db{}du{}.h5' 
        parsed = parse.parse(format_string,model_name)
        training_type = parsed[0]
        byte = int(parsed[1]) if not parsed[1] == 'all' else 'all'
        convolution_blocks = int(parsed[2])
        kernel_size_list = parsed[3][1:-1]
    

        kernel_size_list = kernel_size_list.split(',')   
        kernel_size = [int(elem) for elem in kernel_size_list]

        filters = int(parsed[4])
        pooling_size = int(parsed[5])
        dense_blocks = int(parsed[6])
        dense_units = int(parsed[7])


        process_eval = Process(target=test_model, args=(training_type,byte,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units,False))
        process_eval.start()
        process_eval.join()
        break
                            


    print("$ Done !")