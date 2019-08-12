import numpy as np
import random 
from numba import jit, prange  
import augmenter
import keras 

#----------------------------------------------------------------------------------------------------------------------

def load_data_and_labels(train_path, valid_path, input_size):
    print ("Train and validation data are loading ...!")
    
    train_data_count  = 63676  
    valid_data_count  = 9354   
    
    train_data  = np.zeros(shape=(train_data_count, input_size[0], input_size[0], 6), dtype=np.float16)
    train_label = np.zeros(shape=(train_data_count, 1),                               dtype=np.float16)
    valid_data  = np.zeros(shape=(valid_data_count, input_size[0], input_size[0], 6), dtype=np.float16)
    valid_label = np.zeros(shape=(valid_data_count, 1),                               dtype=np.float16) 
    
    ## LOAD DATA THERE ##
    train_data[:,:,:,:] = np.load(train_path +  "all_train_data_"+str(input_size[0])+".npy")
    train_label[:,:]    = np.load(train_path + "all_train_label_"+str(input_size[0])+".npy") 
    print ("Train data is loaded.")
    
    valid_data[:,:,:,:] = np.load(valid_path +  "all_valid_data_"+str(input_size[0])+".npy")
    valid_label[:,:]    = np.load(valid_path + "all_valid_label_"+str(input_size[0])+".npy") 
    print ("Validation data is loaded.")
    
    print ("Merging the data for 12 channels")
    train_data_12  = np.zeros(shape=(int(train_data_count/2), input_size[0], input_size[0], 12), dtype=np.float16)
    train_label_12 = np.zeros(shape=(int(train_data_count/2), 1),                                dtype=np.float16)
    valid_data_12  = np.zeros(shape=(int(valid_data_count/2), input_size[0], input_size[0], 12), dtype=np.float16)
    valid_label_12 = np.zeros(shape=(int(valid_data_count/2), 1),                                dtype=np.float16) 
    
    for i in range(0, train_data.shape[0], 2):
        train_data_12[int(i/2),:,:,:6] = train_data[i].copy()
        train_data_12[int(i/2),:,:,6:] = train_data[i+1].copy() 
        train_label_12[int(i/2)] = train_label[i].copy()
    
    print ("train data: ", train_data_12.shape)
    print ("train label: ", train_label_12.shape)
    
    train_data  = None
    train_label = None
    
    for i in range(0, valid_data.shape[0], 2):
        valid_data_12[int(i/2),:,:,:6] = valid_data[i].copy()
        valid_data_12[int(i/2),:,:,6:] = valid_data[i+1].copy() 
        valid_label_12[int(i/2)] = valid_label[i].copy()
    
    print ("valid data: ", valid_data_12.shape)
    print ("valid label: ", valid_label_12.shape)
    
    valid_data  = None
    valid_label = None
    
    ## convert labels to one-hot vector 
    train_label_12 = keras.utils.to_categorical(train_label_12, 1108)
    valid_label_12 = keras.utils.to_categorical(valid_label_12, 1108)
    
    return train_data_12, train_label_12, valid_data_12, valid_label_12
    
#----------------------------------------------------------------------------------------------------------------------

def get_a_random_data(data, label):
    rand_index = random.randint(0, data.shape[0]-1) 
    x = data[rand_index] .copy()
    y = label[rand_index].copy()
    
    return x, y

#----------------------------------------------------------------------------------------------------------------------

@jit(parallel=True)
def get_batch_data(data, label, batch_size, do_aug=False):
    x, y = [], []  
    
    for i in prange(batch_size):      
        cur_x, cur_y = get_a_random_data(data, label) 
        
        if do_aug: 
            aug_type = random.randint(-1, 18)     
            if aug_type > -1:  
                if aug_type == 0:   
                    x2,    y2    = get_a_random_data(data, label) 
                    cur_x, cur_y = augmenter.mix_up(cur_x, x2, cur_y, y2) 
                else:
                    cur_x = augmenter.apply_augmentation(cur_x, aug_type)  
        
        x.append(cur_x) 
        y.append(cur_y)  
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    x = x/255. 
    
    return x, y 
    
#---------------------------------------------------------------------------------------------------------------------- 

## test ::
load_data_and_labels("I:\\Cellular\\saved_npy_data\\train\\", "I:\\Cellular\\saved_npy_data\\validation\\", [128, 128, 12])
    
