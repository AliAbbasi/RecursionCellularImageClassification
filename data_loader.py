import keras
import numpy as np
import random
import time
from numba import jit, prange 
import cv2
from skimage import transform
import augmenter

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
    
    ## convert labels to one-hot vector 
    train_label = keras.utils.to_categorical(train_label, 1108)
    valid_label = keras.utils.to_categorical(valid_label, 1108)
    
    return train_data, train_label, valid_data, valid_label
    
#----------------------------------------------------------------------------------------------------------------------

def get_a_random_data(data, label):
    rand_index = random.randint(0, data.shape[0]-1) 
    x = data[rand_index] .copy()
    y = label[rand_index].copy()
    
    return x, y

#----------------------------------------------------------------------------------------------------------------------

@jit(parallel=True)
def get_batch_data(data, label, batch_size, do_aug):
    x, y = [], []  
    
    for i in prange(batch_size):      
        cur_x, cur_y = get_a_random_data(data, label)
        
        if do_aug:
            # augmentation
            aug_type = random.randint(-1, 18)   
            if aug_type > -1:
                ## mixup
                if aug_type == 0:  
                    x2,    y2    = get_a_random_data(data, label) 
                    cur_x, cur_y = augmenter.mix_up(cur_x, x2, cur_y, y2)
                ## other augmentations
                else:
                    cur_x = augmenter.apply_augmentation(cur_x, aug_type)  
        
        x.append(cur_x) 
        y.append(cur_y)  
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    x = x/255. 
    
    return x, y 
    
#---------------------------------------------------------------------------------------------------------------------- 
