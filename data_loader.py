import keras
import numpy as np
import random
import time
from numba import jit, prange

#----------------------------------------------------------------------------------------------------------------------

def load_data_and_labels(train_path, valid_path, input_size):
    print ("Train and validation data are loading ...!")
    
    train_data_count  = 63676  
    valid_data_count  = 9354   
    
    train_data  = np.zeros(shape=(train_data_count, input_size[0], input_size[0], 6), dtype=np.float16)
    train_label = np.zeros(shape=(train_data_count, 1),                               dtype=np.float16)
    valid_data  = np.zeros(shape=(valid_data_count, input_size[0], input_size[0], 6), dtype=np.float16)
    valid_label = np.zeros(shape=(valid_data_count, 1),                               dtype=np.float16)
    
    max_sirna_value   = 1107.0
    
    ## LOAD DATA THERE ##
    train_data[:,:,:,:] = np.load(train_path +  "all_train_data_"+str(input_size[0])+".npy")
    train_label[:,:]    = np.load(train_path + "all_train_label_"+str(input_size[0])+".npy") 
    valid_data[:,:,:,:] = np.load(valid_path +  "all_valid_data_"+str(input_size[0])+".npy")
    valid_label[:,:]    = np.load(valid_path + "all_valid_label_"+str(input_size[0])+".npy")
    
    ## convert labels to one-hot vector 
    train_label = keras.utils.to_categorical(train_label, 1108)
    valid_label = keras.utils.to_categorical(valid_label, 1108)
    
    ## normalize the data
    train_data = train_data/255.
    valid_data = valid_data/255. 
    
    return train_data, train_label, valid_data, valid_label
    
#----------------------------------------------------------------------------------------------------------------------

def get_a_random_data(data, label):
    rand_index = random.randint(0, data.shape[0]-1) 
    x = data[rand_index]
    y = label[rand_index]
    
    return x, y

#----------------------------------------------------------------------------------------------------------------------

@jit(parallel=True)
def get_batch_data(data, label, batch_size):
    x, y = [], []  
    
    # s = time.time()
    for i in prange(batch_size):   
        cur_x, cur_y = get_a_random_data(data, label)
        
        ## augment the data with probability of 2/3
        aug_flag = random.randint(0, 2)
        if aug_flag:
            cur_x = apply_augmentation(cur_x)
            
        ## mixup the data with probability of 2/3
        mixup_flag = random.randint(0, 2)
        if mixup_flag:
            x2, y2 = get_a_random_data(data, label)
            cur_x, cur_y = mix_up(cur_x, x2, cur_y, y2)
            
        x.append(cur_x) 
        y.append(cur_y)  
    
    x = np.asarray(x)
    y = np.asarray(y)

    # print (time.time() - s)
    return x, y 
    
#---------------------------------------------------------------------------------------------------------------------- 

def mix_up(x1, x2, y1, y2): 
    lam = np.random.beta(0.4, 0.4) 
    mixed_x = lam * x1 + (1 - lam) * x2 
    mixed_y = lam * y1 + (1 - lam) * y2  
    return mixed_x, mixed_y
    
#----------------------------------------------------------------------------------------------------------------------

def apply_augmentation(x):

    ## TODO:  
    ## 0 mixup 
    ## 1 noise s&p
    ## 2 blur
    ## 3 4 5 rotations 90 180 270
    ## 6 7 flips horiz, verti.
    ## 8 random erasing   ??
    
    # s = time.time()
    aug_type = random.randint(0, 8)
    
    ## noise s&p
    if aug_type == 1:  
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(x)   
        
        # Salt mode
        num_salt = np.ceil(amount * x.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in x.shape]
        out[tuple(coords)] = 1
        
        # Pepper mode
        num_pepper = np.ceil(amount* x.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in x.shape]
        out[tuple(coords)] = 1
        x = out.copy()
    
    ## blur
    elif aug_type == 2:
        row, col, c= x.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row, col, c))
        gauss = gauss.reshape(row, col, c)
        x = x + gauss 
    
    ## 90 rotate
    elif aug_type == 3:
        x = np.rot90(x) 
        
    ## 180 rotate
    elif aug_type == 4:
        x = np.rot90(x)
        x = np.rot90(x) 
        
    ## 270 rotate
    elif aug_type == 5:
        x = np.rot90(x)
        x = np.rot90(x)
        x = np.rot90(x)
        
    ## horizontal flip
    elif aug_type == 6:
        x = np.fliplr(x)
    
    ## vertical flip
    elif aug_type == 7:
        x = np.flipud(x)
    
    else:
        pass
        
    # print (time.time() - s)
    
    return x 
    
#----------------------------------------------------------------------------------------------------------------------

    