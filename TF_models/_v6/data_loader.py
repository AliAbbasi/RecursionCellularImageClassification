import numpy as np
import random 
from numba import jit, prange  
import augmenter
import keras 

#----------------------------------------------------------------------------------------------------------------------

def load_data_and_labels(train_path, valid_path, input_size, experiment):
    print ("Train and validation data are loading ...!") 
    
    ## LOAD DATA THERE ##
    train_data  = np.load(train_path + "train_" +experiment+ "_data.npy")
    train_label = np.load(train_path + "train_" +experiment+ "_label.npy")
    print ("Train data is loaded.") 
    
    print ("Merging the data for 12 channels")
    train_data_12  = np.zeros(shape=(int(train_data.shape[0]/2), input_size[0], input_size[0], 12), dtype=np.float16)
    train_label_12 = np.zeros(shape=(int(train_data.shape[0]/2), 1),                                dtype=np.float16) 
    
    for i in range(0, train_data.shape[0], 2):
        train_data_12[int(i/2),:,:,:6] = train_data[i].copy()
        train_data_12[int(i/2),:,:,6:] = train_data[i+1].copy() 
        train_label_12[int(i/2)] = train_label[i].copy() 
    
    train_data  = None
    train_label = None 
    
    ## convert labels to one-hot vector 
    train_label_12 = keras.utils.to_categorical(train_label_12, 1108) 
    
    ## shuffle the data
    combined = list(zip(train_data_12, train_label_12))
    random.seed(1000)
    random.shuffle(combined)
    train_data_12, train_label_12 = zip(*combined)
    train_data_12  = np.asarray(train_data_12)
    train_label_12 = np.asarray(train_label_12)
    
    ## split data into train and validation
    split_ratio = int (0.1 * train_data_12.shape[0])  
    
    return train_data_12[split_ratio:], train_label_12[split_ratio:], train_data_12[:split_ratio], train_label_12[:split_ratio]
    
#----------------------------------------------------------------------------------------------------------------------

def get_a_random_data(data, label):
    rand_index = random.randint(0, data.shape[0]-1) 
    x = data[rand_index] .copy()
    y = label[rand_index].copy()
    
    return x, y

#----------------------------------------------------------------------------------------------------------------------

## @jit(parallel=True)
## def get_batch_data(data, label, batch_size, do_aug=False):
##     x, y = [], []   
##     
##     for i in prange(batch_size):      
##         cur_x, cur_y = get_a_random_data(data, label) 
##         
##         if do_aug:    
##             ## mixup with probability 3/4
##             # if random.randint(0, 3) == 0:   
##             x2,    y2    = get_a_random_data(data, label) 
##             cur_x, cur_y = augmenter.mix_up(cur_x, x2, cur_y, y2) 
##             
##             ## augmentation
##             cur_x = augmenter.apply_augmentation(cur_x)  
##         
##         x.append(cur_x) 
##         y.append(cur_y)  
##     
##     x = np.asarray(x)
##     y = np.asarray(y)
##     
##     x = x/255. 
##     
##     return x, y 
    
    
def get_batch_data(data, label, batch_size, do_aug=False):
    x, y = [], []   
    
    ## get whole data at once
    x_sub, y_sub = zip(*random.sample(list(zip(data, label)), k=batch_size*2)) 
    
    for i in range(0, batch_size, 2):      
        cur_x, cur_y = x_sub[i], y_sub[i]
        
        if do_aug:    
            ## mixup  
            cur_x, cur_y = augmenter.mix_up(cur_x, x_sub[i+1], cur_y, y_sub[i+1]) 
            
            ## augmentation
            cur_x = augmenter.apply_augmentation(cur_x)  
        
        x.append(cur_x) 
        y.append(cur_y)  
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    x = x/255. 
    
    return x, y 
    
#---------------------------------------------------------------------------------------------------------------------- 

## test ::
# load_data_and_labels("I:\\Cellular\\saved_npy_data\\train\\", "I:\\Cellular\\saved_npy_data\\validation\\", [128, 128, 12], "HEPG2")
    
