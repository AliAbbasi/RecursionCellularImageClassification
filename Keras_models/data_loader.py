import numpy as np
import random 
from numba import jit, prange  
import augmenter
import keras , sys
import time

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
    
    val = []
    val_labels = []
    idx_to_delete = []
    split_ratio = 0.2
    random.seed(1000)
    np.random.seed(1000)
    # print (train_label_12)
    
    ## stratified random sampling to seperate validation data from train
    uniqueValues, occurCount = np.unique(train_label_12, return_counts=True)
    for i in range(len(uniqueValues)): 
        cnt = int(occurCount[i] * split_ratio)
        related_tupels = np.where(train_label_12==uniqueValues[i])
        
        indecies = related_tupels[0] 
        np.random.shuffle(indecies) 
        
        ## get validation data
        for idx in range(cnt):
            val.append(train_data_12[indecies[idx]])
            val_labels.append(train_label_12[indecies[idx]]) 
            idx_to_delete.append(indecies[idx]) 
    
    print ("start deleteing...")
    ## delete validation data from train
    train_data_12 = np.delete(train_data_12, idx_to_delete, axis=0)
    train_label_12 = np.delete(train_label_12, idx_to_delete, axis=0)
    
    val  = np.asarray(val)
    val_labels = np.asarray(val_labels)
    
    ## convert labels to one-hot vector 
    train_label_12 = keras.utils.to_categorical(train_label_12, 1108) 
    val_labels = keras.utils.to_categorical(val_labels, 1108)  
    
    return train_data_12 , train_label_12 , val, val_labels
    
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
##             ## if random.randint(0, 3) == 0:   
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
    if do_aug:
        x_sub, y_sub = zip(*random.sample(list(zip(data, label)), k=batch_size*2))  
    else:
        x_sub, y_sub = zip(*random.sample(list(zip(data, label)), k=batch_size)) 
        x = np.asarray(x_sub)
        y = np.asarray(y_sub)
        x = x/255.  
        return x, y 
    
    if do_aug:    
        for i in range(0, batch_size, 2):     
            cur_x, cur_y = x_sub[i], y_sub[i]
            
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


    
