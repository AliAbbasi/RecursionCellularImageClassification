import keras
import numpy as np
import random
import time
from numba import jit, prange 
import cv2
from skimage import transform

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
    ## train_data = train_data/255.
    ## valid_data = valid_data/255. 
    
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
        
        ## augment the data with probability of 3/4
        aug_type = random.randint(-1, 17)
        if aug_type > -1:
            ## mixup
            if aug_type == 0: 
                x2, y2 = get_a_random_data(data, label)
                cur_x, cur_y = mix_up(cur_x, x2, cur_y, y2)
            ## augmentation
            else:
                cur_x = apply_augmentation(cur_x, aug_type)  
                
        x.append(cur_x) 
        y.append(cur_y)  
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    x = x/255. 

    # print (time.time() - s)
    return x, y 
    
#---------------------------------------------------------------------------------------------------------------------- 

def mix_up(x1, x2, y1, y2): 
    lam = np.random.beta(0.4, 0.4) 
    mixed_x = lam * x1 + (1 - lam) * x2 
    mixed_y = lam * y1 + (1 - lam) * y2  
    return mixed_x, mixed_y
    
#----------------------------------------------------------------------------------------------------------------------

def apply_augmentation(x, aug_type): 
    
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
        
    ## rescale and crop (on 1 channel)
    elif aug_type == 8:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        a_chan_image = cv2.resize(a_chan_image, (x.shape[0]*2, x.shape[0]*2))
        random_x = random.randint(0, x.shape[0])
        random_y = random.randint(0, x.shape[0])
        a_chan_image = a_chan_image[random_x:random_x+x.shape[0], random_y:random_y+x.shape[0]]
        x[:,:,random_channel] = a_chan_image
        
    ## translate right (on 1 channel)
    elif aug_type == 9:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(0, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[random_trans_value:, :] = a_chan_image[:x.shape[0]-random_trans_value, :]
        x[:,:,random_channel] = canvas
    
    ## translate left (on 1 channel)
    elif aug_type == 10: 
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(0, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[:x.shape[0]-random_trans_value, :] = a_chan_image[random_trans_value:, :]
        x[:,:,random_channel] = canvas
        
    ## translate up (on 1 channel)
    elif aug_type == 11:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(0, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[:, :x.shape[0]-random_trans_value] = a_chan_image[:, random_trans_value:]
        x[:,:,random_channel] = canvas
    
    ## translate down (on 1 channel)
    elif aug_type == 12:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(0, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[:, random_trans_value:] = a_chan_image[:, :x.shape[0]-random_trans_value]
        x[:,:,random_channel] = canvas
    
    ## rotate (on 1 channel)
    elif aug_type == 13:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        a_chan_image = transform.rotate(a_chan_image, random.randint(-10, 10))
        x[:,:,random_channel] = a_chan_image
        
    ## shearing (on 1 channel)
    elif aug_type == 14:  
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        afine_tf = transform.AffineTransform(shear=0.2) 
        a_chan_image = transform.warp(a_chan_image, inverse_map=afine_tf)
        x[:,:,random_channel] = a_chan_image
    
    ## poisson
    elif aug_type == 15: 
        x = x.astype(np.uint8)
        vals = len(np.unique(x))
        vals = 2 ** np.ceil(np.log2(vals))
        x = np.random.poisson(x * vals) / float(vals) 
    
    ## speckle
    elif aug_type == 16: 
        row,col,chan = x.shape
        gauss = np.random.randn(row, col,chan)
        gauss = gauss.reshape(row,col,chan)        
        x = x + x * gauss  
        
    ## remove some part of data (one channel)
    elif aug_type == 17:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        rect_size = x.shape[0] // 6
        
        rand_x, rand_y = random.randint(0, x.shape[0]-rect_size-1), random.randint(0, x.shape[0]-rect_size-1)
        a_chan_image[rand_x:rand_x+rect_size, rand_y:rand_y+rect_size] = 0
        x[:,:,random_channel] = a_chan_image
        
    else:
        pass
        
    ## TODO:  
    ## 1 noise s&p
    ## 2 blur
    ## 3 4 5 rotations 90 180 270
    ## 6 7 flips horiz, verti.  
    ## 8 scale  
    ## 9, 10, 11, 12 translate     
    ## 13 rotate 
    ## 14 shear
    ## 15 poisson
    ## 16 speckle
    ## 17 remove some part of data
    
    ## mixed augmentaion is also possible
   
    
    ## apply on 1, 2, 3, or all chennels 
    ## TODO: seperate the augmentation class and, write each augmentation as a function
    ## TODO: visualize all  augmented images to check the result of them
    
    return x 
    
#----------------------------------------------------------------------------------------------------------------------

    