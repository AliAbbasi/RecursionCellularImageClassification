import keras
import numpy as np
import random
import time
from numba import jit, prange 
import cv2
from skimage import transform

#----------------------------------------------------------------------------------------------------------------------

def mix_up(x1, x2, y1, y2): 
    lam = np.random.beta(0.4, 0.4) 
    mixed_x = lam * x1 + (1 - lam) * x2 
    mixed_y = lam * y1 + (1 - lam) * y2  
    return mixed_x, mixed_y
    
#----------------------------------------------------------------------------------------------------------------------

def salt_and_pepper(x):
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
    
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def blur(x):
    row, col, c= x.shape
    mean = 0
    var = 25
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row, col, c))
    gauss = gauss.reshape(row, col, c)
    x = x + gauss   
    return x

#----------------------------------------------------------------------------------------------------------------------

def rotate(x, rot_count):
    for i in range(rot_count):
        x = np.rot90(x) 
    return x    
    
#----------------------------------------------------------------------------------------------------------------------

def flip(x, direction):
    if direction:
        x = np.fliplr(x)
    else:
        x = np.flipud(x)
    return x

#----------------------------------------------------------------------------------------------------------------------

def remove_some_part(x):
    random_channel = random.randint(0, 5)
    rect_size = x.shape[0] // 6
    
    a_chan_image = (x[:,:,random_channel]).astype(np.uint8) 
    rand_x, rand_y = random.randint(0, x.shape[0]-rect_size-1), random.randint(0, x.shape[0]-rect_size-1)
    a_chan_image[rand_x:rand_x+rect_size, rand_y:rand_y+rect_size] = 0
    x[:,:,random_channel] = a_chan_image
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def his_equ(x):
    random_channel = random.randint(0, 5)
    a_chan_image = (x[:,:,random_channel]).astype(np.uint8) 
    a_chan_image = cv2.equalizeHist(a_chan_image)
    x[:,:,random_channel] = a_chan_image
    return x

#----------------------------------------------------------------------------------------------------------------------

def brightness(x):
    random_channel = random.randint(0, 5)
    a_chan_image =  x[:,:,random_channel]  #.astype(np.uint8)
    
    a_chan_image[np.where(a_chan_image < 255)] += float(random.randint(-15, 15))
    x[:,:,random_channel] = a_chan_image 
    return x

#----------------------------------------------------------------------------------------------------------------------

def poisson(x):
    x = x.astype(np.uint8)
    vals = len(np.unique(x))
    vals = 2 ** np.ceil(np.log2(vals))
    x = np.random.poisson(x * vals) / float(vals)
    return x

#----------------------------------------------------------------------------------------------------------------------

def rescale_and_crop(x):
    random_channel = random.randint(0, 5)
    a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
    
    a_chan_image = cv2.resize(a_chan_image, (x.shape[0]*2, x.shape[0]*2))
    random_x = random.randint(0, x.shape[0])
    random_y = random.randint(0, x.shape[0])
    a_chan_image = a_chan_image[random_x:random_x+x.shape[0], random_y:random_y+x.shape[0]]
    x[:,:,random_channel] = a_chan_image
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def apply_augmentation(x, aug_type): 
    
    # for i in range(6):
        # print(i)
        # a_chan_image = ((x[:,:,i]).copy()).astype(np.uint8) 
        # cv2.imwrite("aug_res\\_" +str(0) + "_" + str(i) + ".png", a_chan_image ) 
    # print ("--------------------")
    
    ## noise s&p
    if aug_type == 1:   
        x = salt_and_pepper(x)
    
    ## blur
    elif aug_type == 2:
        x = blur(x)
    
    ## 90 rotate
    elif aug_type == 3:
        x = rotate(x, 1)
        
    ## 180 rotate
    elif aug_type == 4:
        x = rotate(x, 2) 
        
    ## 270 rotate
    elif aug_type == 5:
        x = rotate(x, 3)
        
    ## horizontal flip
    elif aug_type == 6:
        x = flip(x, 0)
    
    ## vertical flip
    elif aug_type == 7:
        x = flip(x, 1)
        
    ## remove some part of data (one channel)
    elif aug_type == 8:
        x = remove_some_part(x)
    
    ## histogram equalization (1 channel)
    elif aug_type == 9:
        x = his_equ(x)
    
    ## brightness change (1 channel)
    elif aug_type == 10:
        x = brightness(x)
    
    ## poisson
    elif aug_type == 11: 
        x = poisson(x) 
        
    ## <><><><><><><><>><><><><><><><><><><><><><>><><><><><><><><><><><><><>><><><><><><><><><><><><><>><><><><><>
        
    ## rescale and crop (on 1 channel)
    elif aug_type == 12:
        x = rescale_and_crop(x)
        
    ## translate down (on 1 channel)
    elif aug_type == 13:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(x.shape[0] // 10, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[random_trans_value:, :] = a_chan_image[:x.shape[0]-random_trans_value, :]
        x[:,:,random_channel] = canvas
    
    ## translate up (on 1 channel)
    elif aug_type == 14: 
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(x.shape[0] // 10, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[:x.shape[0]-random_trans_value, :] = a_chan_image[random_trans_value:, :]
        x[:,:,random_channel] = canvas
        
    ## translate left (on 1 channel)
    elif aug_type == 15:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(x.shape[0] // 10, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[:, :x.shape[0]-random_trans_value] = a_chan_image[:, random_trans_value:]
        x[:,:,random_channel] = canvas
    
    ## translate right (on 1 channel)
    elif aug_type == 16:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(x.shape[0] // 10, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[:, random_trans_value:] = a_chan_image[:, :x.shape[0]-random_trans_value]
        x[:,:,random_channel] = canvas
    
    ## rotate (on 1 channel)
    elif aug_type == 17:
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]) 
        a_chan_image = a_chan_image.astype(np.uint8) 
        rows, cols = a_chan_image.shape 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),random.randint(-15, 15),1)
        a_chan_image = cv2.warpAffine(a_chan_image,M,(cols,rows))
        
        # a_chan_image = a_chan_image.astype(np.float16)
        x[:,:,random_channel] = a_chan_image
        
    ## shearing (on 1 channel)
    elif aug_type == 18:  
        random_channel = random.randint(0, 5)
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        shear_factor = (-0.2, 0.2)
        shear_factor = random.uniform(*shear_factor)
        M = np.array([[1, abs(shear_factor), 0],[0,1,0]]) 
        nW =  a_chan_image.shape[1] + abs(shear_factor*a_chan_image.shape[0])  
        a_chan_image = cv2.warpAffine(a_chan_image, M, (int(nW), a_chan_image.shape[0]))
        
        x[:,:,random_channel] = a_chan_image[:x.shape[0], :x.shape[0]]
    
        
    else:
        pass
    
    # for i in range(6):
        # a_chan_image = (x[:,:,i]).astype(np.uint8)
        # cv2.imwrite("aug_res\\_" +str(aug_type) + "_" + str(i) + ".png", a_chan_image ) 
    
    
    ## 1 noise s&p
    ## 2 blur
    ## 3 4 5 rotations 90 180 270
    ## 6 7 flips horiz, verti.  
    ## 8 scale  
    ## 9, 10, 11, 12 translate     
    ## 13 rotate 
    ## 14 shear
    ## 15 poisson 
    ## 16 remove some part of data   
    ## 17 hist equalization 
    ## 18 brightness change 
    
    ## TODO:: mixed augmentaion is also possible 
    ## TODO:  
    ## apply on 1, 2, 3, or all chennels   
    ## TODO: train with aug type only between 1-11
    
    ## TODO: add @jit to all augmentaion functions
    return x 
    
#----------------------------------------------------------------------------------------------------------------------

    