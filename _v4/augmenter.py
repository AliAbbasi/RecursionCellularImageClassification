import numpy as np
import random
import time
from numba import jit, prange 
import cv2 

#----------------------------------------------------------------------------------------------------------------------

channel_count = 12
channel_array = [0,1,2,3,4,5,6,7,8,9,10,11]

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

def rotate_90(x):
    x = np.rot90(x) 
    return x    

#----------------------------------------------------------------------------------------------------------------------

def rotate_180(x):
    x = np.rot90(x) 
    x = np.rot90(x) 
    return x    
    
#----------------------------------------------------------------------------------------------------------------------

def rotate_270(x):
    x = np.rot90(x) 
    x = np.rot90(x) 
    x = np.rot90(x) 
    return x    
    
#----------------------------------------------------------------------------------------------------------------------

def flip_up(x): 
    x = np.flipud(x)
    return x

#----------------------------------------------------------------------------------------------------------------------

def flip_right(x): 
    x = np.fliplr(x)  
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def remove_some_part(x):
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s:
        rect_size = x.shape[0] // 6 
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8) 
        rand_x, rand_y = random.randint(0, x.shape[0]-rect_size-1), random.randint(0, x.shape[0]-rect_size-1)
        a_chan_image[rand_x:rand_x+rect_size, rand_y:rand_y+rect_size] = 0
        x[:,:,random_channel] = a_chan_image
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def his_equ(x):
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s:
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8) 
        a_chan_image = cv2.equalizeHist(a_chan_image)
        x[:,:,random_channel] = a_chan_image
    return x

#----------------------------------------------------------------------------------------------------------------------

def brightness(x):
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s:
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
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s:
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        a_chan_image = cv2.resize(a_chan_image, (x.shape[0]*2, x.shape[0]*2))
        random_x = random.randint(0, x.shape[0])
        random_y = random.randint(0, x.shape[0])
        a_chan_image = a_chan_image[random_x:random_x+x.shape[0], random_y:random_y+x.shape[0]]
        x[:,:,random_channel] = a_chan_image
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def shearing(x):
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s:
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        shear_factor = (-0.2, 0.2)
        shear_factor = random.uniform(*shear_factor)
        M = np.array([[1, abs(shear_factor), 0],[0,1,0]]) 
        nW =  a_chan_image.shape[1] + abs(shear_factor*a_chan_image.shape[0])  
        a_chan_image = cv2.warpAffine(a_chan_image, M, (int(nW), a_chan_image.shape[0]))
        
        x[:,:,random_channel] = a_chan_image[:x.shape[0], :x.shape[0]]
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def translate_down(x):
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s:
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(x.shape[0] // 10, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[random_trans_value:, :] = a_chan_image[:x.shape[0]-random_trans_value, :]
        x[:,:,random_channel] = canvas
        
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def translate_up(x):
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s:
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(x.shape[0] // 10, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[:x.shape[0]-random_trans_value, :] = a_chan_image[random_trans_value:, :]
        x[:,:,random_channel] = canvas
        
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def translate_left(x):
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s:
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(x.shape[0] // 10, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[:, :x.shape[0]-random_trans_value] = a_chan_image[:, random_trans_value:]
        x[:,:,random_channel] = canvas
        
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def translate_right(x):
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s:
        a_chan_image = (x[:,:,random_channel]).astype(np.uint8)
        
        random_trans_value = random.randint(x.shape[0] // 10, x.shape[0] // 6)
        
        canvas = np.zeros_like(a_chan_image)
        canvas[:, random_trans_value:] = a_chan_image[:, :x.shape[0]-random_trans_value]
        x[:,:,random_channel] = canvas
        
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def random_rotate(x): 
    random_channel_count = random.randint(1, channel_count)
    random_channel_s = random.sample(channel_array, k=random_channel_count)
    for random_channel in random_channel_s: 
        a_chan_image = (x[:,:,random_channel]) 
        a_chan_image = a_chan_image.astype(np.uint8) 
        rows, cols = a_chan_image.shape 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),random.randint(-15, 15),1)
        a_chan_image = cv2.warpAffine(a_chan_image,M,(cols,rows)) 
        x[:,:,random_channel] = a_chan_image
        
    return x
    
#----------------------------------------------------------------------------------------------------------------------

def apply_augmentation(x, aug_type): 
    augmentation_functions = [salt_and_pepper, blur, rotate_90, rotate_180, rotate_270, flip_up, flip_right, 
                              remove_some_part, his_equ, brightness, poisson, rescale_and_crop, translate_down, 
                              translate_left, translate_right, translate_up, random_rotate, shearing]
    

    ## first augmentation    
    random_augmentaion_function = random.choice(augmentation_functions) 
    x = random_augmentaion_function(x)
    
    ## # for i in range(6):
    ##     # a_chan_image = (x[:,:,i]).astype(np.uint8)
    ##     # cv2.imwrite("aug_res\\_" +str(0) + "_" + str(i) + ".png", a_chan_image ) 
    
    ## second augmentaion with probability 1/2
    if random.randint(0, 1):
        augmentation_functions.remove(random_augmentaion_function)
        random_augmentaion_function = random.choice(augmentation_functions) 
        x = random_augmentaion_function(x)
    
    ## # for i in range(6):
    ##     # a_chan_image = (x[:,:,i]).astype(np.uint8)
    ##     # cv2.imwrite("aug_res\\_" +str(1) + "_" + str(i) + ".png", a_chan_image )  
    ## # print ("--------------------------")
    ## # a = input()
    
    
    
    ## 1 noise s&p 
    ## 2 blur 
    ## 3 90 rotate 
    ## 4 180 rotate 
    ## 5 270 rotate  
    ## 6 horizontal flip 
    ## 7 vertical flip 
    ## 8 remove some part of data (on random channels)  
    ## 9 histogram equalization (on random channels)  
    ## 10 brightness change (on random channels)  
    ## 11 poisson
    ## 12 scale and crop 
    ## 13 14 15 16 translate   
    ## 17 rotate
    ## 18 shearing  
    
    return x 
    
#----------------------------------------------------------------------------------------------------------------------

    