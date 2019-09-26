
#-----------------------------------------------------------------------------------------------------------------

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"        

#-----------------------------------------------------------------------------------------------------------------

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from model import Model
from keras.callbacks import TensorBoard, Callback
from keras.models import load_model
import random, glob, cv2, numpy as np 
from PIL import Image 

import augmenter
from tvtb import TrainValTensorBoard

import data_loader

#-----------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   
sess = tf.Session(config=config)
set_session(sess)   

#-----------------------------------------------------------------------------------------------------------------

dropout=        0.5                                           
epoch_num=      1000                               
batch_size=     256                                           
train_path=     "I:\\Cellular\\Recursion_Cellular_Image_Classification\\"  
valid_path=     "I:\\Cellular\\Recursion_Cellular_Image_Classification\\"  
input_size0=    128                                      
input_size1=    128                                      
input_size2=    12                                       
output_size=    1108                                     
restore=        False                                     
weights=        "model_147000.hdf5"                      
train=          True                                     
directory=      "saved_weights\\"                        
logs=           "logs\\"                                 
augmentation=   True                                     
experiment=     "HUVEC"  


model_obj = Model(input_size0, input_size1, input_size2) 
model = model_obj.get_model() 

train, train_labels, val, val_labels = data_loader.load_data_and_labels(train_path, valid_path, [input_size0, input_size1, input_size2], experiment)
print ("train data size: ",train.shape)
print ("val data size: ",val.shape)  
val = (val / 255.)


## shuffle the train data
# combined = list(zip(train, train_labels)) 
# random.shuffle(combined)
# train, train_labels = zip(*combined)
# train  = np.asarray(train)
# train_labels = np.asarray(train_labels)

## shuffle the valid data
# combined = list(zip(val, val_labels)) 
# random.shuffle(combined)
# val, val_labels = zip(*combined)
# val  = np.asarray(val)
# val_labels = np.asarray(val_labels)

#-----------------------------------------------------------------------------------------------------------------

def train_image_generator(data, label, batch_size): 
    max_step = data.shape[0] // batch_size
    mixup = 1
    
    ## normal augmentation
    while True:
        if mixup:
            for i in range(max_step-1): 
                x_sample, y_sample = data[i*batch_size:(i+1)*batch_size*2], label[i*batch_size:(i+1)*batch_size*2]
                x_batch, y_batch = [], []
                
                ## augmentaion 
                for i in range(0, batch_size, 2):     
                    x_i, y_i = x_sample[i], y_sample[i]
                    
                    ## mixup  
                    x_i, y_i = augmenter.mix_up(x_i, x_sample[i+1], y_i, y_sample[i+1]) 
                    
                    ## augmentation
                    x_batch.append(augmenter.apply_augmentation(x_i))
                    y_batch.append(y_i)
            
                x_batch = np.asarray(x_batch)
                x_batch = x_batch/255.   
                y_batch = np.asarray(y_batch)  
                
                yield x_batch, y_batch 
            
            combined = list(zip(data, label)) 
            random.shuffle(combined)
            data, label = zip(*combined)
            data  = np.asarray(data)
            label = np.asarray(label)
            
        else:
            for i in range(max_step): 
                x_sample, y_sample = data[i*batch_size:(i+1)*batch_size], label[i*batch_size:(i+1)*batch_size]
                x_batch = [] 
                
                ## augmentaion  
                for i in range(0, batch_size):     
                    x_i = x_sample[i]      
                    x_batch.append(augmenter.apply_augmentation(x_i))   
                    # x_batch.append( x_i )   
            
                x_batch = np.asarray(x_batch)  
                x_batch = x_batch/255. 
                
                yield x_batch, y_sample 
            
            combined = list(zip(data, label)) 
            random.shuffle(combined)
            data, label = zip(*combined)
            data  = np.asarray(data)
            label = np.asarray(label)
        
#-----------------------------------------------------------------------------------------------------------------

valid_datagen = ImageDataGenerator() 
validation_generator = valid_datagen.flow(
        val,
        val_labels,
        batch_size=batch_size)

print ("Generators are ready...!")

filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

if not os.path.exists('logs'):
    os.mkdir('logs') 
    
callbacks_list = [checkpoint, TrainValTensorBoard(write_graph=False)]

model.fit_generator(
    train_image_generator(train, train_labels, batch_size),
    steps_per_epoch=(train.shape[0]) // batch_size,
    epochs=epoch_num,
    validation_data=validation_generator,
    validation_steps=(val.shape[0]) // batch_size,
    callbacks=callbacks_list)

model.save('model.h5')
