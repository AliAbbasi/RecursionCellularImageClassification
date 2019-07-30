import keras
import numpy as np

#----------------------------------------------------------------------------------------------------------------------

def load_data_and_labels(train_path, valid_path, input_size):
    train_data_count  = 63676
    valid_data_count  = 9354
    
    train_data  = np.zeros(shape=(train_data_count, input_size[0], input_size[0], 6), dtype=np.float16)
    train_label = np.zeros(shape=(train_data_count, 1), dtype=np.float16)
    valid_data  = np.zeros(shape=(valid_data_count, input_size[0], input_size[0], 6), dtype=np.float16)
    valid_label = np.zeros(shape=(valid_data_count, 1), dtype=np.float16)
    
    max_sirna_value   = 1107.0
    
    ## LOAD DATA THERE ##
    train_data[:,:,:,:] = np.load(train_path +  "all_train_data_"+str(input_size[0])+".npy")
    train_label[:,:]    = np.load(train_path + "all_train_label_"+str(input_size[0])+".npy") 
    valid_data[:,:,:,:] = np.load(valid_path +  "all_valid_data_"+str(input_size[0])+".npy")
    valid_label[:,:]    = np.load(valid_path + "all_valid_label_"+str(input_size[0])+".npy")
    
    ## convert labels to one-hot vector 
    train_label = keras.utils.to_categorical(train_label, 1108)
    valid_label = keras.utils.to_categorical(valid_label, 1108)
    
    ## shuffle the data
    combined = list(zip(train_data, train_label))
    random.shuffle(combined)
    train_data, train_label = zip(*combined)
    train_data  = np.asarray(train_data)
    train_label = np.asarray(train_label)
    
    ## normalize the data
    train_data = train_data/255.
    valid_data = valid_data/255. 
    
    return train_data, train_label, valid_data, valid_label
    
#----------------------------------------------------------------------------------------------------------------------

def apply_augmentation():
    ## TODO: add augmentation stuff here randomly
    ## mixup
    ## noise
    ## blue
    ## rotations
    ## flips
    ## we should have about 11 million data 
    pass