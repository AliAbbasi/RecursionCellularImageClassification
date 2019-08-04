
import numpy as np
import os, cv2

#----------------------------------------------------------------------------------------------------------------------
batch_size            = 32
epochs                = 200
input_size            = (256, 256, 6)
path_train_input      = 'I:\\Cellular\\saved_npy_data\\train\\'  
path_valid_input      = 'I:\\Cellular\\saved_npy_data\\validation\\'    
load_model_flag       = False
load_model_path       = 'weights-improvement-200-0.00.hdf5' 


max_sirna_value  = 1107.0
train_data_count = 63676
valid_data_count = 9354

train_data  = np.zeros(shape=(train_data_count, 256, 256, 6), dtype=np.float16)
train_label = np.zeros(shape=(train_data_count, 1), dtype=np.float16)
# valid_data  = np.zeros(shape=(valid_data_count, 256, 256, 6), dtype=np.float16)
# valid_label = np.zeros(shape=(valid_data_count, 1), dtype=np.float16)

    
# idx = 0   
# for i in range(0, valid_data_count, 1000):  
    # print (path_valid_input + "valid_data_"  +str(i) + ".npy")
    # data = np.load(path_valid_input + "valid_data_"  +str(i) + ".npy") 
    # data = data.astype(np.int16)
    # resized_data = np.zeros(shape=(data.shape[0], input_size[0], input_size[0], 6), dtype=np.float16)
    # print (data.shape)
    
    # for index in range(resized_data.shape[0]):
        # for channel in range(6): 
            # resized_data[index, :,:,channel] = cv2.resize(data[index, :,:,channel], (256, 256)) 
    # valid_data [idx:idx+resized_data.shape[0], :, :, :] = resized_data.copy()
    # valid_label[idx:idx+resized_data.shape[0], :]       = np.load(path_valid_input + "valid_laebl_" +str(i) + ".npy")
    # idx += resized_data.shape[0]

# np.save("I:\\Cellular\\saved_npy_data\\validation\\all_valid_data_256.npy", valid_data)
# np.save("I:\\Cellular\\saved_npy_data\\validation\\all_valid_label_256.npy", valid_label)


idx = 0   
for i in range(0, train_data_count, 1000):  
    print (path_train_input + "train_data_"  +str(i) + ".npy")
    data = np.load(path_train_input + "train_data_"  +str(i) + ".npy") 
    data = data.astype(np.int16)
    resized_data = np.zeros(shape=(data.shape[0], input_size[0], input_size[0], 6), dtype=np.float16)
    print (data.shape)
    
    for index in range(resized_data.shape[0]):
        for channel in range(6): 
            resized_data[index, :,:,channel] = cv2.resize(data[index, :,:,channel], (256, 256)) 
    train_data [idx:idx+resized_data.shape[0], :, :, :] = resized_data.copy()
    train_label[idx:idx+resized_data.shape[0], :]       = np.load(path_train_input + "train_laebl_" +str(i) + ".npy")
    idx += resized_data.shape[0]

np.save("I:\\Cellular\\saved_npy_data\\validation\\all_train_data_256.npy", train_data)
np.save("I:\\Cellular\\saved_npy_data\\validation\\all_train_label_256.npy", train_label)