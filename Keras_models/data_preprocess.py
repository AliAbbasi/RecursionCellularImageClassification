
import numpy as np
import os, cv2, sys 
import numpy as np
import glob  
from PIL import Image   
import pandas as pd 


#----------------------------------------------------------------------------------------------------------------------

train_path = 'C:\\train\\'
test_path = 'C:\\test\\'

train_np_save_path = "train_np_files\\" 
train_np_save_label_path = "train_np_files_labels\\" 
test_np_save_path = "test_np_files\\" 

input_size = (128, 128, 6)

train_data = []
train_label = []
test_data = []
test_label = []


experiment_keys   = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']
experiment_counts = [7750*2,  17688*2, 7753*2, 3324*2]

df = pd.read_csv('train.csv')
N = df.shape[0] 

print ("The train data processing is started.!")
for key in range(len(experiment_keys)):
    
    index = 0
    data_container  = np.zeros(shape=(experiment_counts[key], input_size[0], input_size[0], 6), dtype=np.float16)
    label_container = np.zeros(shape=(experiment_counts[key], 1),                               dtype=np.float16)
    
    for i in range(N): 
        print (df['id_code'][i])
        
        experiment = df['experiment'][i]
        
        if experiment.split("-")[0] == experiment_keys[key]:
            plate = df['plate'][i]
            well = df['well'][i]
            

            for site in [1, 2]: 
                image_template = np.zeros(input_size, dtype=np.float16) 
                
                for channel in [1,2,3,4,5,6]:
                    image_path = train_path + str(experiment) + "\\Plate" + str(plate.item()) + "\\" + str(well) + "_s" + str(site) + "_w" + str(channel) + ".png"
                    image_template[:,:,channel-1] = cv2.resize(cv2.imread(image_path, 0), (128, 128))
                
                data_container[index, :,:,:] = image_template.copy()
                label_container[index] = int(df['sirna'][i])
                index += 1
                
    np.save("train_" + experiment_keys[key] + "_data",  data_container)
    np.save("train_" + experiment_keys[key] + "_label", label_container)
    

#----------------------------------------------------------------------------------------------------------------------

experiment_keys   = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']
experiment_counts = [4429*2,    8846*2,     4417*2,  2205*2]

## process the test data
df  = pd.read_csv('test.csv') 
N = df.shape[0]

print ("The test data processing is started.!") 
for key in range(len(experiment_keys)):
    
    index = 0
    data_container  = np.zeros(shape=(experiment_counts[key], input_size[0], input_size[0], 6), dtype=np.float16)
    label_container = np.zeros(shape=(experiment_counts[key], 1),                               dtype=np.float16)
    
    for i in range(N): 
        print (df['id_code'][i])
        
        experiment = df['experiment'][i]
        
        if experiment.split("-")[0] == experiment_keys[key]:
            plate = df['plate'][i]
            well = df['well'][i]
            

            for site in [1, 2]: 
                image_template = np.zeros(input_size, dtype=np.float16) 
                
                for channel in [1,2,3,4,5,6]:
                    image_path = test_path + str(experiment) + "\\Plate" + str(plate.item()) + "\\" + str(well) + "_s" + str(site) + "_w" + str(channel) + ".png"
                    image_template[:,:,channel-1] = cv2.resize(cv2.imread(image_path, 0), (128, 128))
                
                data_container[index, :,:,:] = image_template.copy() 
                index += 1
                
    np.save("test_" + experiment_keys[key] + "_data", data_container) 
    
sys.exit(1)

#----------------------------------------------------------------------------------------------------------------------
sys.exit(1)
#----------------------------------------------------------------------------------------------------------------------
batch_size            = 32
epochs                = 200
input_size            = (256, 256, 6)
path_train_input      = 'C:\\'  
path_valid_input      = 'C:\\'    
load_model_flag       = False
load_model_path       = 'weights.hdf5' 

max_sirna_value  = 1107.0
train_data_count = 63676
valid_data_count = 9354

train_data  = np.zeros(shape=(train_data_count, 256, 256, 6), dtype=np.float16)
train_label = np.zeros(shape=(train_data_count, 1), dtype=np.float16) 


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

np.save("C:\\", train_data)
np.save("C:\\", train_label)