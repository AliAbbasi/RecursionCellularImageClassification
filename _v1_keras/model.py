from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers import Input
from keras.models import load_model
from keras.models import Model as keras_model
from keras import optimizers
from keras import backend as K
import tensorflow as tf

class Model: 
    def __init__(self, img_width=None, img_height=None, color_mode=None):
        self.img_width = img_width
        self.img_height = img_height 
        self.channel_num = 12 
        
    def get_model(self): 
        if K.image_data_format() == 'channels_first':
            input_shape = (self.channel_num, self.img_height, self.img_width)
        else:
            input_shape = (self.img_height, self.img_width, self.channel_num)
            
        input = Input(shape=input_shape) 
        conv_0 = Conv2D(16, (3, 3), input_shape=input_shape, padding='same')(input)

        x = Conv2D(16, (3, 3), padding='same')(conv_0)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(16, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        conv_1 = Activation('relu')(x) 
        merge_1 = Concatenate(axis=3)([conv_0, conv_1]) 
        pool_1 = MaxPooling2D(pool_size=(2, 2))(merge_1)
        
        x = Conv2D(16, (3, 3), padding='same')(pool_1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(16, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        conv_2 = Activation('relu')(x)
        merge_2 = Concatenate(axis=3)([pool_1, conv_2]) 
        pool_2 = MaxPooling2D(pool_size=(2, 2))(merge_2)
        
        x = Conv2D(32, (3, 3), padding='same')(pool_2)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        conv_3 = Activation('relu')(x)
        merge_3 = Concatenate(axis=3)([pool_2, conv_3]) 
        pool_3 = MaxPooling2D(pool_size=(2, 2))(merge_3)
        
        x = Conv2D(64, (3, 3), padding='same')(pool_3)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        conv_4 = Activation('relu')(x)
        merge_4 = Concatenate(axis=3)([pool_3, conv_4]) 
        pool_4 = MaxPooling2D(pool_size=(2, 2))(merge_4)
        
        x = Conv2D(128, (3, 3), padding='same')(pool_4)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        conv_5 = Activation('relu')(x)
        merge_5 = Concatenate(axis=3)([pool_4, conv_5]) 
        pool_5 = MaxPooling2D(pool_size=(2, 2))(merge_5)
        pool_5 = Flatten()(pool_5)
        
        x = Dense(256)(pool_5)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1108)(x)
        output = Activation('softmax')(x)

        adam = optimizers.Adam(lr=0.00005) 
        model = keras_model(inputs=[input], outputs=[output]) 
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
        print(model.summary()) 
        return model
        
    
    