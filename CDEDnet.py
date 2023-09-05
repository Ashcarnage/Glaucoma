import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential,Model
from keras.layers import Input,Activation, Dense,Flatten, BatchNormalization,Conv2D,MaxPool2D,UpSampling2D,Concatenate
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import os
import cv2
from matplotlib import pyplot as plt 
import imghdr
# Data aquistion
data = tf.keras.utils.image_dataset_from_directory('data',batch_size=5,image_size=(500,560)) 
data_dir="data"
data = data.map(lambda x,y: (x/255,y))
data_iterator = data.as_numpy_iterator().next()
# batch  =  data_iterator.next()
# print(batch[0].shape)
#print(len(data))
train_size = int(len(data)*0.7)
val_size = int(len(data)*0.2)
test_size = int(len(data)*0.1)+1

train = data.take(train_size)
validate = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
#print(np.array(train.as_numpy_iterator().next()).shape)
# Deep learning model 
class Encoder():
    def __init__(self,inputs):
        self.inputs = inputs
        self.model = None
        self.toggle = True
        self.encdr_features = {}
        self.build()
        
    def conv_block(self,inputs,filters,pool = True):
        x = Conv2D(filters,(3,3),1,padding = "same",activation="relu")(inputs)
        x = Conv2D(filters,(3,3),1,padding = "same",activation="relu")(x)
        if pool:
            p = MaxPool2D(pool_size=(2,2),strides=2)(x)
            return x,p
        else:
            return x
        
    
    def build(self):
        input_size = (500,560,3)
        inputs = Input(input_size)
        n = 64
        x,p = self.conv_block(inputs,n)
        self.encdr_features["x1"] = x
        for i in range(3):
            x,p = self.conv_block(p,n*2)
            input_size = p
            self.encdr_features[f"x{i+2}"] = x
            n*=2
        self.model = Model(inputs=inputs,outputs = p)
        self.model.summary()
        for i in self.encdr_features.values():
            print(i)

            

        # bridge 

class Decoder(Encoder):
    def __init__(self,feature_maps):
        self.model = None
        self.feature_maps = feature_maps
        self.x1 = list(self.feature_maps.keys())[-1]
        self.x_size = self.feature_maps[self.x1].shape
        self.x2 = None
        self.build()
    
    def upsampling_block(self,filters):
        u1 = UpSampling2D((3,3),interpolation = "lanczos3")(self.feature_maps[self.x1])
        print("life saver : ",self.feature_maps[self.x2].shape)
        c1 = Concatenate()([u1,self.feature_maps[self.x2]])
        x0 = self.conv_block(c1,filters,pool = False)
        x5 = self.conv_block(x0,filters//2,pool = False)
        return x5

    def build(self):
        filters = 512
        inputs = Input(self.x_size)
        for i,x in enumerate(list(self.feature_maps.keys())[-2::-1]):
            self.x2 = x
            x0 = self.upsampling_block(filters)
            self.feature_maps[self.x1] = x0
            print("done done")
            filters//=2
        self.model = Model(inputs = inputs,outputs = x0)
        self.model.summary()





        


#print(Input((500,560,3)))
Encdr = Encoder(train)
Decdr = Decoder(Encdr.encdr_features)















