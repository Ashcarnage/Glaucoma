import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential,Model
from keras.layers import Input,Conv2DTranspose,Activation, Dense,Flatten, BatchNormalization,Conv2D,MaxPool2D,UpSampling2D,Concatenate
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
#data_iterator = data.as_numpy_iterator().next()
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
        self.filters=64
        self.build()
        
    def conv_block(self,inputs,filters= None,pool = True):
        if filters!=None:
            self.filters = filters
        x = Conv2D(self.filters,(3,3),1,padding = "same",activation="relu")(inputs)
        if filters==None : self.filters*=2
        if self.toggle : 
            self.filters = 64
            self.toggle = False
        x = Conv2D(self.filters,(3,3),1,padding = "same",activation="relu")(x)
        if pool:
            p = MaxPool2D(pool_size=(2,2),strides=2)(x)
            return x,p
        else:
            return x
    def build(self):
        input_size = (500,560,1)
        inputs = Input(input_size)
        x,p = self.conv_block(inputs,filters=1)
        self.encdr_features["x1"] = x
        for i in range(3):
            x,p = self.conv_block(p)
            input_size = p
            self.encdr_features[f"x{i+2}"] = x
        self.encdr_features[f"x{i+3}"]=p
        self.model = Model(inputs=inputs,outputs = p)
        self.model.summary()
        # bridge 

class Decoder():
    def __init__(self,conv_block,feature_maps):
        self.conv_block = conv_block
        self.toggle = False
        self.feature_maps = feature_maps
        self.build()
    def decoder_block(self,inp,filters,concat_layer):
        print(inp)
        x = Conv2DTranspose(filters,(3,3),strides=(2,2),padding="same",activation="relu")(inp)
        print("AFTERRRRR DARKK ; ", x)
        print(x.shape[1])
        if self.toggle:
            pad_height = 1 - tf.shape(x)[1] % 2  # Calculate how much padding is needed to make it odd
            x = tf.pad(x, [[0, 0], [0, pad_height], [0, 0], [0, 0]])
            print("squuuuuuuuu :",x)
            self.toggle = False
        x = Concatenate(axis = -1)([x,concat_layer]) 
        x = self.conv_block(x,filters,pool = False)
        print("finalllllllllylyyyyyy",x)


        print("---------------------------------------> sex")
        return x
    def build(self):
        inputs = Input((500,560,1))
        x1 = self.feature_maps[list(self.feature_maps.keys())[-1]]
        filter = 256
        print("oooolalalalala : ",x1)
        a = True
        for x in list(self.feature_maps.keys())[-2::-1]:
            print("oollayyayayayayayayay",self.feature_maps[x])
            x1 = self.decoder_block(x1,filter,self.feature_maps[x])
            if filter ==64:
                filter = 3
            filter//=2
            if a :
                self.toggle = True
                a = False
        outputs = x1
        ml = Model(inputs,outputs)
        #print(ml.summary())









#print(Input((500,560,3)))
Encdr = Encoder(train)
Decdr = Decoder(Encdr.conv_block,Encdr.encdr_features)


