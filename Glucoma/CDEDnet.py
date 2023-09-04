import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense,Flatten, BatchNormalization,Conv2D,MaxPool2D
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

# Deep learning model 
class Encoder():
    def __init__(self,inputs):
        self.inputs = inputs
        self.model = None
        self.build()

    def build(self,block = 4):
        model = Sequential()
        model.add(Conv2D(64,(3,3),1,activation = 'relu',input_shape = (500,560,3)))
        for i in range(block):
            if i>0:
                model.add(Conv2D(64*(i+1),(3,3),1,activation = 'relu'))
            model.add(Conv2D(64*(i+1),(3,3),1,activation = 'relu'))
            model.add(MaxPool2D(pool_size=(2,2),strides=2))
        self.model = model

Encdr = Encoder(train)

print(Encdr.model.summary())







