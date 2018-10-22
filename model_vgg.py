# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:03:30 2018

@author: Joran
"""

from keras.models import Sequential 
from keras.layers import Convolution2D,MaxPooling2D 
from keras.layers import Dense,Dropout,Flatten,Dropout 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import numpy as np
import os
import matplotlib.image as mpimg



images = []
labels = []
path_name = './'
def get_path(path_name):
    for item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, item))

        if os.path.isdir(full_path):
            get_path(full_path)
        else:
            if  item.endswith('.jpg') :
                image = mpimg.imread(full_path)
                images.append(image)
                labels.append(path_name)
    return images, labels

images,labels = get_path(path_name)
images = np.array(images)
sizes=np.size(labels)
labelswitch=np.zeros((sizes,1))

j=0
for i in labels:
    if i.endswith('daisy'):
        labelswitch[j]=0
        j = j+1
    if i.endswith('tulip'):
        labelswitch[j]=1
        j = j+1
labels = labelswitch


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, random_state = 30)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /=  255


model = Sequential()
model.add(Convolution2D(32, (3, 3),input_shape = (32,32,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(64, (3, 3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
'''model.add(Convolution2D(128, (3, 3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))'''

model.add(Flatten())
model.add(Dense(activation="relu", units=64))
model.add(Dropout(0.5))
model.add(Dense(activation="sigmoid", units=1))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 10)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
model.save('./model_vgg.h5')
print('loss:', test_loss)
print('accuracy', test_accuracy)