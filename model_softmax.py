# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 00:30:55 2018

@author: Joran
"""



import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
import tensorflow as tf
from sklearn.cross_validation import train_test_split

import os
import matplotlib.image as mpimg

#locate to current folder
path_name = './'
images = []
labels = []

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
# turn image into arrays
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
x_test  /= 255


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))


sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
test_loss, test_accuracy = score
model.save('./model_softmax.h5')
print('loss:', test_loss)
print('accuracy', test_accuracy)