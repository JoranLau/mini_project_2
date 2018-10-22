# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:49:33 2018

@author: Joran
"""

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
model = load_model('./mode_mlp.h5')
test = image.load_img('test.jpg', target_size = (32, 32))
test = image.img_to_array(test)
test = np.expand_dims(test, axis=0)
result = model.predict(test)

if result[0][0] == 1:
    prediction = 'tulip'
else:
    prediction ='daisy'
print(prediction)