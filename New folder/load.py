# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:53:49 2018

@author: Kamal Sai Raj K
"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
from keras.models import model_from_json
K.set_image_dim_ordering('th')

import numpy as np

# Image manipulations and arranging data
import os
from PIL import Image
import theano
theano.config.optimizer="None"
#Sklearn to modify the data
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
os.chdir("C:\\Users\\Kamal Sai Raj K\\Downloads\\ALL_IDB1") ###dataset1 contains images and images1 folder 

# input image dimensions
m,n = 100,100

path="im";

imgfiles=os.listdir(path)
x=[]
y=[]
for img in imgfiles:
    im=Image.open(path+'\\'+img);
    im=im.convert(mode='RGB')
    imrs=im.resize((m,n))
    imrs=img_to_array(imrs)/255;
    imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(3,m,n);
    x.append(imrs)
    y.append((img.split('_')[-1]).split('.')[0])
        
nb_classes=2
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)
uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score,acc=loaded_model.evaluate(np.array(x_test),np.array(Y_test),verbose=1)
print("test score",score)
print("accuracy",acc)