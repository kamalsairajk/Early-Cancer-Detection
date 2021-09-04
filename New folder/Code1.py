# -*- coding: utf-8 -*-



#importing Keras, Library for deep learning 
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
        
x=np.array(x);
y=np.array(y);
batch_size=100
nb_classes=2
nb_epoch=100
nb_pool=2
nb_conv=5

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)

model= Sequential()
model.add(Convolution2D(16,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));
model.add(Convolution2D(32,nb_conv,nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));
model.add(Convolution2D(64,nb_conv,nb_conv));
model.add(Activation('relu'));

##model.add(Dropout(0.5));
model.add(Flatten());

model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
###testing part
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


"""
r=imgfiles[34]
im1=Image.open(path+'\\'+r);
im1=im1.convert(mode='RGB')
imrs1=im1.resize((m,n))
imrs1=img_to_array(imrs1)/255;
imrs1=imrs1.transpose(2,0,1);
imrs1=imrs1.reshape(3,m,n);

q=[]
q.append(imrs)
q=np.array(q)
prediction=model.predict(q)
"""
score,acc=model.evaluate(x_test,y_test,batch_size=batch_size)
print("test score",score)
print("accuracy",acc)"""
