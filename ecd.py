from keras.optimizers import Adam
#importing Keras, Library for deep learning
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import random
# Image manipulations and arranging data
import os
from PIL import Image,ImageFilter,ImageOps
import theano
theano.config.optimizer="None"
#Sklearn to modify the data
import scipy
from scipy.ndimage.filters import gaussian_filter
from sklearn.cross_validation import train_test_split
os.chdir("C:\\Users\\Kamal Sai Raj K\\Downloads\\ALL_IDB1") ###dataset1 contains images 
 
# input image dimensions
m,n = 100,100
 
path="im";
 
imgfiles=os.listdir(path)
x=[]
y=[]
def pp(im):
    imrs=img_to_array(im)/255;
    imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(3,m,n);
    return imrs
def append(im,img):
    x.append(im)
    y.append((img.split('_')[-1]).split('.')[0])
for img in imgfiles:
    im=Image.open(path+'\\'+img);
    im=im.convert(mode='RGB')
    im=im.resize((m,n))
    imrotated=scipy.ndimage.rotate(im, random.randint(-180,180), cval=155)      ##data augmentation
    imrotated2=scipy.ndimage.rotate(im, 90, cval=155)
    imrotated3=scipy.ndimage.rotate(im, 180, cval=155)
    imrotated4=scipy.ndimage.rotate(im, 270, cval=155)
    imrotated=scipy.misc.imresize(imrotated, (100,100))
    imreflected=im.transpose(Image.FLIP_LEFT_RIGHT)
    imreflected2=ImageOps.flip(im)
    imblurred=im.filter(ImageFilter.BLUR)
    imblurred2=gaussian_filter(im,sigma=random.randint(4,8))
    imequalized=ImageOps.equalize(im)
    imtranslated=scipy.ndimage.interpolation.shift(im,(random.randint(25,50),random.randint(25,50),0.0), cval=155)
    append(pp(im),img)
    append(pp(imrotated), img)
    append(pp(imrotated2), img)
    append(pp(imrotated3), img)
    append(pp(imrotated4), img)
    append(pp(imreflected), img)
    append(pp(imreflected2), img)
    append(pp(imblurred), img)
    append(pp(imblurred2), img)
    append(pp(imequalized), img)
    append(pp(imtranslated), img)
 
       
x=np.array(x);
y=np.array(y);
nb_classes=2
nb_pool=2
nb_conv=5
nb_conv1=3
stri=2
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)
 
uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)
 
model= Sequential()                         ## model implementation and architecture
model.add(Convolution2D(16,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));
model.add(Convolution2D(32,nb_conv1,nb_conv1));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));
model.add(Convolution2D(64,nb_conv1,nb_conv1));
model.add(Activation('relu'));
model.add(Convolution2D(128,nb_conv1,nb_conv1,subsample=(2,2)));
model.add(Activation('relu'));
model.add(Flatten());
model.add(Dense(128))
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
adam=Adam(lr=0.001, decay=0.0, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])


pat = 5 #this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
 
#define the model checkpoint callback -> this will keep on saving the model as a physical file
model_checkpoint = ModelCheckpoint('mymodel_2.h5', verbose=1, save_best_only=True)
 
###model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
###testing part
def fit_and_evaluate(t_x, val_x, t_y, val_y, EPOCHS=20, BATCH_SIZE=128):
    results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, model_checkpoint],
              verbose=1, validation_split=0.2) 
    print("Val Score: ", model.evaluate(val_x, val_y))
    return results
 
n_folds=5
epochs=20
batch_size=128
 
#save the model history in a list after fitting so that we can plot later
model_history = []
 
for i in range(n_folds):
    print("Training on Fold: ",i+1)
    t_x, val_x, t_y, val_y = train_test_split(x_train, Y_train, test_size=0.1,random_state = np.random.randint(1,1000, 1)[0])
    model_history.append(fit_and_evaluate(t_x, val_x, t_y, val_y, epochs, batch_size))
    print("======="*12, end="\n\n\n")
 
 
score,acc=model.evaluate(x_test,Y_test,batch_size=batch_size)
print("test score",score)
print("accuracy",acc)