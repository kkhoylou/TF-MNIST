######################################################## Import Modules #####################################################
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Input
from keras.optimizers import adam
from keras.utils.np_utils import to_categorical

######################################################## Import Data #####################################################

#Read Training Set
train = pd.read_csv("train.csv")

#Set train_X to be everything but the initial value which is the y
#Turn values into float32
train_x = train.iloc[:,1:].values.astype('float32')

#Scale the values so mean and std is same throughut
train_x = StandardScaler().fit(train_x).transform(train_x)

#Reshape into a form that Keras will be able to read
train_x = train_x.reshape(train.shape[0], 28, 28, 1)

#Pixel Values are adjust from 0 - 255 to 0 - 1
train_x = train_x/255 

#Create Y Train Variable 
y_train = to_categorical(train["label"].values.astype('float32'))

#Read Testing Set
test = pd.read_csv("test.csv")

#Turn balues into float32
test_x = test.values.astype('float32')

#Scale the values so mean and std is same throughut
test_x = StandardScaler().fit(test_x).transform(test_x)

#Reshape into a form that Keras will be able to read
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

#Pixel Values are adjust from 0 - 255 to 0 - 1
test_x = test_x/255 

######################################################## Create Neural Constants #####################################################

nb_conv = 3
nb_filters_1 = 32
nb_filters_2 = 64
nb_filters_3 = 128

######################################################## Create Model #####################################################

#Initialize Model
model = Sequential()

#Add a single Convolution Layer with 32 filters
model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv,  activation="relu", input_shape=(28, 28, 1),border_mode='valid'))

#Add pooling function
model.add(MaxPooling2D(pool_size=(nb_conv,nb_conv)))

#Add a second Convolution layer with 32 filters
model.add(Convolution2D(nb_filters_3,nb_conv,nb_conv,activation="relu",border_mode='valid'))
model.add(Convolution2D(nb_filters_1, nb_conv, nb_conv,  activation="relu", input_shape=(28, 28, 1),border_mode='valid'))

#Add max Pooling function of 2D
model.add(MaxPooling2D(pool_size=(nb_conv,nb_conv)))

#Drop huge amount of values
model.add(Dropout(.25))

#Flatten the Layers
model.add(Flatten())

#Add Dense factor
model.add(Dense(128,activation='relu'))

#add second dense factor
model.add(Dense(16,activation='relu'))

#Add a Dropout
model.add(Dropout(.05))

#Add a second Dense
model.add(Dense(10,activation='softmax'))

#Compile the code using cross entropy as loss function and adam as optimizer
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

model.fit(train_x,y_train,validation_split=.05,batch_size=128,nb_epoch=7,verbose=1)

nn_pred = pd.DataFrame(data=np.argmax(model.predict(test_x), axis=1),    # values
						  index=range(1,28001), #set index
              			  columns=['Label'])   # 1st column as index

nn_pred.to_csv('output.csv', header=True, index_label='ImageId')

