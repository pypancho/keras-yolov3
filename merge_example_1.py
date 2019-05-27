# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 22:42:34 2019

@author: Chao-Home
"""

#from keras.layers import Dense

from keras.utils.vis_utils import plot_model
#from keras import layers
from keras.layers import Activation, Dense, Merge,Concatenate

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_mldata("MNIST Original")

# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits
data = dataset.data.astype("float") / 255.0
(X_train, X_test, Y_train, Y_test) = train_test_split(data,
	dataset.target, test_size=0.25)



# convert the labels from integers to vectors
lb = LabelBinarizer()
Y_train = lb.fit_transform(Y_train)
Y_test = lb.transform(Y_test)

model_left= Sequential()  
model_left.add(Dense(50, input_shape=(784,)))  
model_left.add(Activation('relu'))  
   
model_right = Sequential()  
model_right.add(Dense(50, input_shape=(784,)))  
model_right.add(Activation('relu'))  
   
model = Sequential()  
model.add(Merge([model_left,model_right], mode='concat'))  
#x=Concatenate([model_left.output, model_right.output])
model.add(Dense(10))  
model.add(Activation('softmax'))  
model.summary()
#plot_model(model, to_file='model.png')

print("[INFO] training network...")
model.compile(loss='categorical_crossentropy',  optimizer='adam',
              metrics=['accuracy'])  
   
H=model.fit([X_train, X_train], Y_train, batch_size=64, nb_epoch=15, validation_data=([X_test, X_test], Y_test))



# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict([X_test,X_test], batch_size=128)
print(classification_report(Y_test.argmax(axis=1),
	predictions.argmax(axis=1)))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

