# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 22:42:34 2019

@author: Chao-Home
"""

#from keras.layers import Dense
#from keras import layers

from keras.utils.vis_utils import plot_model
from keras.layers import Activation, Dense, Concatenate, Conv2D, Merge
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
#print("[INFO] loading MNIST (full) dataset...")
#dataset = datasets.fetch_mldata("MNIST Original")
#
#
## scale the raw pixel intensities to the range [0, 1.0], then
## construct the training and testing splits
#data = dataset.data.astype("float") / 255.0
#(X_train, X_test, Y_train, Y_test) = train_test_split(data,
#	dataset.target, test_size=0.25)

def get_data(annotation_path, input_shape):
    data = []
    with open(annotation_path) as f: 
        for line in f.readlines()[:10]:
            image, box=  get_random_data(line, input_shape, random=True, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True)
            image=image.flatten()
            data.append(image)
    data = np.array(data)
    print('****',data.shape)
    return data
def get_test_data(annotation_path, input_shape):
    data = []
    with open(annotation_path) as f: 
        for line in f.readlines()[11:]:
            image, box=  get_random_data(line, input_shape, random=True, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True)
            image=image.flatten()
            data.append(image)
    data = np.array(data)
    print('****test',data.shape)
    return data
    
def get_label(annotation_path, input_shape):
    box_data =[]
    with open(annotation_path) as f: 
        for line in f.readlines()[:10]:
            image, box=  get_random_data(line, input_shape, random=True, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True)
            box_data.append(box)
    box_data = np.array(box_data)
    print('****box',box_data.shape)
    return box_data
def get_test_label(annotation_path, input_shape):
    box_data =[]
    with open(annotation_path) as f: 
        for line in f.readlines()[11:]:
            image, box =  get_random_data(line, input_shape, random=True, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True)
            box_data.append(box)
    box_data = np.array(box_data)
    print('****box_test',box_data.shape)
    return box_data

        
#        
#def get_label(label_path, input_shape):
#    data = []
#    box_data = []
#    image, box=  get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True)
#    data.append(image)
#    box_data.append(box)
#    
#    
#    return data
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, random=True, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()  # split by SPACE. -libn
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros(len(annotation_line))


        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros(1)


    return image_data, box_data
            
            
def _main(annotation_path_rgb, annotation_path_thermal, classes_path, output_model_path):
    # return
    input_shape = (640,512)
    annotation_path = annotation_path_rgb
    annotation_path_thermal = annotation_path_thermal
        
    model_left= Sequential()   
    model_left.add(Dense(50, input_shape=(983040,)))  
    model_left.add(Activation('relu'))  
       
    model_right = Sequential()  
    model_right.add(Dense(50, input_shape=(983040,)))  
    model_right.add(Activation('relu'))  
       
    model = Sequential()  
    model.add(Merge([model_left,model_right], mode='concat'))  
    #x=Concatenate([model_left.output, model_right.output])
    model.add(Dense(1))  
    model.add(Activation('softmax'))  
    model_left.summary()
    
    model.summary()
    #plot_model(model, to_file='model.png')
    
#    print("[INFO] training network...")
    model.compile(loss='binary_crossentropy',  optimizer='adam',
                  metrics=['accuracy'])  
       
    model.fit([get_data(annotation_path, input_shape) , get_data(annotation_path_thermal, input_shape) ], get_label(annotation_path, input_shape), batch_size=1, nb_epoch=15, validation_data=([get_test_data(annotation_path, input_shape), get_test_data(annotation_path_thermal, input_shape)], get_test_label(annotation_path, input_shape)))
#     evaluate the network
#    print("[INFO] evaluating network...")
#    predictions = model.predict([get_test_data(annotation_path, input_shape), get_test_data(annotation_path_thermal, input_shape)], batch_size=128)
#    print(classification_report(Y_test.argmax(axis=1),
#    	predictions.argmax(axis=1)))
#    
##     plot the training loss and accuracy
#    plt.style.use("ggplot")
#    plt.figure()
#    plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
#    plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
#    plt.plot(np.arange(0, 15), H.history["acc"], label="train_acc")
#    plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
#    plt.title("Training Loss and Accuracy")
#    plt.xlabel("Epoch #")
#    plt.ylabel("Loss/Accuracy")
#    plt.legend()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation_path_rgb", type=str, default='kaist_dataset/rgb.txt', help="input annotation_path")
    parser.add_argument("-b", "--annotation_path_thermal", type=str, default='kaist_dataset/thermal.txt', help="input annotation_path")
    parser.add_argument("-c", "--classes_path", type=str, default='kaist_dataset/classes.txt', help="input classes_path")
    parser.add_argument("-o", "--output_model_path", type=str, default='model_data/pedestrian_model.h5', help="input output_model_path")
    args = parser.parse_args()
    print('annotation_path_rgb = ', args.annotation_path_rgb)
    print('annotation_path_thermal = ', args.annotation_path_thermal)
    print('classes_path = ', args.classes_path)
    print('output_model_path = ', args.output_model_path)

    _main(args.annotation_path_rgb, args.annotation_path_thermal, args.classes_path, args.output_model_path)

