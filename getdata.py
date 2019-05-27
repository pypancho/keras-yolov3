#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 07:47:04 2019

@author: zhji2822
"""

"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.newmodel import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from keras.utils import plot_model  # plot model

import argparse
def data_generator(annotation_lines, batch_size, input_shape):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)   # input of original yolo: image
        box_data = np.array(box_data)       # output of original yolo: boxes
        yield [image_data], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, input_shape):
    n = len(annotation_lines)
    batch_size = 32 
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape)
def _main(annotation_path, annotation_path_visual):
    # return
    #annotation_path
    annotation_path = annotation_path
    annotation_path_visual = annotation_path_visual   
    input_shape = (416,416) 
#    print("num_classes:",num_classes)
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    with open(annotation_path_visual) as f_visual:
        lines_visual = f_visual.readlines()
#    np.random.seed(10101)
#    np.random.shuffle(lines)
#    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
#    print('data_generator_wrappe:', data_generator_wrapper(lines[:num_train]))
#    print('data_generator_wrapper:', data_generator_wrapper(lines[num_train:]))
    print('new_data_generator_wrapper:', newdata_generator_wrapper(lines_visual[:num_train],lines[:num_train], input_shape))
    print('new_data_generator_wrapper:', newdata_generator_wrapper(lines_visual[num_train:],lines[:num_train], input_shape))
    print('new_data_generator_wrapper:', data_generator_wrapper(lines_visual[:num_train], input_shape))
    print('new_data_generator_wrapper:', data_generator_wrapper(lines_visual[num_train:], input_shape))
def newdata_generator(annotation_lines, annotation_lines_visual, batch_size, input_shape):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        image_data_visual =[]
        box_data = []
        box_data_visual =[]
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_visual, box_visual = get_random_data(annotation_lines_visual[i], input_shape, random=True)
            image_data.append(image)
            image_data_visual.append(image_visual)
            box_data.append(box)
            box_data_visual.append(box_visual)
            i = (i+1) % n
        image_data = np.array(image_data)   # input of original yolo: image
        image_data_visual = np.array(image_data_visual)
        box_data = np.array(box_data)       # output of original yolo: boxes
        box_data_visual = np.array(box_data_visual)
        yield [image_data, image_data_visual], np.zeros(batch_size)

def newdata_generator_wrapper(annotation_lines, annotation_lines_visual, input_shape):
    n = len(annotation_lines)
    batch_size=32
    if n==0 or batch_size<=0: return None
    return newdata_generator(annotation_lines, annotation_lines_visual, batch_size, input_shape)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--annotation_path", type=str, default='kaist_dataset/thermal.txt', help="input annotation_path")
    parser.add_argument("-d", "--annotation_path_visual", type=str, default='kaist_dataset/visual.txt', help="input annotation_path")

    args = parser.parse_args()
    print('annotation_path = ', args.annotation_path)
    print('annotation_path_visual = ', args.annotation_path_visual)
    _main(args.annotation_path, args.annotation_path_visual)
    






