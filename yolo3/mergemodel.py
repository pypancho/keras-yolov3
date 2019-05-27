import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.models import Model
from utils import get_random_data
import argparse
def get_data(lines, lines_rgb):
    batch_size = 32
    input_shape = (416,416)
    n = len(lines)
    i = 0
    while True:
        image_data = []
        image_data_visual =[]
        box_data = []
        box_data_visual =[]
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            image, box = get_random_data(lines[i], input_shape, random=False)
            image_visual, box_visual = get_random_data(lines_rgb[i], input_shape, random=False)
            image_data.append(image)
            image_data_visual.append(image_visual)
            box_data.append(box)
            box_data_visual.append(box_visual)
            i = (i+1) % n
        image_data = np.array(image_data)   # input of original yolo: image
        image_data_visual = np.array(image_data_visual)
        box_data = np.array(box_data)       # output of original yolo: boxes
        box_data_visual = np.array(box_data_visual)
        yield [image_data, image_data_visual]       


def _main(annotation_path_thermal, annotation_path_rgb):
    # First, define the vision modules
    digit_input = Input(shape=(27, 27, 1))
    x = Conv2D(64, (3, 3))(digit_input)
    x = Conv2D(64, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    out = Flatten()(x)  
    vision_model = Model(digit_input, out)
    # Then define the tell-digits-apart model
    digit_a = Input(shape=(27, 27, 1))
    digit_b = Input(shape=(27, 27, 1))
    # The vision model will be shared, weights and all
    out_a = vision_model(digit_a)
    out_b = vision_model(digit_b)
    concatenated = keras.layers.concatenate([out_a, out_b])
    out = Dense(1, activation='sigmoid')(concatenated)
    classification_model = Model([digit_a, digit_b], out)
    classification_model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    with open(annotation_path_thermal) as f:
        lines = f.readlines()
    with open(annotation_path_rgb) as f:
        lines_rgb = f.readlines()
    classification_model.fit(get_data(lines, lines_rgb), [out], epochs=100, batch_size=32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation_path_thermal", type=str, default='thermal.txt', help="input annotation_path")
    parser.add_argument("-d", "--annotation_path_rgb", type=str, default='visual.txt', help="input annotation_path")
    args = parser.parse_args()
    print('annotation_path_thermal = ', args.annotation_path_thermal)
    print('annotation_path_rgb = ', args.annotation_path_rgb)
    _main(args.annotation_path_thermal, args.annotation_path_rgb)
