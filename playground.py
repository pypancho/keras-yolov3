#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:55:51 2019

@author: smart3
"""

import argparse
from yolo import YOLO
from PIL import Image
import cv2
import os
import glob
import numpy 


path = '/home/smart3/yolo3_fuse/test/I00000.jpg'

image = Image.open(path)

YOLO(**vars(FLAGS))
r_image, no_boxes = yolo.detect_image([image,image])