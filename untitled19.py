# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:27:00 2019

@author: Khizar Shabir
"""
from __future__ import print_function, division
import os
import cv2 
import torch
import pandas as pd
#import skimage
#from skimage import io, transform
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from darknet import Darknet
# Ignore warnings
'''
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")

image = cv2.imread('C:/Users/Khizar Shabir/Desktop/1.jpg')
newImage=cv2.resize(image, (200,200), interpolation = cv2.INTER_AREA)

cv2.imshow('image',newImage)
cv2.imwrite('C:/Users/Khizar Shabir/Desktop/1.jpg',newImage)
cv2.waitKey(0)