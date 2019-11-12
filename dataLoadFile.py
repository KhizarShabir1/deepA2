# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import cv2 
import torch
import pandas as pd
#import skimage
#from skimage import io, transform
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from darknet import Darknet
import torchvision.transforms.functional as F
# Ignore warnings

import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")


def returnImageNames(filename):
    data = pd.read_csv(filename, sep=" ", header=None)
    data.columns = ["Image", "CLASS-ID", "SPECIES", "BREED-ID"]
    names=np.array(data["Image"])
    classIDs=np.array(data["CLASS-ID"])
  #  print (classIDs[:4])
    return names,classIDs

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


class CatDogDataset(Dataset):


    def __init__(self,nams,annotArray , root_dir, transform=None):

        self.names=nams
        self.bboxes = annotArray
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        
        
        print (img_name)
        image = cv2.imread(img_name)
        newImage=cv2.resize(image, (416,416), interpolation = cv2.INTER_AREA)
        bBox = self.bboxes[idx]
        bBox = np.array([bBox])
        
  #      landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': newImage, 'bBox': bBox}
        
        #if self.transform:
        
        sample=ToTensor.__call__(sample)
            
       
        return sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(sample):
        image, bBox = sample['image'], sample['bBox']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = torch.FloatTensor(1, 416, 416)
        #image = F.to_pil_image(image)
        #print("Shape before: ",image.shape)
        image = image.transpose((2, 0, 1))
        #print("Shape after: ",image.shape)
        img1 = torch.from_numpy(image)
        img1 = img1.type('torch.FloatTensor')
        
        bbox1 = torch.from_numpy(bBox)
      #  bbox1 = bbox1.type('torch.DoubleTensor')
        
        sample1 = {'image': img1, 'bBox': bbox1}
        return sample1
        #return {'image': torch.from_numpy(image),
         #       'bBox': torch.from_numpy(bBox)}
        




