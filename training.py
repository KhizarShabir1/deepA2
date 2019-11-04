# -*- coding: utf-8 -*-

from __future__ import print_function, division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import yolov3loss
import pickle as pkl
import pandas as pd
import random

from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


import warnings
warnings.filterwarnings("ignore")


       
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 64)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)

    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()




args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 2
#classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32


if CUDA:
    model.cuda()
    
    









names,classIDs=returnImageNames('annotations/list.txt')
annotArray=[[0, 0, 0, 0]]
newNames=np.array([])
print (len(names))
for name in names:
 #   print (name)    
    boxes=[]
    try:
        name, boxes = read_content("annotations/xmls/"+name+".xml")
#        print (name, boxes)
        newNames=np.append(newNames,name)
        annotArray=np.append(annotArray,boxes,0)
        
    except:        
        continue

annotArray=annotArray[1:]
print (len(annotArray))
print (len(newNames))
import torch.optim as optim

cd_dataset = CatDogDataset(newNames,annotArray,
                                    root_dir='images/',
                                transform=transforms.Compose([ToTensor()])
                                )



'''
for i in range(len(cd_dataset)):
    sample = cd_dataset[i]

    print(i, sample['image'].size(), sample['bBox'].size())

    if i == 3:
        break

'''


# optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters())
dataloader = DataLoader(cd_dataset, batch_size=4, shuffle=True)

print (dataloader)
 #trainin the model
for i_batch, sample_batched in enumerate(dataloader,1):
    
    print ("data getting load")
    
    print(i_batch, sample_batched['image'].size(),sample_batched['bBox'].size())
    
    loss=model.forward(sample_batched['image'] ,sample_batched['bBox'].size(), None)
    
    
   optimizer.zero_grad()

   epoch_loss += loss.item()
   loss.backward()
   optimizer.step()
   
 #testing the model 
    
   for i_batch, sample_batched in enumerate(dataloader,1):
       
       
       print ("data getting load")
        
       print(i_batch, sample_batched['image'].size(),sample_batched['bBox'].size())
        
       prediction=model.forward(sample_batched['image'] ,None, None)
       print (prediction)
    

   
   
   
   
   
    
    
