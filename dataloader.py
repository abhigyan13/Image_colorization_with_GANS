import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.autograd import Variable
import time
import os
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import glob
from PIL import Image
import pickle
import shutil
import matplotlib.pyplot as plt


dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available()==False:
  dtype=torch.FloatTensor
print(device,dtype)



class Places365_loader(data.Dataset):
  def __init__(self , mode = 'train' , path = '/content/drive/MyDrive/colorization/' ):
    self.mode = mode
    if(mode == 'train'):
      path = '/content/drive/MyDrive/colorization/dataset/' 
      self.image_path = glob.glob(path+'train/*.*')
    elif mode == 'test':
      self.test_path = path 

  def __getitem__(self , index ):
    if self.mode == 'test':
      image = cv2.imread(self.test_path)
      image = cv2.resize(image , (256,256))
    else:
      image = cv2.imread(self.image_path[index])  
    numpy_lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    numpy_lab_image = cv2.cvtColor(numpy_lab_image , cv2.COLOR_RGB2LAB)
    numpy_lab_image= numpy_lab_image.astype(np.float64)
    numpy_lab_image /= 255
    numpy_lab_image=torch.from_numpy(numpy_lab_image.transpose(2,0,1)  )
    gray_image = numpy_lab_image[0,:,:].unsqueeze(0)
    lab_image =  numpy_lab_image[1:, :, :]
    mean = torch.Tensor([0.5] )
    lab_image = lab_image - mean.expand_as(lab_image)
    lab_image = 2*lab_image
    gray_image  = gray_image - mean.expand_as(gray_image) 
    gray_image = 2*gray_image 
    return gray_image , lab_image

  def __len__(self):
    if self.mode == 'train':
      return len(self.image_path) 
    else:
      return 1
    

def train_collate(batch):

  gray_list,lab_list =[],[]
  for i,sample in enumerate(batch):
    gray_list.append(torch.tensor(sample[0]) )
    lab_list.append(torch.tensor(sample[1] ) )

  lab_imgs=torch.stack(lab_list)
  gray_imgs=torch.stack(gray_list)

  return gray_imgs , lab_imgs

def test_collate(batch):
  gray_img = torch.tensor(batch[0][0]).unsqueeze(0)
  lab_img = torch.tensor(batch[0][1]).unsqueeze(0)
  
  return gray_img , lab_img

def imfakeshow(img, flag = False  ):
  npimg=img.detach().cpu().numpy()
  npimg=npimg/2 +0.5
  np_lab_img=npimg.transpose(1,2,0)
  np_lab_img*=255
  np_rgb_img=cv2.cvtColor(np_lab_img.astype(np.uint8), cv2.COLOR_LAB2RGB)
  if flag:
    return np_rgb_img
  plt.imshow(np_rgb_img)
  plt.show()

def imgrayshow(img, flag = False):
  npimg=img.detach().cpu().numpy()
  npimg=npimg/2 + 0.5
  np_lab_img=npimg.transpose(1,2,0)
  np_lab_img*=255
  if flag:
    return np_lab_img
  cv2_imshow(np_lab_img)
  

