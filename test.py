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
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import glob

from dataloader import *
from model import *
from config import *


import matplotlib.pyplot as plt


dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available()==False:
  dtype=torch.FloatTensor
print(device,dtype)




model = Generator().to(device)
model.eval()
cfg = set_config()
cfg.mode = 'test'
t1 = glob.glob(cfg.gpath+'G_epoch*')
checkpoint = torch.load(t1[0])
model.load_state_dict(checkpoint['state_dict'])  #loading trained weights


cfg.test_img = '/content/drive/MyDrive/colorization/test/i22.jpg' #setting test_img_path
fname = cfg.test_img.split('/')[-1]
fname = fname.split('.')[0]


with torch.no_grad():
  test_data=Places365_loader('test' , cfg.test_img )
  test_loader=DataLoader(test_data,1,shuffle=False,collate_fn=test_collate )
  for i,(gray_img , lab_img ) in enumerate(test_loader):
    lab_img =lab_img.cuda().type(dtype)
    gray_img=gray_img.cuda().type(dtype)
    fake_img= model(gray_img)
    imgrayshow(gray_img[0]  )
    imfakeshow(torch.cat((gray_img[0] , fake_img[0]) , 0 )  ) 
    imfakeshow(torch.cat((gray_img[0] , lab_img[0]) , 0 )  )
    
    
