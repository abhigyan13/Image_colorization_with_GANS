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


from dataloader import*
from model import*
from config import*


dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available()==False:
  dtype=torch.FloatTensor
print(device,dtype)

generator  = Generator().cuda()
discriminator = Discriminator().cuda()
generator.train()
discriminator.train()

dataset=Places365_loader()
optimizer_G=optim.Adam( generator.parameters(),lr=cfg.lr,betas=(0.5, 0.999),eps=1e-8)
optimizer_D=optim.Adam( discriminator.parameters(),lr=cfg.lr,betas=(0.5, 0.999),eps=1e-8)
train_loader=DataLoader(dataset,cfg.batch_size,shuffle=True,collate_fn=train_collate)
loss_G = []
loss_D = []
if cfg.resume:
  t1 = glob.glob(cfg.gpath+'G_epoch*')
  t2 = glob.glob(cfg.dpath+'D_epoch*')
  checkpoint_G = torch.load(t1[0] )
  checkpoint_D = torch.load(t2[0])
  
  generator.load_state_dict(checkpoint_G['state_dict'])
  optimizer_G.load_state_dict(checkpoint_G['optimizer'])
  discriminator.load_state_dict(checkpoint_D['state_dict'])
  optimizer_D.load_state_dict(checkpoint_D['optimizer'])
  step = checkpoint_G['epoch']
  open_file=open(cfg.loss_path +'Gen_loss_hist.pkl','rb')
  loss_G=pickle.load(open_file)
  open_file.close()
  open_file=open(cfg.loss_path+'Dis_loss_hist.pkl','rb')
  loss_D=pickle.load(open_file)
  open_file.close()
  print('Resuming training from ' , step )
else:
  generator.initialize_weights()
  discriminator.initialize_weights()
  step = 1
  cfg.resume = True


torch.autograd.set_detect_anomaly(True)
from google.colab.patches import cv2_imshow

D_BCE=nn.BCELoss()
G_BCE=nn.BCELoss()
L1=nn.L1Loss()
training = True
time_last = time.time()
while training:
  for i , (gray_imgs , lab_imgs ) in enumerate(train_loader):

    lab_imgs=Variable(lab_imgs.cuda().type(dtype)  )
    gray_imgs=Variable(gray_imgs.cuda().type(dtype)  )

    #update discriminator
    discriminator.zero_grad()
    out = discriminator(torch.cat((gray_imgs , lab_imgs) , 1 )) #Using Ground Truth Image
    out = torch.squeeze(out)
    loss_d_real = D_BCE( out , (0.9*torch.ones(cfg.batch_size ) ).cuda() ) #discriminator Binary Cross Entropy Loss for actual image
    fake_img = generator(gray_imgs).detach()
    out = discriminator(torch.cat((gray_imgs.detach() , fake_img) , 1).detach()) #Using Generated Image
    out =torch.squeeze(out)
    loss_d_fake = D_BCE( out,(torch.zeros(cfg.batch_size)).cuda()) #Discriminator BCE loss for fake generated image
    loss_d = loss_d_real + loss_d_fake
    loss_d.backward()
    loss_D.append(loss_d.item()) 
    optimizer_D.step() 

    #update Generator
    generator.zero_grad()
    fake_img = generator(gray_imgs)
    out = discriminator(torch.cat((gray_imgs , fake_img) , 1) )
    out = torch.squeeze(out)
    loss_g_fake= G_BCE( out ,(torch.ones(cfg.batch_size)).cuda())  #Binary Cross Entropy Loss for generator
    loss_g_l1 = cfg.lambd* L1(fake_img.view(fake_img.size(0),-1),lab_imgs.view(lab_imgs.size(0),-1)) #L1 loss 
    loss_g = loss_g_fake + loss_g_l1
    
    loss_g.backward()
    optimizer_G.step()
    loss_G.append(loss_g.item())
    time_this = time.time()
    batch_time = time_this-time_last
    if i%40==0:
      print("Time for batch " , i , "/", len(train_loader) , "= " , batch_time , ", Loss Generator  = " , loss_g.item() , ' , Loss Discriminator = ' , loss_d.item() )
      if i%7 == 0 :
        with torch.no_grad():
          print('Generated Image ---------------Ground Truth Image')
          f = plt.figure()
          f.add_subplot(1,2, 1)
          i1 = imfakeshow(torch.cat((gray_imgs[0] , fake_img[0]) , 0 ) , True ) 
          i2 = imfakeshow(torch.cat((gray_imgs[0] , lab_imgs[0]) , 0 ) , True )
          plt.imshow(i1)
          f.add_subplot(1,2, 2)
          plt.imshow(i2)
          plt.show(block=True)
          
    time_last = time.time()


  
  save_latest(generator , discriminator , optimizer_G , optimizer_D , step  ) #save weights
  save_loss(loss_G , loss_D)
  print('\n epoch ' , step , ' completed' )
  step+=1
  print('\n \n Generator Loss Plot')
  plt.plot(np.array(loss_G), 'r')
  plt.show()
  print('\n\n Discriminator Loss Plot')
  plt.plot(np.array(loss_D), 'r')
  plt.show()
  if step > cfg.epoch_ul: 
    break

