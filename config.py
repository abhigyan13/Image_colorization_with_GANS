import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from google.colab.patches import cv2_imshow
import cv2
import glob
from PIL import Image
import pickle
import matplotlib.pyplot as plt

dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available()==False:
  dtype=torch.FloatTensor

class set_config:

  def __init__(self):
    self.cuda=torch.cuda.is_available()
    self.weight_decay=0 
    self.lr=0.00009 
    self.test_img = None
    self.batch_size = 8 
    self.mode='train'
    self.resume= True
    self.path='/content/drive/MyDrive/colorization/'
    self.lambd =100
    self.epoch_ul = 60
    self.gpath= self.path+ 'weights/'
    self.dpath = self.path+'weights/'
    self.loss_path = '/content/drive/MyDrive/colorization/loss_history/'
    self.save_test = True

def save_checkpoint(state , filename ):
  torch.save(state, filename )


def save_latest( net1 , net2 , o1 , o2 , epoch , path = '/content/drive/MyDrive/colorization/weights/'):
  
  t1 = glob.glob(path+'G_epoch*')
  assert len(t1)<=1, "Multiple weights file, delete others."
  if t1:
    os.remove(t1[0])
  t1 = glob.glob(path+'D_epoch*')
  assert len(t1)<=1, "Multiple weights file, delete others."
  if t1:
    os.remove(t1[0])
    
  print('Saving check point for ', epoch, ' epoch ')
  save_checkpoint({'epoch': epoch + 1,
                             'state_dict': net1.state_dict(),
                             'optimizer': o1.state_dict(),
                             },
                             filename=path+'G_epoch%d.pth.tar' \
                             % epoch)
  save_checkpoint({'epoch': epoch + 1,
                             'state_dict': net2.state_dict(),
                             'optimizer': o2.state_dict(),
                             },
                             filename=path+'/D_epoch%d.pth.tar' \
                             % epoch)

def save_loss(loss_g,loss_d , path = '/content/drive/MyDrive/colorization/loss_history/'):

  loss_g_hist=glob.glob(path+'Gen_loss_*')
  loss_d_hist=glob.glob(path+'Dis_loss_*')
  assert len(loss_g_hist)<=1, "Multiple files of Gen History"
  assert len(loss_d_hist)<=1, "Multiple files of Dis History"
  if loss_g_hist:
    os.remove(loss_g_hist[0])
  if loss_d_hist:
    os.remove(loss_d_hist[0])
  open_file = open(path+"Gen_loss_hist.pkl", "wb")
  pickle.dump(loss_g, open_file)
  open_file.close()
  open_file = open( path +"Dis_loss_hist.pkl", "wb")
  pickle.dump(loss_d, open_file)
  open_file.close()


cfg=set_config() 