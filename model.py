import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

############## Generator Model

class Generator(nn.Module):

  def __init__(self):
    super().__init__()
    
    self.c1 = nn.Sequential(nn.Conv2d(  1, 64,1, bias=False  ) , nn.LeakyReLU(0.2))
    self.c2 = nn.Sequential(nn.Conv2d( 64, 64,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(64 , momentum = 0.5), nn.LeakyReLU(0.2))
    self.c3 = nn.Sequential(nn.Conv2d( 64,128,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(128 , momentum = 0.5), nn.LeakyReLU(0.2))
    self.c4 = nn.Sequential(nn.Conv2d(128,256,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(256 , momentum = 0.5), nn.LeakyReLU(0.2))
    self.c5 = nn.Sequential(nn.Conv2d(256,512,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(512 , momentum = 0.5 ), nn.LeakyReLU(0.2))
    self.c6 = nn.Sequential(nn.Conv2d(512,512,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(512 , momentum = 0.5), nn.LeakyReLU(0.2))
    self.c7 = nn.Sequential(nn.Conv2d(512,512,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(512 , momentum = 0.5), nn.LeakyReLU(0.2))
    self.c8 = nn.Sequential(nn.Conv2d(512,512,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(512 , momentum = 0.5 ), nn.LeakyReLU(0.2))
    self.dconv7 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=False)
    self.bn_relu7 = nn.Sequential(nn.BatchNorm2d(1024 , momentum = 0.5) , nn.ReLU(inplace = True ))
    self.dconv6 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False)
    self.bn_relu6 = nn.Sequential(nn.BatchNorm2d(1024 , momentum = 0.5 ) , nn.ReLU(inplace = True ))
    self.dconv5 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False)
    self.bn_relu5 = nn.Sequential(nn.BatchNorm2d(1024 , momentum = 0.5 ) , nn.ReLU(inplace = True ))
    self.dconv4 = nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1, bias=False)
    self.bn_relu4 = nn.Sequential(nn.BatchNorm2d(512, momentum = 0.5 ) , nn.ReLU(inplace = True ))
    self.dconv3 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, bias=False)
    self.bn_relu3 = nn.Sequential(nn.BatchNorm2d(256, momentum = 0.5 ) , nn.ReLU(inplace = True ))
    self.dconv2 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1, bias=False)
    self.bn_relu2 = nn.Sequential(nn.BatchNorm2d(128, momentum = 0.5 ) , nn.ReLU(inplace = True ))
    self.dconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
    self.bn_relu1 = nn.Sequential(nn.BatchNorm2d(128, momentum = 0.5 ) , nn.ReLU(inplace = True ))
    self.final = nn.Conv2d(128 , 2 , 1 , bias = False )
    self.Tanh = nn.Tanh()

  def forward(self , x):
    out = x
    out = self.c1(out)
    p1 = out
    out = self.c2(out)
    p2 = out
    out = self.c3(out)
    p3 = out
    out = self.c4(out)
    p4 = out
    out = self.c5(out)
    p5 = out
    out = self.c6(out)
    p6 = out
    out = self.c7(out)
    p7 = out
    out = self.c8(out)
    out = self.dconv7(out)
    out = torch.cat((out , p7) , 1 )
    out = self.bn_relu7(out)
    
    out = self.dconv6(out)
    out = torch.cat((out , p6) , 1 )
    out = self.bn_relu6(out)
    
    out = self.dconv5(out)
    out = torch.cat((out , p5) , 1 )
    out = self.bn_relu5(out)
    
    out = self.dconv4(out)
    out = torch.cat((out , p4) , 1 )
    out = self.bn_relu4(out)
    
    out = self.dconv3(out)
    out = torch.cat((out , p3) , 1 )
    out = self.bn_relu3(out)
    
    out = self.dconv2(out)
    out = torch.cat((out , p2) , 1 )
    out = self.bn_relu2(out)
    
    out = self.dconv1(out)
    out = torch.cat((out , p1) , 1 )
    out = self.bn_relu1(out)
    
    out = self.final(out)
    out = self.Tanh(out)
    return out

  def initialize_weights(self):

      for name,module in self.named_modules():
        if isinstance(module,nn.Conv2d) or isinstance(module,nn.ConvTranspose2d):
          nn.init.xavier_uniform_(module.weight.data)
          if module.bias is not None:
            module.bias.data.zero_()


################ Discriminator Model


class Discriminator(nn.Module):

  def __init__(self , mode = 'train'):
    
    super().__init__()
    self.mode = mode
    self.c1 = nn.Sequential(nn.Conv2d(  3 , 64,1, bias=False  ) , nn.LeakyReLU(0.2))
    self.c2 = nn.Sequential(nn.Conv2d( 64, 64,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(64 ,  momentum = 0.5 ), nn.LeakyReLU(0.2))
    self.c3 = nn.Sequential(nn.Conv2d( 64,128,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(128 ,  momentum = 0.5 ), nn.LeakyReLU(0.2))
    self.c4 = nn.Sequential(nn.Conv2d(128,256,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(256 , momentum = 0.5 ), nn.LeakyReLU(0.2))
    self.c5 = nn.Sequential(nn.Conv2d(256,512,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(512, momentum = 0.5 ), nn.LeakyReLU(0.2))
    self.c6 = nn.Sequential(nn.Conv2d(512,512,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(512, momentum = 0.5 ), nn.LeakyReLU(0.2))
    self.c7 = nn.Sequential(nn.Conv2d(512,512,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(512, momentum = 0.5 ), nn.LeakyReLU(0.2))
    self.c8 = nn.Sequential(nn.Conv2d(512,512,4,padding = 1 , stride = 2, bias=False  ) , nn.BatchNorm2d(512, momentum = 0.5 ), nn.LeakyReLU(0.2))
    self.flat=nn.Flatten()
    self.fc1=nn.Linear(2048,100)
    self.relu9=nn.LeakyReLU(0.2)
    self.fc2=nn.Linear(100,1)

  def forward(self , x ):
    out = x
    out = self.c1(out)
    out = self.c2(out)
    out = self.c3(out)
    out = self.c4(out)
    out = self.c5(out)
    out = self.c6(out)
    out = self.c7(out)
    out = self.c8(out)
    out = self.flat(out)
    out = self.fc1(out)
    out = self.relu9(out)
    out = self.fc2(out)
    out = torch.sigmoid(out)
    return out

  def initialize_weights(self):

      for name,module in self.named_modules():
        if isinstance(module,nn.Conv2d) or isinstance(module,nn.ConvTranspose2d):
          nn.init.xavier_uniform_(module.weight.data)
          if module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module,nn.Linear):
          nn.init.xavier_uniform_(module.weight.data)



  
    


