import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import functools
import operator
from torch.utils.data import TensorDataset, DataLoader,Dataset
from .utility import RunningAverage,set_logger
import torch.nn.init as init

from torch.optim import SGD, Adam, lr_scheduler
import os


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
    
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.gn1 = nn.GroupNorm(NUM_GROUP, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.gn2 = nn.GroupNorm(NUM_GROUP, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
          self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) )
           
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out



class Resnet20_batchNorm(nn.Module):
    """implementation of ResNet20 with BN layers"""
    def __init__(self, lr, device, n_classes, input_shape = (28,28)):
    #def __init__(self, num_classes=100):
      super(Resnet20_batchNorm, self).__init__()
      block = BasicBlock
      num_blocks = [3,3,3]
      self.num_classes = n_classes
      self.device = device
      self.lr = lr
      self.in_planes = 16
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(16)
      self.relu = nn.ReLU()
      
      self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
      self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
      self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
      self.linear = nn.Linear(64, n_classes)

      self.apply(_weights_init)
      #self.weights = self.apply(_weights_init)
      self.size = self.model_size()
      print(f"size definito {self.size}")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = torch.nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        try:
            out = self.linear(out)
        except:
            out = out
            
        return out
      
    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size
        
    def summary(self):
        return "summary"