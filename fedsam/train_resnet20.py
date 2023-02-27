import os
import logging
import numpy as np
import matplotlib.pyplot as plt 
from statistics import mean 
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
from torchvision import transforms
from torchvision.models import alexnet, vgg16, resnet18, resnet50
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


DEVICE = 'cuda' # 'cuda' or 'cpu'

LR = 0.1      # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 0.0001  # Regularization, you can keep this at the default

STEP_SIZE = 20    # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 1         # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 100

PRE_TRAINED = False     # set to True to load the pre-trained AlexNet

NETWORK_TYPE = "resnet20"       #define which network we will use:
                              #alexnet, vgg, resnet

FREEZING = "no_freezing"        # define which layers of the network will be kept frozen
                                # None : train the whole network
                                # "CONV" : train only the FC-layers
                                # "FC" : train only the conv-layers

AUG_PROB = 0.5   # the probability with witch each image is transformed at training time during each epoch
AUG_TYPE = "RC-RHF"         # define the type of augmentation pipeline 
                            # None for no data augmentation
                            # "CS-HF" for contrast + saturation + horizontal flip
                            # "H-RP" for hue + random perspective
                            # "B-GS-R" for brightness + grayscale + rotation
                            # "RC-RHF" random crop + random horizontal flip => for the project


if PRE_TRAINED:
  normalizer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) #pretrained on imagenet
else:
  #normalizer = transforms.Normalize(mean = (0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023)) 
  normalizer = transforms.Normalize(mean = (0.5071, 0.4867, 0.4408), std= (0.2675, 0.2565, 0.2761)) #mean and std of cifar100

# Define transforms for training phase
train_transform = transforms.Compose([transforms.ToTensor()])

# Define transforms for the evaluation phase
test_transform = transforms.Compose([transforms.ToTensor()])

bright_t = transforms.ColorJitter(brightness=[1,2])
contrast_t = transforms.ColorJitter(contrast = [2,5])
saturation_t = transforms.ColorJitter(saturation = [1,3])
hue_t = transforms.ColorJitter(hue = 0.2)
gs_t = transforms.Grayscale(3)
rp_t = transforms.RandomPerspective(p = 1, distortion_scale = 0.5)
rot_t = transforms.RandomRotation(degrees = 90)
rand_crop = transforms.RandomCrop(32, padding = 4)
hflip_t = transforms.RandomHorizontalFlip(p = 1)

aug_transformations = {
    "CS-HF": transforms.Compose([contrast_t, saturation_t, hflip_t]),
    "H-RP": transforms.Compose([hue_t, rp_t]),
    "B-GS-R": transforms.Compose([bright_t, gs_t, rot_t]),
    "RC-RHF": transforms.Compose([rand_crop, hflip_t])
    }

if AUG_TYPE is not None:
  aug_transformation = aug_transformations[AUG_TYPE]
  aug_pipeline = transforms.Compose([ 
                                      transforms.ToPILImage(),
                                      transforms.RandomApply([aug_transformation], p = AUG_PROB),
                                      transforms.ToTensor(),
                                      normalizer
                                     ])
else:
  aug_pipeline = normalizer




def evaluate(net, dataloader, print_tqdm = True):
  # Define loss function
  criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy
  
  with torch.no_grad():
    net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
    net.train(False) # Set Network to evaluation mode
    running_corrects = 0
    iterable = tqdm(dataloader) if print_tqdm else dataloader
    losses = []
    for images, labels in iterable: 
      norm_images = []
      for image in images:
        norm_image = normalizer(image)
        norm_images.append(norm_image)
      norm_images = torch.stack(norm_images)  
      norm_images = norm_images.to(device = DEVICE, dtype=torch.float)
      labels = labels.to(DEVICE)
      # Forward Pass
      outputs = net(norm_images)
      loss = criterion(outputs, labels)
      losses.append(loss.item())
      # Get predictions
      _, preds = torch.max(outputs.data, 1)
      # Update Corrects
      running_corrects += torch.sum(preds == labels.data).data.item()
    # Calculate Accuracy
    accuracy = running_corrects / float(len(dataloader.dataset))

  return accuracy, mean(losses)

def train_resnet20(model, trainset, validset, testset, epochs, batch_size):
    
    net = model 
                            
    # Define loss function
    criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy

    parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet

    optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)
    
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, drop_last=True)
    
    # By default, everything is loaded to cpu
    net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda

    cudnn.benchmark # Calling this optimizes runtime

    current_step = 0

    val_accuracies = []
    val_losses = []
    train_accuracies = []
    train_losses = []

    # Start iterating over the epochs
    for epoch in range(epochs):
      print('Starting epoch {}/{}, LR = {}'.format(epoch+1, epochs, scheduler.get_last_lr()))

      # Iterate over the dataset
      for images, labels in trainloader:
          aug_images = []

          for image in images:
            aug_image = aug_pipeline(image) 
            aug_images.append(aug_image) 

          aug_images = torch.stack(aug_images)

          # Bring data over the device of choice
          aug_images = aug_images.to(DEVICE)
          labels = labels.to(DEVICE)

          net.train() # Sets module in training mode

          optimizer.zero_grad() # Zero-ing the gradients

          # Forward pass to the network
          outputs = net(aug_images)

          # Compute loss based on output and ground truth
          loss = criterion(outputs, labels)

          # Log loss
          if current_step % LOG_FREQUENCY == 0:
            print('Step {}, Loss {}'.format(current_step, loss.item()))

          # Compute gradients for each layer and update weights
          loss.backward()  # backward pass: computes gradients
          optimizer.step() # update weights based on accumulated gradients

          current_step += 1

      train_acc, train_loss = evaluate(net, trainloader, print_tqdm = False)
      train_accuracies.append(train_acc)
      train_losses.append(train_loss)
      
      val_acc, val_loss = evaluate(net, validloader, print_tqdm = False)
      val_accuracies.append(val_acc)
      val_losses.append(val_loss)

      # Step the scheduler
      scheduler.step() 
    
    print(f'train_acc = {train_accuracies}')
    print(f'train_loss = {train_losses}')
    print(f'val_acc = {val_accuracies}')
    print(f'val_loss = {val_losses}')

    return train_accuracies, train_losses, val_accuracies, val_losses  