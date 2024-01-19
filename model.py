
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import torchvision.models as models #To load the ResNet model from the torchvision

DEVICE = 'cuda'
#Import the models you want from torchvision or write from scratch
model = models.resnet18(pretrained = True)
    
for param in model.parameters():
    param.requires_grad = True   
    
model.fc = nn.Sequential(
               nn.Linear(512, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 7))

# cl_model = model 
cl_model = model #For the ResNet pre-trained network
cl_model.to(DEVICE) #Load the models to GPU
