import torch
import torchvision 
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader  

import requests
from PIL import Image
from io import BytesIO

import numpy as np

# The network inherits from the nn.Module
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        # 1: input channels 8: output channels, 3: kernel size, 1: stride, 1: padding
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=(3,3), stride = (1,1), padding= (1,1))
        # 8: input channels 16: output channels, 3: kernel size, 1: stride, 1: padding
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        #max pooling layer 2x2
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # Fully connected layer: input size, output size
        self.fc1 = nn.Linear(784, num_classes)

    # it is inherit from nn.Module, nn.Module have both forward() and backward()
    # In this case, forward() link all layers together,
    # backward is already implemented to compute the gradient descents.
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
