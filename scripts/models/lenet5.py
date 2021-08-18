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

"""
Creates a Lenet Model.
Based on: https://medium.com/@krishna.ramesh.tx/training-a-cnn-to-distinguish-between-mnist-digits-using-pytorch-620f06aa9ffa 
"""

# The function creates a model using the nn.Sequential container to pass in modules.
# Function takes no Inputs 
# Function return the model.
def create_lenet():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2), # 1: input channels 6: output channels, 5: kernel size, 1: padding 
        nn.ReLU(), # applies rectified linear unit function element wise.
        nn.AvgPool2d(2, stride=2), # Applies a 2D average pooling over an input signal using 2: kernel size and 2: stride
        nn.Conv2d(6, 16, 5, padding=0), # 6: input channels 16: output channels, 5: kernel size, 0: padding 
        nn.ReLU(), # applies rectified linear unit function element wise.
        nn.AvgPool2d(2, stride=2), # Applies a 2D average pooling over an input signal using 2: kernel size and 2: stride
        nn.Flatten(), # Flattens input by reshaping it to a 1d tensor
        nn.Linear(400, 120), # applies linear transformation using 400 as size of input sample and 120 size of output sample
        nn.ReLU(), # applies rectified linear unit function element wise.
        nn.Linear(120, 84), # applies linear transformation using 120 as size of input sample and 84 size of output sample
        nn.ReLU(), # applies rectified linear unit function element wise.
        nn.Linear(84, 10) # applies linear transformation using 84 as size of input sample and 10 size of output sample
    )
    return model
