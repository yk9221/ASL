import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights

# out_features1 = 256
# out_features2 = 128
resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# class RESNET18(nn.Module):
#     def __init__(self):
#         super(RESNET18, self).__init__()
#         # remove fully connected layer at the end
#         self.resnet = nn.Sequential(*list(resnet18.children())[:-1])

#         # freeze parameters
#         for param in self.resnet.parameters():
#             param.requires_grad = False
        
#         self.fc1 = nn.Linear(resnet18.fc.in_features, out_features1)
#         self.fc2 = nn.Linear(out_features1, out_features2)
#         self.fc3 = nn.Linear(out_features2, 30)

#         self.batchnorm1 = nn.BatchNorm1d(out_features1)
#         self.batchnorm2 = nn.BatchNorm1d(out_features2)

#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.resnet(x)
#         x = x.view(x.size(0), -1)  # Flatten the output
        
#         # Forward pass through your fully connected layers
#         x = F.relu(self.batchnorm1(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.batchnorm2(self.fc2(x)))
#         x = self.dropout(x)
#         x = self.fc3(x)

#         return x
    
# out_features1 = 256
# out_features2 = 128
# out_features3 = 64

# p = 0.6

# class RESNET18_2(nn.Module):
#     def __init__(self):
#         super(RESNET18_2, self).__init__()
#         # remove fully connected layer at the end
#         self.resnet = nn.Sequential(*list(resnet18.children())[:-1])

#         # freeze parameters
#         for param in self.resnet.parameters():
#             param.requires_grad = False
        
#         self.fc1 = nn.Linear(resnet18.fc.in_features, out_features1)
#         self.fc2 = nn.Linear(out_features1, out_features2)
#         self.fc3 = nn.Linear(out_features2, out_features3)
#         self.fc4 = nn.Linear(out_features3, 30)

#         self.batchnorm1 = nn.BatchNorm1d(out_features1)
#         self.batchnorm2 = nn.BatchNorm1d(out_features2)
#         self.batchnorm3 = nn.BatchNorm1d(out_features3)

#         self.dropout = nn.Dropout(p)

#     def forward(self, x):
#         x = self.resnet(x)
#         x = x.view(x.size(0), -1)  # Flatten the output
        
#         # Forward pass through your fully connected layers
#         x = F.relu(self.batchnorm1(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.batchnorm2(self.fc2(x)))
#         x = self.dropout(x)
#         x = F.relu(self.batchnorm3(self.fc3(x)))
#         x = self.fc4(x)

#         return x


out_features1 = 128
p = 0.25

class RESNET18(nn.Module):
    def __init__(self):
        super(RESNET18, self).__init__()

        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])
        
        self.fc1 = nn.Linear(resnet18.fc.in_features, out_features1)
        self.fc2 = nn.Linear(out_features1, 30)

        self.batchnorm = nn.BatchNorm1d(out_features1)

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.batchnorm(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    

def resize_and_pad_image(image, output_size=(256, 256)):
    h, w, _ = image.shape
    pad_width = abs(h-w) // 2
    
    if h > w:
        image = cv2.copyMakeBorder(image, 0, 0, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif h < w:
        image = cv2.copyMakeBorder(image, pad_width, pad_width, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    image = cv2.resize(image, output_size)

    return Image.fromarray(image)

def transform_image_normalize():
    return transforms.Compose([
        transforms.Lambda(lambda x: resize_and_pad_image(np.array(x), (224, 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def transform_image():
    return transforms.Compose([
        transforms.Lambda(lambda x: resize_and_pad_image(np.array(x), (224, 224))),
        transforms.ToTensor(),
    ])