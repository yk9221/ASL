import os
import random
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights

out_features1 = 1000
out_features2 = 512
out_features3 = 128
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

class RESNET50(nn.Module):
    def __init__(self):
        super(RESNET50, self).__init__()
        self.resnet = nn.Sequential(*list(resnet50.children())[:-1])

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.fc1 = nn.Linear(resnet50.fc.in_features, out_features1)
        self.fc2 = nn.Linear(out_features1, out_features2)
        self.fc3 = nn.Linear(out_features2, out_features3)
        self.fc4 = nn.Linear(out_features3, 30)
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

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

def transform_image():
    return transforms.Compose([
        transforms.Lambda(lambda x: resize_and_pad_image(np.array(x), (224, 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])