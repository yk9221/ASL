import numpy as np

import cv2
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

num_out_feature = 128
p = 0.5

class RESNET18(nn.Module):
    def __init__(self):
        super(RESNET18, self).__init__()

        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])
        
        self.fc1 = nn.Linear(resnet18.fc.in_features, num_out_feature)
        self.fc2 = nn.Linear(num_out_feature, 30)

        self.batchnorm = nn.BatchNorm1d(num_out_feature)

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

def transform_image():
    return transforms.Compose([
        transforms.Lambda(lambda x: resize_and_pad_image(np.array(x), (224, 224))),
        transforms.ToTensor(),
    ])