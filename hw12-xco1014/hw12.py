import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import pandas as pd

import os
import sys

patharg =sys.argv


target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((32, 32)),
    # 水平翻轉 (Augmentation)
    #transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    #transforms.RandomRotation(15, fill=(0,)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])

target_dataset = ImageFolder(os.path.join(patharg[1], "test_data"), transform=target_transform)

test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 10, 3, 1, 1),
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(10, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
feature_extractor.load_state_dict(torch.load('extractor_model.bin'))
label_predictor.load_state_dict(torch.load('predictor_model.bin'))

result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)


result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(patharg[2],index=False)