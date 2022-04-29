import torch
import torch.nn as nn
import numpy as np


class Classifier(nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(3 , 32 , 3 , 1 , "same")
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32 , 64 , 3 , 1 , "same")
        self.pool2 = nn.MaxPool2d(2, 2)      
    
        self.fc1 = nn.Linear(64 * 8 * 8 , 128)
        self.fc2 = nn.Linear(128 , 10)
        
    def forward(self , x):
        
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
