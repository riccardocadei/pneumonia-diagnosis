import torch
from torch import nn
from torch.nn import functional as F

#Define the Convolutional Encoder
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
       
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 256, 3, padding=1)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(14, 14)



    def forward(self, x):
        x = F.relu(self.conv1(x))   # 16 * 224 * 224
        x = self.pool1(x)           # 16 * 56 * 56
        x = F.relu(self.conv2(x))   # 64 * 56 * 56
        x = self.pool1(x)           # 64 * 14 * 14 
        x = F.relu(self.conv3(x))   # 256 * 14 * 14
        x = self.pool2(x)           # 256 * 1 * 1
        x = x.reshape(-1)           # 256
        return x