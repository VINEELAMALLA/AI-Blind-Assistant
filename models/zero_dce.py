import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(32, 3, 3, 1, 1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        x_r = torch.tanh(self.conv7(x6))
        return x_r
