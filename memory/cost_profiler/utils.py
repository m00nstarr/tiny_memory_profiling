import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace = True) /6

        return out

class hsigmoid(nn.Module):
    def forward(self,x ):
        out = F.relu6(x+3, inplace = True) / 6
        
        return out

class SeModule(nn.Module):

    def __init__(self, in_size, reduction = 4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
