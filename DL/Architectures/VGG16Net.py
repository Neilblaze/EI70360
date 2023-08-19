import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets


# List defining the VGG16 architecture's convolutional layers
conv_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']


# Definition of the VGG16Net architecture
class VGG16Net(nn.Module):
    def __init__(self):
        super(VGG16Net, self).__init__()
        self.conv = nn.ModuleList()
        in_channel = 3
        for out_channel in conv_layers:
            if out_channel == 'M':
                self.conv.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer
            else:
                self.conv.append(nn.Conv2d(in_channels=in_channel,
                                           out_channels=out_channel,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1))  # Convolutional layer
            in_channel = out_channel



    def forward(self, x):
        pass

