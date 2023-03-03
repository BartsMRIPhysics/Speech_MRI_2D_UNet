# UNet_n_classes.py
# Code to create U-Net model that estimates segmentations with N classes

# Author: Matthieu Ruthven (matthieuruthven@nhs.net)
# Last modified: 3rd March 2023

# Import required modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create U-Net model
class UNet_n_classes(nn.Module):

    def __init__(self, n_classes):
        super(UNet_n_classes, self).__init__()
        # Specify number of classes
        self.n_classes = n_classes
        # Layer 1 (encoder)
        # 1 input channel, 64 output channels, 3x3 convolution kernel
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Layer 2 (encoder)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # Layer 3 (encoder)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        # Layer 4 (encoder)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.dropout1 = nn.Dropout2d()  # Default dropout probability is 0.5
        # Layer 5 (encoder)
        self.conv9 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(1024)
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(1024)
        self.dropout2 = nn.Dropout2d()  # Default dropout probability is 0.5
        self.conv11 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.bn11 = nn.BatchNorm2d(512)
        # Layer 6 (decoder)
        # self.conv12 = nn.Conv2d(512, 512, 3, padding = 1) # Without skip connections
        # With skip connections
        self.conv12 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.bn14 = nn.BatchNorm2d(256)
        # Layer 7 (decoder)
        # self.conv15 = nn.Conv2d(256, 256, 3, padding = 1) # Without skip connections
        # With skip connections
        self.conv15 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn15 = nn.BatchNorm2d(256)
        self.conv16 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn16 = nn.BatchNorm2d(256)
        self.conv17 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.bn17 = nn.BatchNorm2d(128)
        # Layer 8 (decoder)
        # self.conv18 = nn.Conv2d(128, 128, 3, padding = 1) # Without skip connections
        # With skip connections
        self.conv18 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn18 = nn.BatchNorm2d(128)
        self.conv19 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn19 = nn.BatchNorm2d(128)
        self.conv20 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.bn20 = nn.BatchNorm2d(64)
        # Layer 9 (decoder)
        # self.conv21 = nn.Conv2d(64, 64, 3, padding = 1) # Without skip connections
        self.conv21 = nn.Conv2d(128, 64, 3, padding=1)  # With skip connections
        self.bn21 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.conv23 = nn.Conv2d(64, self.n_classes, 3, padding=1)

    def forward(self, x):
        # Layer 1 (encoding)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        # Layer 2 (encoding)
        x2 = F.max_pool2d(x1, 2)  # 2x2 max pooling window with stride = 2
        x2 = F.relu(self.bn3(self.conv3(x2)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        # Layer 3 (encoding)
        x3 = F.max_pool2d(x2, 2)
        x3 = F.relu(self.bn5(self.conv5(x3)))
        x3 = F.relu(self.bn6(self.conv6(x3)))
        # Layer 4 (encoding)
        x4 = F.max_pool2d(x3, 2)
        x4 = F.relu(self.bn7(self.conv7(x4)))
        x4 = F.relu(self.bn8(self.conv8(x4)))
        x4 = self.dropout1(x4)
        # Layer 5 (encoding)
        x5 = F.max_pool2d(x4, 2)
        x5 = F.relu(self.bn9(self.conv9(x5)))
        x5 = F.relu(self.bn10(self.conv10(x5)))
        x5 = self.dropout2(x5)
        x5 = F.relu(self.bn11(self.conv11(x5)))
        # Layer 6 (decoding)
        x6 = torch.cat((x4, x5), 1)  # Skip connection
        x6 = F.relu(self.bn12(self.conv12(x6)))
        x6 = F.relu(self.bn13(self.conv13(x6)))
        x6 = F.relu(self.bn14(self.conv14(x6)))
        # Layer 7 (decoding)
        x7 = torch.cat((x3, x6), 1)  # Skip connection
        x7 = F.relu(self.bn15(self.conv15(x7)))
        x7 = F.relu(self.bn16(self.conv16(x7)))
        x7 = F.relu(self.bn17(self.conv17(x7)))
        # Layer 8 (decoding)
        x8 = torch.cat((x2, x7), 1)  # Skip connection
        x8 = F.relu(self.bn18(self.conv18(x8)))
        x8 = F.relu(self.bn19(self.conv19(x8)))
        x8 = F.relu(self.bn20(self.conv20(x8)))
        # Layer 9 (decoding)
        x9 = torch.cat((x1, x8), 1)  # Skip connection
        x9 = F.relu(self.bn21(self.conv21(x9)))
        x9 = F.relu(self.bn22(self.conv22(x9)))
        x9 = self.conv23(x9)

        return x9
