import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os 
from multiprocessing import freeze_support
from tqdm import tqdm 

from data_loader import OxfordPetInpaintingDataset, transform

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNetConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = UNetConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = UNetConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels_in=3, n_channels_out=3, bilinear=True):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.inc = UNetConvBlock(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        logits = self.outc(x_up4)
        return logits
    
class UNet5(nn.Module): # This is UNet_Depth5
    def __init__(self, n_channels_in=3, n_channels_out=3, bilinear=True):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = UNetConvBlock(n_channels_in, 64)     
        self.down1 = Down(64, 128)                      
        self.down2 = Down(128, 256)                     
        self.down3 = Down(256, 512)                     
        self.down4 = Down(512, 1024 // factor)          
        self.down5 = Down(1024 // factor, 2048 // factor) 

        self.up1 = Up((2048//factor) + (1024//factor), 1024 // factor, bilinear)
        self.up2 = Up((1024//factor) + 512, 512 // factor, bilinear)
        self.up3 = Up((512//factor) + 256, 256 // factor, bilinear)
        self.up4 = Up((256//factor) + 128, 128 // factor, bilinear)
        self.up5 = Up((128//factor) + 64, 64, bilinear)

        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        out = self.up1(x6, x5)
        out = self.up2(out, x4)
        out = self.up3(out, x3)
        out = self.up4(out, x2)
        out = self.up5(out, x1)
        logits = self.outc(out)
        return logits
    
class UNet3(nn.Module):
    def __init__(self, n_channels_in=3, n_channels_out=3, bilinear=True):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        factor = 2 if bilinear else 1 

        self.inc = UNetConvBlock(n_channels_in, 64)     
        self.down1 = Down(64, 128)                      
        self.down2 = Down(128, 256)                     
        self.down3 = Down(256, 512 // factor)          

        self.up1 = Up((512//factor) + 256, 256 // factor, bilinear)
        self.up2 = Up((256//factor) + 128, 128 // factor, bilinear)
        self.up3 = Up((128//factor) + 64, 64, bilinear)
        
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) 

        out = self.up1(x4, x3) 
        out = self.up2(out, x2)
        out = self.up3(out, x1)
        logits = self.outc(out)
        return logits