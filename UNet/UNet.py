import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os # For saving models
from multiprocessing import freeze_support
from tqdm import tqdm # <--- IMPORT TQDM

# Assuming your Pet dataset loader script is named 'data_loader.py'
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

        # Encoder
        self.inc = UNetConvBlock(n_channels_in, 64)     # x1: 64
        self.down1 = Down(64, 128)                      # x2: 128
        self.down2 = Down(128, 256)                     # x3: 256
        self.down3 = Down(256, 512)                     # x4: 512
        self.down4 = Down(512, 1024 // factor)          # x5: 1024//factor (e.g., 512 if bilinear)
        self.down5 = Down(1024 // factor, 2048 // factor) # x6 (bottleneck): 2048//factor (e.g., 1024 if bilinear)

        # Decoder
        # up_stage1 (deepest): Takes bottleneck (x6) and skip from down4 (x5)
        # x6 (upsampled) has 2048//factor channels. x5 (skip) has 1024//factor channels.
        # Total input: (2048//factor) + (1024//factor)
        # Output channels: 1024//factor
        self.up1 = Up((2048//factor) + (1024//factor), 1024 // factor, bilinear)

        # up_stage2: Takes output of up_stage1 and skip from down3 (x4)
        # up_stage1_out (upsampled) has 1024//factor channels. x4 (skip) has 512 channels.
        # Total input: (1024//factor) + 512
        # Output channels: 512//factor
        self.up2 = Up((1024//factor) + 512, 512 // factor, bilinear)

        # up_stage3: Takes output of up_stage2 and skip from down2 (x3)
        # up_stage2_out (upsampled) has 512//factor channels. x3 (skip) has 256 channels.
        # Total input: (512//factor) + 256
        # Output channels: 256//factor
        self.up3 = Up((512//factor) + 256, 256 // factor, bilinear)

        # up_stage4: Takes output of up_stage3 and skip from down1 (x2)
        # up_stage3_out (upsampled) has 256//factor channels. x2 (skip) has 128 channels.
        # Total input: (256//factor) + 128
        # Output channels: 128//factor
        self.up4 = Up((256//factor) + 128, 128 // factor, bilinear)

        # up_stage5 (shallowest up): Takes output of up_stage4 and skip from inc (x1)
        # up_stage4_out (upsampled) has 128//factor channels. x1 (skip) has 64 channels.
        # Total input: (128//factor) + 64
        # Output channels: 64
        self.up5 = Up((128//factor) + 64, 64, bilinear)

        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5) # Bottleneck

        out = self.up1(x6, x5) # x6 is from lower layer, x5 is skip
        out = self.up2(out, x4)
        out = self.up3(out, x3)
        out = self.up4(out, x2)
        out = self.up5(out, x1)
        logits = self.outc(out)
        return logits
    
class UNet3(nn.Module): # This is UNet_Depth3
    def __init__(self, n_channels_in=3, n_channels_out=3, bilinear=True):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        factor = 2 if bilinear else 1 # factor for channel reduction in Up blocks if bilinear

        # Encoder
        self.inc = UNetConvBlock(n_channels_in, 64)     # x1: 64 channels
        self.down1 = Down(64, 128)                      # x2: 128 channels
        self.down2 = Down(128, 256)                     # x3: 256 channels
        self.down3 = Down(256, 512 // factor)           # x4 (bottleneck for Depth3): 512//factor channels (e.g. 256 if bilinear)

        # Decoder
        # For Up(in_total_ch, out_ch_of_block, bilinear)
        # in_total_ch = (channels from previous up-layer after upsample) + (channels from skip connection)
        
        # up_stage1: Takes bottleneck (x4) and skip from down2 (x3)
        # x4 (upsampled) will have 512//factor channels. x3 (skip) has 256 channels.
        # Total input to UNetConvBlock in Up: (512//factor) + 256
        # Output channels of this Up block: 256//factor
        self.up1 = Up((512//factor) + 256, 256 // factor, bilinear)

        # up_stage2: Takes output of up_stage1 and skip from down1 (x2)
        # up_stage1_out (upsampled) will have 256//factor channels. x2 (skip) has 128 channels.
        # Total input to UNetConvBlock in Up: (256//factor) + 128
        # Output channels of this Up block: 128//factor
        self.up2 = Up((256//factor) + 128, 128 // factor, bilinear)

        # up_stage3: Takes output of up_stage2 and skip from inc (x1)
        # up_stage2_out (upsampled) will have 128//factor channels. x1 (skip) has 64 channels.
        # Total input to UNetConvBlock in Up: (128//factor) + 64
        # Output channels of this Up block: 64
        self.up3 = Up((128//factor) + 64, 64, bilinear)
        
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # Bottleneck

        out = self.up1(x4, x3) # x4 is from lower layer, x3 is skip
        out = self.up2(out, x2)
        out = self.up3(out, x1)
        logits = self.outc(out)
        return logits