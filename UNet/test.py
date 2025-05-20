import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt # <--- IMPORT MATPLOTLIB

# --- Assuming your data_loader.py is in the same directory ---
from data_loader import OxfordPetInpaintingDataset, transform

# --- Configuration ---
IMG_SIZE = 64
BATCH_SIZE = 16
DATA_DIR = './data_pets'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINED_MODEL_PATH = 'models/unet_inpainting_epoch_10.pth' # Or your latest/best model
RESULTS_SAVE_DIR = './test_results'
SHOW_IMAGES_COUNT = 1 # How many comparison images to display directly

# --- 1. U-Net Model Definition (Same as used for training) ---
# (UNetConvBlock, Down, Up, OutConv, UNet class definitions - ensure they are here)
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
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels_in=3, n_channels_out=3, bilinear=True):
        super(UNet, self).__init__()
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

# --- Utility for denormalization ---
def denormalize_imagenet(tensor):
    denorm_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    return torch.clamp(denorm_transform(tensor), 0, 1)

# --- Main Testing Function ---
def test_model():
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"Error: Trained model not found at {TRAINED_MODEL_PATH}")
        return

    if not os.path.exists(RESULTS_SAVE_DIR):
        os.makedirs(RESULTS_SAVE_DIR)

    test_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR, split='test', transform=transform, download=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = UNet(n_channels_in=3, n_channels_out=3).to(DEVICE)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))
    model.eval()

    total_l1_loss, total_psnr, total_ssim, num_samples = 0.0, 0.0, 0.0, 0
    criterion_l1 = nn.L1Loss()
    images_shown_count = 0

    print(f"Starting testing with model: {TRAINED_MODEL_PATH} on {DEVICE}...")
    with torch.no_grad():
        for i, (masked_images, original_images, masks) in enumerate(tqdm(test_dataloader, desc="Testing Progress")):
            masked_images = masked_images.to(DEVICE)
            original_images = original_images.to(DEVICE)
            inpainted_images = model(masked_images)

            loss_l1_batch = criterion_l1(inpainted_images, original_images)
            total_l1_loss += loss_l1_batch.item() * masked_images.size(0)

            batch_size_current = masked_images.size(0)
            for k in range(batch_size_current):
                gt_img = denormalize_imagenet(original_images[k].cpu())
                out_img = denormalize_imagenet(inpainted_images[k].cpu())
                gt_img_np = gt_img.permute(1, 2, 0).numpy()
                out_img_np = out_img.permute(1, 2, 0).numpy()

                psnr = peak_signal_noise_ratio(gt_img_np, out_img_np, data_range=1.0)
                total_psnr += psnr
                ssim = structural_similarity(gt_img_np, out_img_np, channel_axis=-1, data_range=1.0)
                total_ssim += ssim
            num_samples += batch_size_current

            if i < 2: # Save samples from the first 2 batches
                for j in range(min(4, batch_size_current)):
                    idx_in_dataset = i * BATCH_SIZE + j
                    save_masked = denormalize_imagenet(masked_images[j].cpu())
                    save_inpainted = denormalize_imagenet(inpainted_images[j].cpu())
                    save_original = denormalize_imagenet(original_images[j].cpu())
                    save_mask_vis = masks[j].cpu().repeat(3, 1, 1)
                    comparison_img_tensor = torch.cat((save_masked, save_mask_vis, save_inpainted, save_original), dim=2)
                    torchvision.utils.save_image(comparison_img_tensor,
                                                 os.path.join(RESULTS_SAVE_DIR, f"sample_{idx_in_dataset}_comparison.png"))

                    # --- Display Image using Matplotlib ---
                    if images_shown_count < SHOW_IMAGES_COUNT:
                        print(f"\nDisplaying sample {images_shown_count + 1}...")
                        # Matplotlib expects (H, W, C) for RGB
                        img_to_show_np = comparison_img_tensor.permute(1, 2, 0).numpy()
                        plt.figure(figsize=(12, 4)) # Adjust figure size as needed
                        plt.imshow(img_to_show_np)
                        plt.title(f"Sample {idx_in_dataset}: Masked | Mask | Inpainted | Original")
                        plt.axis('off') # Turn off axis numbers and ticks
                        plt.show() # This will pause the script until you close the image window
                        images_shown_count += 1

    avg_l1_loss = total_l1_loss / num_samples
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    print("\n--- Test Results ---")
    print(f"Average L1 Loss: {avg_l1_loss:.4f}")
    print(f"Average PSNR:    {avg_psnr:.2f} dB")
    print(f"Average SSIM:    {avg_ssim:.4f}")
    print(f"Visual samples saved in: {RESULTS_SAVE_DIR}")

if __name__ == '__main__':
    test_model()