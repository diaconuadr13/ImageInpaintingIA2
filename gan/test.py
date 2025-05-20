import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

# --- Import your data loader and U-Net (Generator) class ---
from data_loader import OxfordPetInpaintingDataset, transform # Your existing data loader
from UNet import UNet 

# --- Configuration ---
IMG_SIZE = 128
BATCH_SIZE = 16
DATA_DIR = '../data_pets'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINED_GENERATOR_PATH = './gan_inpainting_results/netG_final.pth' # Path to your saved generator
RESULTS_SAVE_DIR = './gan_test_results'
SHOW_IMAGES_COUNT = 1

# --- Utility for denormalization ---
def denormalize_imagenet(tensor):
    denorm_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    return torch.clamp(denorm_transform(tensor), 0, 1)

# --- Main Testing Function ---
def test_gan_generator():
    if not os.path.exists(TRAINED_GENERATOR_PATH):
        print(f"Error: Trained generator model not found at {TRAINED_GENERATOR_PATH}")
        return

    if not os.path.exists(RESULTS_SAVE_DIR):
        os.makedirs(RESULTS_SAVE_DIR)

    test_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR,
        split='test',
        transform=transform,
        download=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    netG = UNet(n_channels_in=3, n_channels_out=3).to(DEVICE)

    try:
        netG.load_state_dict(torch.load(TRAINED_GENERATOR_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"ERROR: Failed to load generator weights: {e}")
        return

    netG.eval()

    total_l1_loss, total_psnr, total_ssim, num_samples = 0.0, 0.0, 0.0, 0
    criterion_l1 = nn.L1Loss()
    images_shown_count = 0

    print(f"Starting GAN testing with generator: {TRAINED_GENERATOR_PATH} on {DEVICE}...")
    with torch.no_grad():
        for i, (masked_images, original_images, masks) in enumerate(tqdm(test_dataloader, desc="GAN Testing Progress")):
            masked_images = masked_images.to(DEVICE)
            original_images = original_images.to(DEVICE)
            generated_images = netG(masked_images)

            loss_l1_batch = criterion_l1(generated_images, original_images)
            total_l1_loss += loss_l1_batch.item() * masked_images.size(0)

            batch_size_current = masked_images.size(0)
            for k in range(batch_size_current):
                gt_img = denormalize_imagenet(original_images[k].cpu())
                out_img = denormalize_imagenet(generated_images[k].cpu())
                gt_img_np = gt_img.permute(1, 2, 0).numpy()
                out_img_np = out_img.permute(1, 2, 0).numpy()

                psnr = peak_signal_noise_ratio(gt_img_np, out_img_np, data_range=1.0)
                total_psnr += psnr
                ssim = structural_similarity(gt_img_np, out_img_np, channel_axis=-1, data_range=1.0)
                total_ssim += ssim
            num_samples += batch_size_current

            if i < 2:
                for j in range(min(4, batch_size_current)):
                    idx_in_dataset = i * BATCH_SIZE + j
                    save_masked = denormalize_imagenet(masked_images[j].cpu())
                    save_generated = denormalize_imagenet(generated_images[j].cpu())
                    save_original = denormalize_imagenet(original_images[j].cpu())
                    save_mask_vis = masks[j].cpu().repeat(3, 1, 1)

                    comparison_img_tensor = torch.cat((save_masked, save_mask_vis, save_generated, save_original), dim=2)
                    torchvision.utils.save_image(comparison_img_tensor,
                                                 os.path.join(RESULTS_SAVE_DIR, f"gan_sample_{idx_in_dataset}_comparison.png"))

                    if images_shown_count < SHOW_IMAGES_COUNT:
                        img_to_show_np = comparison_img_tensor.permute(1, 2, 0).numpy()
                        plt.figure(figsize=(16, 4))
                        plt.imshow(img_to_show_np)
                        plt.title(f"GAN Sample {idx_in_dataset}: Masked | Mask | Generated | Original")
                        plt.axis('off')
                        plt.show()
                        images_shown_count += 1

    avg_l1_loss = total_l1_loss / num_samples if num_samples > 0 else 0
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0

    print("\n--- GAN Test Results (Generator Performance) ---")
    print(f"Average L1 Loss: {avg_l1_loss:.4f}")
    print(f"Average PSNR:    {avg_psnr:.2f} dB")
    print(f"Average SSIM:    {avg_ssim:.4f}")
    print(f"Visual samples saved in: {RESULTS_SAVE_DIR}")

if __name__ == '__main__':
    # If using num_workers > 0 in DataLoader on Windows, ensure freeze_support if needed
    # from multiprocessing import freeze_support
    # freeze_support() # Make sure to import it if you use it
    test_gan_generator()