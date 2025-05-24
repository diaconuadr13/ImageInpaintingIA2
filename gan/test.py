# gan/test.py (Modified for Ablation Study Evaluation - NO PARSER, float32 fix)
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
# Removed argparse
import json # For saving metrics

# Assuming your data_loader.py and UNet.py (for Generator) are accessible
from data_loader import OxfordPetInpaintingDataset, transform #
from UNet import UNet # Generator Model
# If your GAN generator can have depth variants (UNet3, UNet5), import them too.
# from UNet import UNet3, UNet5 

# ==============================================================================
# ======== USER CONFIGURATION: EDIT THESE VALUES BEFORE RUNNING ========
# ==============================================================================
GENERATOR_MODEL_PATH_TO_TEST = './gan_inpainting_results/lambda_150.0/netG_final.pth' #<-- EDIT THIS
GENERATOR_CLASS_NAME_TO_TEST = "UNet" # Options: "UNet", "UNet3", "UNet5" (if applicable for GAN generator) #<-- EDIT THIS
VARIANT_NAME_FOR_OUTPUTS = "GAN_lambda150_DefaultLR_Best" #<-- EDIT THIS (e.g., "GAN_SGD_SGD_LR1e-3_Best")
RESULTS_BASE_DIR = './gan_test_results_ablations_no_parser' # Base directory for these test results
# ==============================================================================

# --- Static Configuration (Defaults) ---
IMG_SIZE_CONFIG = 128
BATCH_SIZE_CONFIG = 16 # Your gan/test.py used 16
DATA_DIR_CONFIG = '../data_pets' # Your gan/test.py used '../data_pets'
DEVICE_CONFIG = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_SAMPLES_DIR = './fixed_test_samples' # Directory where fixed samples are saved

# --- Utility for denormalization ---
def denormalize_imagenet(tensor): #
    denorm_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    return torch.clamp(denorm_transform(tensor), 0, 1)

# --- Main Testing Function ---
def run_gan_test(generator_model_path, generator_class_name, variant_name, results_base_dir):
    print(f"Starting GAN testing for variant: {variant_name}")
    print(f"Generator Model Path: {generator_model_path}")
    print(f"Generator Class: {generator_class_name}")

    if not os.path.exists(generator_model_path):
        print(f"Error: Trained generator model not found at {generator_model_path}")
        return

    variant_results_dir = os.path.join(results_base_dir, variant_name)
    fixed_samples_comparison_dir = os.path.join(variant_results_dir, "fixed_samples_comparison")
    general_test_samples_dir = os.path.join(variant_results_dir, "general_test_samples")

    if not os.path.exists(variant_results_dir):
        os.makedirs(variant_results_dir)
    if not os.path.exists(fixed_samples_comparison_dir):
        os.makedirs(fixed_samples_comparison_dir)
    if not os.path.exists(general_test_samples_dir):
        os.makedirs(general_test_samples_dir)

    # --- Load Generator Model ---
    if generator_class_name == "UNet":
        gen_model_class = UNet
    # Add UNet3, UNet5 conditions if your GAN generator might use these depths
    # elif generator_class_name == "UNet3":
    #     gen_model_class = UNet3 # Make sure UNet3 is imported
    # elif generator_class_name == "UNet5":
    #     gen_model_class = UNet5 # Make sure UNet5 is imported
    else:
        print(f"Warning: Unknown generator class name '{generator_class_name}'. Defaulting to 'UNet'.")
        gen_model_class = UNet

    netG = gen_model_class(n_channels_in=3, n_channels_out=3).to(DEVICE_CONFIG) #
    try:
        netG.load_state_dict(torch.load(generator_model_path, map_location=DEVICE_CONFIG)) #
    except Exception as e:
        print(f"ERROR: Failed to load generator weights: {e}") #
        return
    netG.eval()

    # --- 1. Evaluation on the general test set for metrics ---
    print(f"\nEvaluating GAN generator {variant_name} on general test set for metrics...")
    test_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR_CONFIG, split='test', transform=transform, download=True
    ) #
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE_CONFIG, shuffle=False, num_workers=0
    ) #

    total_l1_loss, total_psnr, total_ssim, num_samples = 0.0, 0.0, 0.0, 0
    criterion_l1 = nn.L1Loss() #

    with torch.no_grad(): #
        for i, (masked_images, original_images, masks) in enumerate(tqdm(test_dataloader, desc=f"Testing GAN {variant_name}")):
            masked_images = masked_images.to(DEVICE_CONFIG)
            original_images = original_images.to(DEVICE_CONFIG)
            generated_images = netG(masked_images)

            loss_l1_batch = criterion_l1(generated_images, original_images)
            total_l1_loss += loss_l1_batch.item() * masked_images.size(0)

            batch_size_current = masked_images.size(0)
            for k in range(batch_size_current):
                gt_img_np = denormalize_imagenet(original_images[k].cpu()).permute(1, 2, 0).numpy()
                out_img_np = denormalize_imagenet(generated_images[k].cpu()).permute(1, 2, 0).numpy()
                
                psnr_val = peak_signal_noise_ratio(gt_img_np, out_img_np, data_range=1.0)
                total_psnr += psnr_val
                
                ssim_val = structural_similarity(gt_img_np, out_img_np, channel_axis=-1, data_range=1.0, win_size=7) # Added win_size
                total_ssim += ssim_val
            num_samples += batch_size_current

            if i < 1: # Save samples from the first general test batch
                for j_gen in range(min(4, batch_size_current)): #
                    idx_in_dataset = i * BATCH_SIZE_CONFIG + j_gen
                    save_masked_gen = denormalize_imagenet(masked_images[j_gen].cpu())
                    save_generated_gen = denormalize_imagenet(generated_images[j_gen].cpu())
                    save_original_gen = denormalize_imagenet(original_images[j_gen].cpu())
                    save_mask_vis_gen = masks[j_gen].cpu().repeat(3, 1, 1)
                    comparison_img_tensor_gen = torch.cat((save_masked_gen, save_mask_vis_gen, save_generated_gen, save_original_gen), dim=2) #
                    torchvision.utils.save_image(comparison_img_tensor_gen,
                                                 os.path.join(general_test_samples_dir, f"general_gan_sample_{idx_in_dataset}_comparison.png"))

    avg_l1_loss = total_l1_loss / num_samples if num_samples > 0 else 0.0
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0

    # FIX: Convert metrics to standard Python floats before adding to dictionary
    metrics = {
        'variant_name': variant_name,
        'generator_model_path': generator_model_path,
        'avg_l1_loss': float(avg_l1_loss),    # <--- CONVERT HERE
        'avg_psnr': float(avg_psnr),        # <--- CONVERT HERE
        'avg_ssim': float(avg_ssim)         # <--- CONVERT HERE
    }
    print(f"\n--- GAN Test Results for {variant_name} (Generator Performance) ---") #
    print(f"Average L1 Loss: {metrics['avg_l1_loss']:.4f}")
    print(f"Average PSNR:    {metrics['avg_psnr']:.2f} dB")
    print(f"Average SSIM:    {metrics['avg_ssim']:.4f}")

    with open(os.path.join(variant_results_dir, 'lambda150.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {os.path.join(variant_results_dir, 'metric_rms.json')}")


    # --- 2. Generate comparisons on the FIXED test samples ---
    print(f"\nGenerating comparisons for GAN {variant_name} on FIXED samples...")
    try:
        fixed_masked_images = torch.load(os.path.join(FIXED_SAMPLES_DIR, 'fixed_masked_images.pt')).to(DEVICE_CONFIG)
        fixed_original_images = torch.load(os.path.join(FIXED_SAMPLES_DIR, 'fixed_original_images.pt')).to(DEVICE_CONFIG)
        # Load masks and ensure they are Bx3xHxW for visualization
        fixed_masks_batch = torch.load(os.path.join(FIXED_SAMPLES_DIR, 'fixed_masks.pt')).cpu()
        if fixed_masks_batch.dim() == 3: 
            if fixed_masks_batch.size(0) == 1 : 
                 fixed_masks_batch = fixed_masks_batch.repeat(fixed_masked_images.size(0), 3, 1, 1) if fixed_masked_images.size(0) > 1 else fixed_masks_batch.repeat(3,1,1).unsqueeze(0)
            else: 
                 fixed_masks_batch = fixed_masks_batch.unsqueeze(0).repeat(fixed_masked_images.size(0), 3, 1, 1) 
        elif fixed_masks_batch.dim() == 4 and fixed_masks_batch.size(1) == 1: 
            fixed_masks_batch = fixed_masks_batch.repeat(1,3,1,1) 
    except FileNotFoundError:
        print(f"Error: Fixed sample files not found in {FIXED_SAMPLES_DIR}. Skipping fixed sample comparison.")
        return

    with torch.no_grad():
        generated_fixed_images = netG(fixed_masked_images)
        for j in range(fixed_masked_images.size(0)):
            s_masked = denormalize_imagenet(fixed_masked_images[j].cpu())
            s_generated = denormalize_imagenet(generated_fixed_images[j].cpu())
            s_original = denormalize_imagenet(fixed_original_images[j].cpu())
            s_mask_vis = fixed_masks_batch[j]
            if s_mask_vis.dim() == 4: s_mask_vis = s_mask_vis.squeeze(0) # If BxCxHxW and B=1
            if s_mask_vis.size(0) == 1: s_mask_vis = s_mask_vis.repeat(3,1,1) # If 1xHxW, make 3xHxW

            comparison_tensor = torch.cat((s_masked, s_mask_vis, s_generated, s_original), dim=2)
            img_path = os.path.join(fixed_samples_comparison_dir, f"fixed_sample_{j}_{variant_name}_comparison.png")
            torchvision.utils.save_image(comparison_tensor, img_path)
    print(f"Fixed sample comparison images saved to {fixed_samples_comparison_dir}")
    print(f"\nFinished GAN testing for {variant_name}.")


if __name__ == '__main__':
    # Call the main testing function with the hardcoded/editable values from the top
    run_gan_test(
        generator_model_path=GENERATOR_MODEL_PATH_TO_TEST,
        generator_class_name=GENERATOR_CLASS_NAME_TO_TEST,
        variant_name=VARIANT_NAME_FOR_OUTPUTS,
        results_base_dir=RESULTS_BASE_DIR
    )