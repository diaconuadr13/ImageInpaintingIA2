# UNet/test.py (Modified for Ablation Study Evaluation - NO PARSER)
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

# Assuming your data_loader.py and UNet.py are in the same directory or accessible
from data_loader import OxfordPetInpaintingDataset, transform #
# You'll need to import the UNet model class(es) correctly.
# If UNet_depth3, UNet_depth5 etc. are in the same UNet.py file:
from UNet import UNet, UNet3, UNet5 # Assuming these are defined in UNet.py
# Or, if you are testing a specific model type each time:
# from UNet import UNet as ModelToTest

# ==============================================================================
# ======== USER CONFIGURATION: EDIT THESE VALUES BEFORE RUNNING ========
# ==============================================================================
MODEL_PATH_TO_TEST = './models_unet_depth_ablation/Depth4/unet_inpainting_final_Depth4.pth' #<-- EDIT THIS
MODEL_CLASS_NAME_TO_TEST = "UNet3"  # Options: "UNet", "UNet3", "UNet5" #<-- EDIT THIS (must match the model type in MODEL_PATH_TO_TEST)
VARIANT_NAME_FOR_OUTPUTS = "Depth3_Default" #<-- EDIT THIS (e.g., "Depth4_Adam", "Depth3_SGD")
RESULTS_BASE_DIR = './test_results_unet_ablations_no_parser' # Base directory for results
# ==============================================================================

# --- Static Configuration (Defaults) ---
IMG_SIZE_CONFIG = 128 # Ensure consistency with training
BATCH_SIZE_CONFIG = 16
DATA_DIR_CONFIG = '../data_pets'
DEVICE_CONFIG = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_SAMPLES_DIR = './fixed_test_samples'

# --- Utility for denormalization ---
def denormalize_imagenet(tensor): #
    denorm_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])
    return torch.clamp(denorm_transform(tensor), 0, 1)

# --- Main Testing Function (now takes parameters directly) ---
def run_test(model_path, model_class_name, variant_name, results_base_dir):
    print(f"Starting testing for variant: {variant_name}")
    print(f"Model Path: {model_path}")
    print(f"Model Class: {model_class_name}")

    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
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

    # --- Load Model ---
    if model_class_name == "UNet":
        model_class = UNet
    elif model_class_name == "UNet3":
        model_class = UNet3
    elif model_class_name == "UNet5":
        model_class = UNet5
    else:
        print(f"Error: Unknown model class name '{model_class_name}'. Available: 'UNet', 'UNet3', 'UNet5'.")
        return
        
    model = model_class(n_channels_in=3, n_channels_out=3).to(DEVICE_CONFIG)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE_CONFIG))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return
    model.eval()

    # --- 1. Evaluation on the general test set for metrics ---
    print(f"\nEvaluating {variant_name} on general test set for metrics...")
    test_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR_CONFIG, split='test', transform=transform, download=True
    ) #
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE_CONFIG, shuffle=False, num_workers=0
    ) #

    total_l1_loss, total_psnr, total_ssim, num_samples = 0.0, 0.0, 0.0, 0
    criterion_l1 = nn.L1Loss() #

    with torch.no_grad(): #
        for i, (masked_images, original_images, masks) in enumerate(tqdm(test_dataloader, desc=f"Testing {variant_name}")):
            masked_images = masked_images.to(DEVICE_CONFIG)
            original_images = original_images.to(DEVICE_CONFIG)
            inpainted_images = model(masked_images)

            loss_l1_batch = criterion_l1(inpainted_images, original_images)
            total_l1_loss += loss_l1_batch.item() * masked_images.size(0)

            batch_size_current = masked_images.size(0)
            for k in range(batch_size_current):
                gt_img_np = denormalize_imagenet(original_images[k].cpu()).permute(1, 2, 0).numpy()
                out_img_np = denormalize_imagenet(inpainted_images[k].cpu()).permute(1, 2, 0).numpy()
                
                psnr = peak_signal_noise_ratio(gt_img_np, out_img_np, data_range=1.0)
                total_psnr += psnr
                ssim = structural_similarity(gt_img_np, out_img_np, channel_axis=-1, data_range=1.0, win_size=7)
                total_ssim += ssim
            num_samples += batch_size_current

            if i < 1: 
                for j_gen in range(min(4, batch_size_current)):
                    idx_in_dataset = i * BATCH_SIZE_CONFIG + j_gen
                    save_masked_gen = denormalize_imagenet(masked_images[j_gen].cpu())
                    save_inpainted_gen = denormalize_imagenet(inpainted_images[j_gen].cpu())
                    save_original_gen = denormalize_imagenet(original_images[j_gen].cpu())
                    save_mask_vis_gen = masks[j_gen].cpu().repeat(3, 1, 1)
                    comparison_img_tensor_gen = torch.cat((save_masked_gen, save_mask_vis_gen, save_inpainted_gen, save_original_gen), dim=2)
                    torchvision.utils.save_image(comparison_img_tensor_gen,
                                                 os.path.join(general_test_samples_dir, f"general_sample_{idx_in_dataset}_comparison.png"))

    avg_l1_loss = total_l1_loss / num_samples if num_samples > 0 else 0
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0

    metrics = {
        'variant_name': variant_name,
        'model_path': model_path,
        'avg_l1_loss': float(avg_l1_loss),   
        'avg_psnr': float(avg_psnr),       
        'avg_ssim': float(avg_ssim)      
    }


    print(f"\n--- Test Results for {variant_name} ---")
    print(f"Average L1 Loss: {avg_l1_loss:.4f}")
    print(f"Average PSNR:    {avg_psnr:.2f} dB")
    print(f"Average SSIM:    {avg_ssim:.4f}")

    with open(os.path.join(variant_results_dir, 'metrics3.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {os.path.join(variant_results_dir, 'metrics.json')}")

    # --- 2. Generate comparisons on the FIXED test samples ---
    print(f"\nGenerating comparisons for {variant_name} on FIXED samples...")
    try:
        fixed_masked_images = torch.load(os.path.join(FIXED_SAMPLES_DIR, 'fixed_masked_images.pt')).to(DEVICE_CONFIG)
        fixed_original_images = torch.load(os.path.join(FIXED_SAMPLES_DIR, 'fixed_original_images.pt')).to(DEVICE_CONFIG)
        # Load masks and ensure they are Bx3xHxW for visualization
        fixed_masks_batch = torch.load(os.path.join(FIXED_SAMPLES_DIR, 'fixed_masks.pt')).cpu()
        if fixed_masks_batch.dim() == 3: # HxW or 1xHxW (if single mask saved)
            if fixed_masks_batch.size(0) == 1 : # 1xHxW
                 fixed_masks_batch = fixed_masks_batch.repeat(fixed_masked_images.size(0), 3, 1, 1) if fixed_masked_images.size(0) > 1 else fixed_masks_batch.repeat(3,1,1).unsqueeze(0)
            else: # HxW - this case is unlikely for a saved batch
                 fixed_masks_batch = fixed_masks_batch.unsqueeze(0).repeat(fixed_masked_images.size(0), 3, 1, 1) # Bx3xHxW
        elif fixed_masks_batch.dim() == 4 and fixed_masks_batch.size(1) == 1: # Bx1xHxW
            fixed_masks_batch = fixed_masks_batch.repeat(1,3,1,1) # Bx3xHxW
        # else: assume it's Bx3xHxW or BxHxW (latter will fail repeat)
        
    except FileNotFoundError:
        print(f"Error: Fixed sample files not found in {FIXED_SAMPLES_DIR}. Skipping fixed sample comparison.")
        return

    with torch.no_grad():
        inpainted_fixed_images = model(fixed_masked_images)
        for j in range(fixed_masked_images.size(0)):
            s_masked = denormalize_imagenet(fixed_masked_images[j].cpu())
            s_inpainted = denormalize_imagenet(inpainted_fixed_images[j].cpu())
            s_original = denormalize_imagenet(fixed_original_images[j].cpu())
            # Get the j-th mask, ensure it's 3xHxW
            s_mask_vis = fixed_masks_batch[j] 
            if s_mask_vis.dim() == 4: s_mask_vis = s_mask_vis.squeeze(0) # if it was BxCxHxW with B=1
            if s_mask_vis.size(0) == 1: s_mask_vis = s_mask_vis.repeat(3,1,1) # if it's 1xHxW

            comparison_tensor = torch.cat((s_masked, s_mask_vis, s_inpainted, s_original), dim=2)
            img_path = os.path.join(fixed_samples_comparison_dir, f"fixed_sample_{j}_{variant_name}_comparison.png")
            torchvision.utils.save_image(comparison_tensor, img_path)
    print(f"Fixed sample comparison images saved to {fixed_samples_comparison_dir}")
    print(f"\nFinished testing for {variant_name}.")


if __name__ == '__main__':
    # Call the main testing function with the hardcoded/editable values from the top
    run_test(
        model_path=MODEL_PATH_TO_TEST,
        model_class_name=MODEL_CLASS_NAME_TO_TEST,
        variant_name=VARIANT_NAME_FOR_OUTPUTS,
        results_base_dir=RESULTS_BASE_DIR
    )