import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import cv2 # OpenCV
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
# --- Assuming your data_loader.py is accessible ---
# It should define OxfordPetInpaintingDataset and the 'transform' object
# that normalizes images (e.g., to [-1, 1] or [0, 1]) and generates masks.
from gan.data_loader import OxfordPetInpaintingDataset, transform

# --- Configuration ---
IMG_SIZE = 128 # Should match your dataset preparation
BATCH_SIZE = 16 # Can be adjusted for testing; will process one by one for OpenCV
DATA_DIR = './data_pets'
DEVICE = torch.device("cpu") # OpenCV runs on CPU; PyTorch tensors will be moved to CPU
RESULTS_SAVE_DIR = './opencv_inpainting_results'
SHOW_IMAGES_COUNT = 1 # How many comparison images to display directly
INPAINT_RADIUS = 3 # Radius for OpenCV inpainting algorithms

# --- Define ImageNet mean and std ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Correct Denormalization Transform ---
# This transform takes an ImageNet-normalized tensor and converts it to a [0,1] tensor
denormalize_transform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in IMAGENET_STD]),
    transforms.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1., 1., 1.]),
])

if not os.path.exists(RESULTS_SAVE_DIR):
    os.makedirs(RESULTS_SAVE_DIR)

# --- Utility for denormalization and conversion ---
# Adjust this if your 'transform' normalizes differently
def tensor_to_cv2_img(tensor_img):
    """Converts a PyTorch tensor (CHW, normalized) to an OpenCV image (HWC, BGR, 0-255)."""
    # Denormalize using the correct ImageNet denormalization
    img_denorm = denormalize_transform(tensor_img.cpu().detach()).clamp(0, 1)
    
    img_np = img_denorm.permute(1, 2, 0).numpy() * 255 # HWC, 0-255
    img_cv2 = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV
    return img_cv2

def mask_tensor_to_cv2_mask(mask_tensor):
    """Converts a PyTorch binary mask tensor (1HW or HW, 0 or 1) to OpenCV mask (HW, uint8, 0 or 255)."""
    mask_np = mask_tensor.cpu().detach().squeeze().numpy() # HW
    # Ensure mask is 0 for background, 255 for hole
    cv2_mask = (mask_np * 255).astype(np.uint8)
    return cv2_mask

def cv2_img_to_tensor(cv2_img):
    """Converts an OpenCV image (HWC, BGR, 0-255) back to a PyTorch tensor (CHW, 0-1)."""
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    # Apply a simple ToTensor, assuming no complex normalization needed for metric comparison
    # against an already denormalized ground truth.
    # If comparing against normalized ground truth, apply the same normalization.
    return transforms.ToTensor()(img_pil) # CHW, [0,1]


# --- Main Testing Function ---
def test_opencv_inpainting():
    print(f"Starting OpenCV inpainting test...")

    # --- Load Test Dataset ---
    # OxfordPetInpaintingDataset should yield (masked_image, original_image, mask)
    # where mask is binary (0s and 1s)
    test_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR,
        split='test',
        transform=transform, # Your global transform object from data_loader.py
        download=True # Set to False if already downloaded
    )
    # DataLoader is used just to iterate easily, OpenCV processes one by one
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0 # Process one image at a time
    )

    # --- Initialize Metrics ---
    metrics_telea = {'l1': [], 'psnr': [], 'ssim': []}
    metrics_ns = {'l1': [], 'psnr': [], 'ssim': []}
    metrics_mean_fill = {'l1': [], 'psnr': [], 'ssim': []} # New metrics for mean fill
    
    criterion_l1 = nn.L1Loss()
    images_shown_count = 0

    for i, (masked_images_batch, original_images_batch, masks_batch) in enumerate(tqdm(test_dataloader, desc="OpenCV Testing")):
        # Process one image at a time from the batch
        original_image_tensor = original_images_batch[0].to(DEVICE) # CHW, normalized
        mask_tensor = masks_batch[0].to(DEVICE)                     # 1HW or HW, binary 0/1

        # For OpenCV, we need the original image before masking, and the mask
        # The 'masked_image' from dataloader is original * (1-mask)
        # We can reconstruct the original unmasked image for OpenCV input if needed,
        # or use the original_image_tensor directly and apply the mask to it for inpainting.
        # OpenCV's inpaint function takes the "damaged image" (original with holes) and the mask.
        # Let's create the damaged image for OpenCV:
        original_cv2 = tensor_to_cv2_img(original_image_tensor) # HWC, BGR, uint8
        mask_cv2 = mask_tensor_to_cv2_mask(mask_tensor)       # HW, uint8 (0 or 255)

        # Create the damaged image by blacking out masked regions (OpenCV expects BGR)
        # This is essentially what your 'masked_images_batch[0]' would look like if converted to CV2
        damaged_img_cv2 = original_cv2.copy()
        damaged_img_cv2[mask_cv2 == 255] = 0 # Black out the holes

        # --- Apply Telea Inpainting ---
        inpainted_telea_cv2 = cv2.inpaint(damaged_img_cv2, mask_cv2, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        
        # --- Apply Navier-Stokes (NS) Inpainting ---
        inpainted_ns_cv2 = cv2.inpaint(damaged_img_cv2, mask_cv2, INPAINT_RADIUS, cv2.INPAINT_NS)

        # --- Apply Mean Color Fill ---
        inpainted_mean_fill_cv2 = damaged_img_cv2.copy()
        # Calculate mean color of the unmasked region from the original_cv2
        # Ensure mask_cv2_inverted is 0 for hole, 1 for background for selecting pixels
        mask_cv2_inverted_for_mean = cv2.bitwise_not(mask_cv2)
        mean_color = cv2.mean(original_cv2, mask=mask_cv2_inverted_for_mean)[:3] # Get B, G, R
        inpainted_mean_fill_cv2[mask_cv2 == 255] = mean_color

        # --- Convert results back to PyTorch tensors for metric calculation ---
        # We need to compare against the original image tensor in the [0,1] range
        # Ensure original_image_tensor_metric is also in [0,1] range
        original_image_tensor_metric = denormalize_transform(original_image_tensor.cpu().detach()).clamp(0,1)
        # If original was [0,1]:
        # original_image_tensor_metric = original_image_tensor.cpu().detach().clamp(0,1)


        inpainted_telea_tensor = cv2_img_to_tensor(inpainted_telea_cv2).to(DEVICE) # CHW, [0,1]
        inpainted_ns_tensor = cv2_img_to_tensor(inpainted_ns_cv2).to(DEVICE)       # CHW, [0,1]
        inpainted_mean_fill_tensor = cv2_img_to_tensor(inpainted_mean_fill_cv2).to(DEVICE) # CHW, [0,1]

        # --- Calculate Metrics for Telea ---
        metrics_telea['l1'].append(criterion_l1(inpainted_telea_tensor, original_image_tensor_metric).item())
        # For PSNR/SSIM, convert to NumPy arrays [0,1] range, HWC
        gt_np = original_image_tensor_metric.cpu().permute(1, 2, 0).numpy()
        telea_np = inpainted_telea_tensor.cpu().permute(1, 2, 0).numpy()
        metrics_telea['psnr'].append(peak_signal_noise_ratio(gt_np, telea_np, data_range=1.0))
        metrics_telea['ssim'].append(structural_similarity(gt_np, telea_np, channel_axis=-1, data_range=1.0, win_size=7))


        # --- Calculate Metrics for NS ---
        ns_np = inpainted_ns_tensor.cpu().permute(1, 2, 0).numpy()
        metrics_ns['l1'].append(criterion_l1(inpainted_ns_tensor, original_image_tensor_metric).item())
        metrics_ns['psnr'].append(peak_signal_noise_ratio(gt_np, ns_np, data_range=1.0))
        metrics_ns['ssim'].append(structural_similarity(gt_np, ns_np, channel_axis=-1, data_range=1.0, win_size=7))

        # --- Calculate Metrics for Mean Fill ---
        mean_fill_np = inpainted_mean_fill_tensor.cpu().permute(1, 2, 0).numpy()
        metrics_mean_fill['l1'].append(criterion_l1(inpainted_mean_fill_tensor, original_image_tensor_metric).item())
        metrics_mean_fill['psnr'].append(peak_signal_noise_ratio(gt_np, mean_fill_np, data_range=1.0))
        metrics_mean_fill['ssim'].append(structural_similarity(gt_np, mean_fill_np, channel_axis=-1, data_range=1.0, win_size=7))


        # --- Save and/or Display some visual samples ---
        if i < 5 or images_shown_count < SHOW_IMAGES_COUNT : # Save first 5, show up to SHOW_IMAGES_COUNT
            # Prepare tensors for saving grid (denormalized, CHW)
            save_original = original_image_tensor_metric.cpu() # Already [0,1]
            save_masked = cv2_img_to_tensor(damaged_img_cv2).cpu() # Convert damaged BGR to RGB tensor [0,1]
            save_mask_vis = mask_tensor.cpu().repeat(3, 1, 1) # Make mask 3-channel
            save_telea = inpainted_telea_tensor.cpu()
            save_ns = inpainted_ns_tensor.cpu()
            save_mean_fill = inpainted_mean_fill_tensor.cpu() # Prepare mean_fill tensor

            comparison_img_tensor = torch.cat((
                save_original.unsqueeze(0),
                save_masked.unsqueeze(0),
                save_mask_vis.unsqueeze(0),
                save_telea.unsqueeze(0),
                save_ns.unsqueeze(0),
                save_mean_fill.unsqueeze(0) # Add mean_fill to comparison
            ), dim=0) # Stack along batch dimension for make_grid

            grid = torchvision.utils.make_grid(comparison_img_tensor, nrow=6, padding=5) # Adjusted nrow
            torchvision.utils.save_image(grid,
                                         os.path.join(RESULTS_SAVE_DIR, f"opencv_sample_{i}_comparison.png"))

            if images_shown_count < SHOW_IMAGES_COUNT:
                print(f"\\nDisplaying OpenCV sample {images_shown_count + 1}...")
                img_to_show_np = grid.permute(1, 2, 0).numpy()
                plt.figure(figsize=(24, 5)) # Adjusted figsize
                plt.imshow(img_to_show_np)
                plt.title(f"OpenCV Sample {i}: Original | Masked | Mask | Telea | NS | Mean Fill") # Adjusted title
                plt.axis('off')
                plt.show()
                images_shown_count += 1
        
        if i >= 100 and SHOW_IMAGES_COUNT == 0 : # Limit processing for quick test if not showing images
            print("Stopping early after 100 samples for quick metric check (as SHOW_IMAGES_COUNT is 0).")
            # break # Uncomment to run on fewer samples for a quick test

    # --- Calculate Average Metrics ---
    avg_metrics_telea = {k: np.mean(v) for k, v in metrics_telea.items()}
    avg_metrics_ns = {k: np.mean(v) for k, v in metrics_ns.items()}
    avg_metrics_mean_fill = {k: np.mean(v) for k, v in metrics_mean_fill.items()} # Avg for mean_fill

    print("\\n--- OpenCV Inpainting Test Results ---")
    print("\nTelea Algorithm:")
    print(f"  Average L1 Loss: {avg_metrics_telea['l1']:.4f}")
    print(f"  Average PSNR:    {avg_metrics_telea['psnr']:.2f} dB")
    print(f"  Average SSIM:    {avg_metrics_telea['ssim']:.4f}")

    print("\nNavier-Stokes (NS) Algorithm:")
    print(f"  Average L1 Loss: {avg_metrics_ns['l1']:.4f}")
    print(f"  Average PSNR:    {avg_metrics_ns['psnr']:.2f} dB")
    print(f"  Average SSIM:    {avg_metrics_ns['ssim']:.4f}")

    print("\nMean Color Fill Algorithm:") # New print section for mean_fill
    print(f"  Average L1 Loss: {avg_metrics_mean_fill['l1']:.4f}")
    print(f"  Average PSNR:    {avg_metrics_mean_fill['psnr']:.2f} dB")
    print(f"  Average SSIM:    {avg_metrics_mean_fill['ssim']:.4f}")

    print(f"\\nVisual samples saved in: {RESULTS_SAVE_DIR}")

if __name__ == '__main__':
    # Ensure Pillow is imported if cv2_img_to_tensor uses it
    from PIL import Image 
    test_opencv_inpainting()