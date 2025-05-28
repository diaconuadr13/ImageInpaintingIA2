import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import cv2 
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from gan.data_loader import OxfordPetInpaintingDataset, transform

IMG_SIZE = 128 
BATCH_SIZE = 16 
DATA_DIR = './data_pets'
DEVICE = torch.device("cpu") 
RESULTS_SAVE_DIR = './opencv_inpainting_results'
SHOW_IMAGES_COUNT = 1 
INPAINT_RADIUS = 3 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

denormalize_transform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in IMAGENET_STD]),
    transforms.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1., 1., 1.]),
])

if not os.path.exists(RESULTS_SAVE_DIR):
    os.makedirs(RESULTS_SAVE_DIR)

def tensor_to_cv2_img(tensor_img):
    """Converts a PyTorch tensor (CHW, normalized) to an OpenCV image (HWC, BGR, 0-255)."""
    img_denorm = denormalize_transform(tensor_img.cpu().detach()).clamp(0, 1)
    
    img_np = img_denorm.permute(1, 2, 0).numpy() * 255 
    img_cv2 = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR) 
    return img_cv2

def mask_tensor_to_cv2_mask(mask_tensor):
    """Converts a PyTorch binary mask tensor (1HW or HW, 0 or 1) to OpenCV mask (HW, uint8, 0 or 255)."""
    mask_np = mask_tensor.cpu().detach().squeeze().numpy() 
    cv2_mask = (mask_np * 255).astype(np.uint8)
    return cv2_mask

def cv2_img_to_tensor(cv2_img):
    """Converts an OpenCV image (HWC, BGR, 0-255) back to a PyTorch tensor (CHW, 0-1)."""
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return transforms.ToTensor()(img_pil)


def test_opencv_inpainting():
    print(f"Starting OpenCV inpainting test...")

    test_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR,
        split='test',
        transform=transform, 
        download=True 
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0 
    )

    metrics_telea = {'l1': [], 'psnr': [], 'ssim': []}
    metrics_ns = {'l1': [], 'psnr': [], 'ssim': []}
    metrics_mean_fill = {'l1': [], 'psnr': [], 'ssim': []} 
    
    criterion_l1 = nn.L1Loss()
    images_shown_count = 0

    for i, (masked_images_batch, original_images_batch, masks_batch) in enumerate(tqdm(test_dataloader, desc="OpenCV Testing")):
        original_image_tensor = original_images_batch[0].to(DEVICE) 
        mask_tensor = masks_batch[0].to(DEVICE)                     
    
        original_cv2 = tensor_to_cv2_img(original_image_tensor) 
        mask_cv2 = mask_tensor_to_cv2_mask(mask_tensor)      

        damaged_img_cv2 = original_cv2.copy()
        damaged_img_cv2[mask_cv2 == 255] = 0

        inpainted_telea_cv2 = cv2.inpaint(damaged_img_cv2, mask_cv2, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        
        inpainted_ns_cv2 = cv2.inpaint(damaged_img_cv2, mask_cv2, INPAINT_RADIUS, cv2.INPAINT_NS)

        inpainted_mean_fill_cv2 = damaged_img_cv2.copy()

        mask_cv2_inverted_for_mean = cv2.bitwise_not(mask_cv2)
        mean_color = cv2.mean(original_cv2, mask=mask_cv2_inverted_for_mean)[:3] 
        inpainted_mean_fill_cv2[mask_cv2 == 255] = mean_color

        original_image_tensor_metric = denormalize_transform(original_image_tensor.cpu().detach()).clamp(0,1)

        inpainted_telea_tensor = cv2_img_to_tensor(inpainted_telea_cv2).to(DEVICE) 
        inpainted_ns_tensor = cv2_img_to_tensor(inpainted_ns_cv2).to(DEVICE)       
        inpainted_mean_fill_tensor = cv2_img_to_tensor(inpainted_mean_fill_cv2).to(DEVICE)

        metrics_telea['l1'].append(criterion_l1(inpainted_telea_tensor, original_image_tensor_metric).item())
        gt_np = original_image_tensor_metric.cpu().permute(1, 2, 0).numpy()
        telea_np = inpainted_telea_tensor.cpu().permute(1, 2, 0).numpy()
        metrics_telea['psnr'].append(peak_signal_noise_ratio(gt_np, telea_np, data_range=1.0))
        metrics_telea['ssim'].append(structural_similarity(gt_np, telea_np, channel_axis=-1, data_range=1.0, win_size=7))


        ns_np = inpainted_ns_tensor.cpu().permute(1, 2, 0).numpy()
        metrics_ns['l1'].append(criterion_l1(inpainted_ns_tensor, original_image_tensor_metric).item())
        metrics_ns['psnr'].append(peak_signal_noise_ratio(gt_np, ns_np, data_range=1.0))
        metrics_ns['ssim'].append(structural_similarity(gt_np, ns_np, channel_axis=-1, data_range=1.0, win_size=7))

        mean_fill_np = inpainted_mean_fill_tensor.cpu().permute(1, 2, 0).numpy()
        metrics_mean_fill['l1'].append(criterion_l1(inpainted_mean_fill_tensor, original_image_tensor_metric).item())
        metrics_mean_fill['psnr'].append(peak_signal_noise_ratio(gt_np, mean_fill_np, data_range=1.0))
        metrics_mean_fill['ssim'].append(structural_similarity(gt_np, mean_fill_np, channel_axis=-1, data_range=1.0, win_size=7))


        if i < 5 or images_shown_count < SHOW_IMAGES_COUNT : 
            save_original = original_image_tensor_metric.cpu() 
            save_masked = cv2_img_to_tensor(damaged_img_cv2).cpu() 
            save_mask_vis = mask_tensor.cpu().repeat(3, 1, 1)
            save_telea = inpainted_telea_tensor.cpu()
            save_ns = inpainted_ns_tensor.cpu()
            save_mean_fill = inpainted_mean_fill_tensor.cpu() 

            comparison_img_tensor = torch.cat((
                save_masked.unsqueeze(0),
                save_telea.unsqueeze(0),
                save_ns.unsqueeze(0),
                save_mean_fill.unsqueeze(0),
                save_original.unsqueeze(0)
            ), dim=0)

            grid = torchvision.utils.make_grid(comparison_img_tensor, nrow=6, padding=5)
            torchvision.utils.save_image(grid,
                                         os.path.join(RESULTS_SAVE_DIR, f"opencv_sample_{i}_comparison.png"))

            if images_shown_count < SHOW_IMAGES_COUNT:
                print(f"\\nDisplaying OpenCV sample {images_shown_count + 1}...")
                img_to_show_np = grid.permute(1, 2, 0).numpy()
                plt.figure(figsize=(24, 5))
                plt.imshow(img_to_show_np)
                plt.axis('off')
                plt.show()
                images_shown_count += 1
        
        if i >= 100 and SHOW_IMAGES_COUNT == 0 : 
            print("Stopping early after 100 samples for quick metric check (as SHOW_IMAGES_COUNT is 0).")

    avg_metrics_telea = {k: np.mean(v) for k, v in metrics_telea.items()}
    avg_metrics_ns = {k: np.mean(v) for k, v in metrics_ns.items()}
    avg_metrics_mean_fill = {k: np.mean(v) for k, v in metrics_mean_fill.items()} 

    print("\\n--- OpenCV Inpainting Test Results ---")
    print("\nTelea Algorithm:")
    print(f"  Average L1 Loss: {avg_metrics_telea['l1']:.4f}")
    print(f"  Average PSNR:    {avg_metrics_telea['psnr']:.2f} dB")
    print(f"  Average SSIM:    {avg_metrics_telea['ssim']:.4f}")

    print("\nNavier-Stokes (NS) Algorithm:")
    print(f"  Average L1 Loss: {avg_metrics_ns['l1']:.4f}")
    print(f"  Average PSNR:    {avg_metrics_ns['psnr']:.2f} dB")
    print(f"  Average SSIM:    {avg_metrics_ns['ssim']:.4f}")

    print("\nMean Color Fill Algorithm:") 
    print(f"  Average L1 Loss: {avg_metrics_mean_fill['l1']:.4f}")
    print(f"  Average PSNR:    {avg_metrics_mean_fill['psnr']:.2f} dB")
    print(f"  Average SSIM:    {avg_metrics_mean_fill['ssim']:.4f}")

    print(f"\\nVisual samples saved in: {RESULTS_SAVE_DIR}")

if __name__ == '__main__':
    from PIL import Image 
    test_opencv_inpainting()