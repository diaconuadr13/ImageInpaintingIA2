import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
import numpy as np

# --- Configuration ---
IMG_SIZE = 128  # Resize images to this size (can be 224 or other common sizes too)
BATCH_SIZE = 32 # Adjusted batch size, can be tuned
DATA_DIR = '../data_pets' # Directory to store dataset

# --- 1. Transformations ---
# Oxford-IIIT Pet images have varying sizes.
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize to a fixed square size
    # transforms.CenterCrop(IMG_SIZE), # Optional: if Resize was to a larger dim first
    transforms.ToTensor(), # Converts to [0, 1] range
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats, common for pets
])

# --- 2. Mask Generation Function (same as before) ---
def generate_random_mask(img_tensor, min_mask_ratio=0.15, max_mask_ratio=0.5): # Adjusted ratios slightly
    """
    Generates a random rectangular mask.
    Args:
        img_tensor (torch.Tensor): The input image tensor (C, H, W).
        min_mask_ratio (float): Minimum ratio of mask area to image area.
        max_mask_ratio (float): Maximum ratio of mask area to image area.
    Returns:
        torch.Tensor: A binary mask (1 for masked, 0 for not masked) of the same H, W as img_tensor.
                       Shape: (1, H, W) for broadcasting.
    """
    c, h, w = img_tensor.shape
    mask = torch.zeros((1, h, w), dtype=torch.float32)

    mask_area_ratio = random.uniform(min_mask_ratio, max_mask_ratio)
    mask_area = h * w * mask_area_ratio
    # Allow non-square masks by varying aspect ratio of the mask
    aspect_ratio = random.uniform(0.5, 2.0)
    mask_h = int(np.sqrt(mask_area * aspect_ratio))
    mask_w = int(np.sqrt(mask_area / aspect_ratio))


    # Ensure mask dimensions are within image bounds
    mask_h = min(max(1, mask_h), h -1)
    mask_w = min(max(1, mask_w), w -1)

    if mask_h == 0 or mask_w == 0: # if somehow mask is too small
        return generate_random_mask(img_tensor, min_mask_ratio, max_mask_ratio) # retry

    top_y = random.randint(0, h - mask_h)
    left_x = random.randint(0, w - mask_w)

    mask[:, top_y : top_y + mask_h, left_x : left_x + mask_w] = 1.0
    return mask

# --- 3. Custom Oxford-IIIT Pet Inpainting Dataset ---
class OxfordPetInpaintingDataset(Dataset):
    def __init__(self, root_dir, split='trainval', transform=None, download=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): One of 'trainval' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            download (bool): If true, downloads the dataset from the internet.
        """
        self.transform = transform
        try:
            # We only need the images for inpainting, not the category labels or segmentation masks
            # target_types='category' is default and returns (image, label)
            self.pet_dataset = torchvision.datasets.OxfordIIITPet(
                root=root_dir,
                split=split,
                target_types='category', # Could also be 'image' if available or 'segmentation'
                transform=self.transform, # Apply transform directly here if it's only for the image
                download=download
            )
        except Exception as e:
            print(f"Error initializing OxfordIIITPet dataset: {e}")
            print(f"Please ensure the dataset is correctly downloaded or consider manually downloading it to the '{root_dir}/oxford-iiit-pet/' directory.")
            raise

    def __len__(self):
        return len(self.pet_dataset)

    def __getitem__(self, idx):
        # Load image and its label/segmentation (we'll ignore the second element)
        # The transform provided to OxfordIIITPet constructor is applied to the image
        original_img_tensor, _ = self.pet_dataset[idx] # original_img_tensor is already transformed

        # If transform was not applied by the dataset loader (e.g. if you passed target_types='image')
        # or if you want to ensure it's a tensor for mask generation:
        # if not isinstance(original_img_tensor, torch.Tensor):
        #     if self.transform:
        #        original_img_tensor = self.transform(original_img_tensor) # original_img_pil
        #     else:
        #        original_img_tensor = transforms.ToTensor()(original_img_tensor) # Fallback

        # Generate mask
        mask = generate_random_mask(original_img_tensor)

        # Create masked image
        # Assuming normalization to [-1, 1] or [0,1] handled by transform.
        # Masked areas will be 0 if original values were 0 after (value * (1-mask))
        # Or you can set them to a specific value e.g. gray (0 for [-1,1] or 0.5 for [0,1])
        # masked_img_tensor = original_img_tensor * (1.0 - mask) # Zeros out masked parts
        # Alternative: Set masked parts to gray (0 for [-1,1] normalized images, 0.5 for [0,1] normalized)
        # For ImageNet stats normalization, gray (0.5,0.5,0.5) becomes:
        # (0.5-0.485)/0.229 = 0.0655
        # (0.5-0.456)/0.224 = 0.1964
        # (0.5-0.406)/0.225 = 0.4177
        # So, let's just use the zeroing out method for simplicity or use 0 which is close to gray for [-1,1] if using a different norm.
        # If using the provided ImageNet normalization, 0 in the masked region is fine.
        masked_img_tensor = original_img_tensor * (1.0 - mask)

        return masked_img_tensor, original_img_tensor, mask

# --- 4. Example Usage: DataLoader ---
# --- 4. Example Usage: DataLoader ---
if __name__ == '__main__':
    print("Attempting to download and prepare Oxford-IIIT Pet dataset...")
    try:
        # Define the split to use
        dataset_split = 'trainval' # Or 'test'

        # Create dataset instance
        train_dataset = OxfordPetInpaintingDataset(
            root_dir=DATA_DIR,
            split=dataset_split, # Use the variable here
            transform=transform,
            download=True
        )
        # Corrected print statement:
        print(f"Successfully loaded Oxford-IIIT Pet {dataset_split} split with {len(train_dataset)} samples.")

        # Create DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

        # --- Test the DataLoader ---
        print(f"\nTesting DataLoader with batch size {BATCH_SIZE}...")
        for i, (masked_images, original_images, masks) in enumerate(train_dataloader):
            print(f"Batch {i+1}:")
            print(f"  Masked images shape: {masked_images.shape}")
            print(f"  Original images shape: {original_images.shape}")
            print(f"  Masks shape: {masks.shape}")

            if i == 0:
                denorm_transform = transforms.Compose([
                    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                ])
                sample_masked_img = denorm_transform(masked_images[0].cpu().squeeze())
                sample_original_img = denorm_transform(original_images[0].cpu().squeeze())
                sample_mask = masks[0].cpu().squeeze(0)

                torchvision.utils.save_image(sample_masked_img, 'sample_pet_masked_image.png')
                torchvision.utils.save_image(sample_original_img, 'sample_pet_original_image.png')
                torchvision.utils.save_image(sample_mask, 'sample_pet_mask.png')
                print("\nSaved sample_pet_masked_image.png, sample_pet_original_image.png, and sample_pet_mask.png from the first batch.")

            if i >= 2:
                break
        print("\nDataLoader test complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Tips: Check internet. Ensure disk space. Manually download if persistent issues.")
        print(f"  Expected location after download: {DATA_DIR}/oxford-iiit-pet/")