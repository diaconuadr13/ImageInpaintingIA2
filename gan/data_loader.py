import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
import numpy as np

# --- Configuration ---
IMG_SIZE = 128  
BATCH_SIZE = 32 
DATA_DIR = '../data_pets' 

# --- 1. Transformations ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# --- 2. Mask Generation Function (same as before) ---
def generate_random_mask(img_tensor, min_mask_ratio=0.10, max_mask_ratio=0.10): 
    c, h, w = img_tensor.shape
    mask = torch.zeros((1, h, w), dtype=torch.float32)

    mask_area_ratio = random.uniform(min_mask_ratio, max_mask_ratio)
    mask_area = h * w * mask_area_ratio
    aspect_ratio = random.uniform(0.5, 2.0)
    mask_h = int(np.sqrt(mask_area * aspect_ratio))
    mask_w = int(np.sqrt(mask_area / aspect_ratio))


    mask_h = min(max(1, mask_h), h -1)
    mask_w = min(max(1, mask_w), w -1)

    if mask_h == 0 or mask_w == 0: 
        return generate_random_mask(img_tensor, min_mask_ratio, max_mask_ratio) # retry

    top_y = random.randint(0, h - mask_h)
    left_x = random.randint(0, w - mask_w)

    mask[:, top_y : top_y + mask_h, left_x : left_x + mask_w] = 1.0
    return mask

# --- 3. Custom Oxford-IIIT Pet Inpainting Dataset ---
class OxfordPetInpaintingDataset(Dataset):
    def __init__(self, root_dir, split='trainval', transform=None, download=True):
        self.transform = transform
        try:
            self.pet_dataset = torchvision.datasets.OxfordIIITPet(
                root=root_dir,
                split=split,
                target_types='category', 
                transform=self.transform, 
                download=download
            )
        except Exception as e:
            print(f"Error initializing OxfordIIITPet dataset: {e}")
            print(f"Please ensure the dataset is correctly downloaded or consider manually downloading it to the '{root_dir}/oxford-iiit-pet/' directory.")
            raise

    def __len__(self):
        return len(self.pet_dataset)

    def __getitem__(self, idx):
        original_img_tensor, _ = self.pet_dataset[idx] 
        mask = generate_random_mask(original_img_tensor)
        masked_img_tensor = original_img_tensor * (1.0 - mask)

        return masked_img_tensor, original_img_tensor, mask

# --- 4. Example Usage: DataLoader ---
if __name__ == '__main__':
    print("Attempting to download and prepare Oxford-IIIT Pet dataset...")
    try:
        dataset_split = 'trainval'

        train_dataset = OxfordPetInpaintingDataset(
            root_dir=DATA_DIR,
            split=dataset_split,
            transform=transform,
            download=True
        )
        print(f"Successfully loaded Oxford-IIIT Pet {dataset_split} split with {len(train_dataset)} samples.")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

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