# UNet/train.py (Modified for U-Net Depth Ablation Study)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from multiprocessing import freeze_support #
from tqdm import tqdm

# Assuming your Pet dataset loader script is named 'data_loader.py'
from data_loader import OxfordPetInpaintingDataset, transform #

# IMPORTANT ASSUMPTION FOR THIS EXAMPLE:
# You have created different U-Net model files or a parameterized U-Net class.
# For simplicity, we'll assume you can import them like this:
# from UNet_depth3 import UNet as UNetDepth3 # Example: A U-Net with 3 down/up layers
from UNet import UNet as UNetDepth4       # Your current U-Net
from UNet import UNet3 as UNetDepth3 # Example: A U-Net with 3 down/up layers
from UNet import UNet5 as UNetDepth5 # Example: A U-Net with 5 down/up layers
# --- Static Configuration (can be overridden by loop if needed) ---
IMG_SIZE_CONFIG = 128 #
BATCH_SIZE_CONFIG = 32 #
DATA_DIR_CONFIG = '../data_pets' #
NUM_EPOCHS_CONFIG = 25 #
LEARNING_RATE_CONFIG = 1e-4 #
DEVICE_CONFIG = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
BASE_MODEL_SAVE_DIR = './models_unet_depth_ablation' # Base directory for this study
BASE_VAL_SAMPLES_DIR = './val_samples_unet_depth_ablation'


# --- Main Ablation Loop ---
if __name__ == '__main__':
    freeze_support() #

    # Define the U-Net configurations to test
    # Each entry could be a tuple: (depth_name, ModelClass)
    # For this example, ensure UNetDepth3 and UNetDepth5 are defined and importable if you use them.
    unet_configurations = [
        ("Depth3", UNetDepth3), # Uncomment if you have UNetDepth3
        ("Depth4", UNetDepth4),   # Your current U-Net
        ("Depth5", UNetDepth5), # Uncomment if you have UNetDepth5
    ]
    # If you only want to run your current UNet as part of this structure, 
    # keep only ("Depth4", UNetDepth4) or add more when ready.

    all_experiment_results_unet = []

    # --- DataLoaders (can be defined once outside the loop) ---
    print(f"Loading datasets from {DATA_DIR_CONFIG}...")
    train_pet_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR_CONFIG, split='trainval', transform=transform, download=True
    ) #
    train_dataloader = DataLoader(
        train_pet_dataset, batch_size=BATCH_SIZE_CONFIG, shuffle=True, num_workers=0
    ) #

    val_pet_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR_CONFIG, split='test', transform=transform, download=True
    ) #
    val_dataloader = DataLoader(
        val_pet_dataset, batch_size=BATCH_SIZE_CONFIG, shuffle=False, num_workers=0
    ) #

    for depth_name, ModelClass in unet_configurations:
        print(f"\n\n===== Starting U-Net Training for {depth_name} =====")
        print(f"Using device: {DEVICE_CONFIG}")

        # --- Per-Experiment Setup ---
        CURRENT_MODEL_SAVE_DIR = os.path.join(BASE_MODEL_SAVE_DIR, depth_name)
        CURRENT_VAL_SAMPLES_DIR = os.path.join(BASE_VAL_SAMPLES_DIR, depth_name)
        MODEL_SAVE_PATH_FINAL = os.path.join(CURRENT_MODEL_SAVE_DIR, f'unet_inpainting_final_{depth_name}.pth')

        if not os.path.exists(CURRENT_MODEL_SAVE_DIR):
            os.makedirs(CURRENT_MODEL_SAVE_DIR)
        if not os.path.exists(CURRENT_VAL_SAMPLES_DIR):
            os.makedirs(CURRENT_VAL_SAMPLES_DIR)

        # Initialize Model for the current depth
        model = ModelClass(n_channels_in=3, n_channels_out=3).to(DEVICE_CONFIG)
        criterion = nn.L1Loss() #
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_CONFIG) #

        # --- Training Loop ---
        for epoch in range(NUM_EPOCHS_CONFIG):
            model.train()
            running_loss = 0.0
            train_progress_bar = tqdm(train_dataloader, desc=f"{depth_name} - Epoch {epoch+1}/{NUM_EPOCHS_CONFIG} [Train]", leave=False)
            for i, (masked_images, original_images, masks) in enumerate(train_progress_bar): #
                masked_images = masked_images.to(DEVICE_CONFIG) #
                original_images = original_images.to(DEVICE_CONFIG) #

                optimizer.zero_grad() #
                outputs = model(masked_images) #
                loss = criterion(outputs, original_images) #
                loss.backward() #
                optimizer.step() #

                running_loss += loss.item() #
                train_progress_bar.set_postfix(loss=f"{loss.item():.4f}") #

            avg_train_loss = running_loss / len(train_dataloader)
            tqdm.write(f"{depth_name} - Epoch [{epoch+1}/{NUM_EPOCHS_CONFIG}] completed. Avg Training Loss: {avg_train_loss:.4f}")

            # Validation step
            model.eval() #
            val_loss = 0.0
            first_val_batch_saved_this_epoch = False
            val_progress_bar = tqdm(val_dataloader, desc=f"{depth_name} - Epoch {epoch+1}/{NUM_EPOCHS_CONFIG} [Val]", leave=False)
            with torch.no_grad(): #
                for masked_images_val, original_images_val, _ in val_progress_bar: #
                    masked_images_val = masked_images_val.to(DEVICE_CONFIG) #
                    original_images_val = original_images_val.to(DEVICE_CONFIG) #

                    outputs_val = model(masked_images_val) #
                    loss_val_batch = criterion(outputs_val, original_images_val) #
                    val_loss += loss_val_batch.item() #
                    val_progress_bar.set_postfix(loss=f"{loss_val_batch.item():.4f}") #

                    if not first_val_batch_saved_this_epoch and (epoch + 1) % 5 == 0 : #
                        # Denormalize for saving
                        denorm_transform = transforms.Compose([
                            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                        ]) #
                        for k in range(min(4, masked_images_val.size(0))): # Save up to 4 images
                            # Save to current experiment's val_samples directory
                            torchvision.utils.save_image(denorm_transform(outputs_val[k].cpu()), os.path.join(CURRENT_VAL_SAMPLES_DIR,f'epoch_{epoch+1}_output_{k}.png'))
                            torchvision.utils.save_image(denorm_transform(masked_images_val[k].cpu()), os.path.join(CURRENT_VAL_SAMPLES_DIR,f'epoch_{epoch+1}_masked_{k}.png'))
                            torchvision.utils.save_image(denorm_transform(original_images_val[k].cpu()), os.path.join(CURRENT_VAL_SAMPLES_DIR,f'epoch_{epoch+1}_original_{k}.png'))
                        first_val_batch_saved_this_epoch = True #
            
            avg_val_loss = val_loss / len(val_dataloader)
            tqdm.write(f"{depth_name} - Epoch [{epoch+1}/{NUM_EPOCHS_CONFIG}], Validation Loss: {avg_val_loss:.4f}")

            if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS_CONFIG: #
                # Save to current experiment's model directory
                torch.save(model.state_dict(), os.path.join(CURRENT_MODEL_SAVE_DIR, f'unet_inpainting_epoch_{epoch+1}.pth'))
                tqdm.write(f"Model for {depth_name} saved at epoch {epoch+1} to {CURRENT_MODEL_SAVE_DIR}")

        print(f"Training finished for {depth_name}.")
        torch.save(model.state_dict(), MODEL_SAVE_PATH_FINAL) #
        print(f"Final model for {depth_name} saved to {MODEL_SAVE_PATH_FINAL}")

        # Store results for this experiment
        all_experiment_results_unet.append({
            'depth_name': depth_name,
            'final_avg_train_loss': avg_train_loss, # Last epoch's average
            'final_avg_val_loss': avg_val_loss,     # Last epoch's average
            'model_path': MODEL_SAVE_PATH_FINAL,
            'val_samples_dir': CURRENT_VAL_SAMPLES_DIR
        })
        # IMPORTANT: You should run your test.py logic here (or after all loops)
        # for each MODEL_SAVE_PATH_FINAL to get PSNR, SSIM on the actual test set
        # and add those to all_experiment_results_unet.

    print("\n\n===== All U-Net Depth Ablation Experiments Finished =====")
    print("Summary of U-Net depth experiments:")
    for res in all_experiment_results_unet:
        print(f"  Depth: {res['depth_name']}, Final Train Loss: {res['final_avg_train_loss']:.4f}, Final Val Loss: {res['final_avg_val_loss']:.4f}, Model: {res['model_path']}")

    import json
    summary_file_path = os.path.join(BASE_MODEL_SAVE_DIR, 'unet_depth_ablation_summary.json')
    with open(summary_file_path, 'w') as f:
        json.dump(all_experiment_results_unet, f, indent=4)
    print(f"U-Net depth ablation summary saved to {summary_file_path}")