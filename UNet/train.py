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
from UNet import UNet
# --- Configuration ---
IMG_SIZE = 128
BATCH_SIZE = 32
DATA_DIR = '../data_pets'
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'models/unet_inpainting_model.pth'

# --- 1. U-Net Model Definition ---
# (Your UNetConvBlock, Down, Up, OutConv, UNet class definitions remain the same)


# --- Main execution block ---
if __name__ == '__main__':
    freeze_support() # Call this first if num_workers > 0 (though current code uses 0)

    print(f"Starting training on {DEVICE}...")

    # --- 2. Training Setup ---
    train_pet_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR,
        split='trainval',
        transform=transform,
        download=True
    )
    train_dataloader = DataLoader(
        train_pet_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 # Kept as 0 based on your provided code
    )

    val_pet_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR,
        split='test',
        transform=transform,
        download=True
    )
    val_dataloader = DataLoader(
        val_pet_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0 # Kept as 0
    )

    model = UNet(n_channels_in=3, n_channels_out=3).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        # Wrap train_dataloader with tqdm
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for i, (masked_images, original_images, masks) in enumerate(train_progress_bar):
            masked_images = masked_images.to(DEVICE)
            original_images = original_images.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(masked_images)
            loss = criterion(outputs, original_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Update tqdm postfix with current batch loss
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Optional: Print less frequently or use tqdm.write if prints clutter the bar
            # if (i + 1) % 100 == 0:
            #     tqdm.write(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_dataloader)}], Train Loss: {loss.item():.4f}")


        avg_train_loss = running_loss / len(train_dataloader)
        # Use tqdm.write for messages that should persist outside the progress bar loop
        tqdm.write(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Average Training Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        first_val_batch_saved_this_epoch = False # To save only first batch images per designated epoch
        # Wrap val_dataloader with tqdm
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for masked_images_val, original_images_val, _ in val_progress_bar:
                masked_images_val = masked_images_val.to(DEVICE)
                original_images_val = original_images_val.to(DEVICE)

                outputs_val = model(masked_images_val)
                loss_val_batch = criterion(outputs_val, original_images_val)
                val_loss += loss_val_batch.item()
                val_progress_bar.set_postfix(loss=f"{loss_val_batch.item():.4f}")


                # Optionally save some validation output images
                if not first_val_batch_saved_this_epoch and (epoch + 1) % 5 == 0 : # Save first batch of designated epochs
                    if not os.path.exists('./val_samples'):
                        os.makedirs('./val_samples')
                    denorm_transform = transforms.Compose([
                        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                    ])
                    for k in range(min(4, masked_images_val.size(0))): # Save up to 4 images
                        torchvision.utils.save_image(denorm_transform(outputs_val[k].cpu()), f'./val_samples/epoch_{epoch+1}_output_{k}.png')
                        torchvision.utils.save_image(denorm_transform(masked_images_val[k].cpu()), f'./val_samples/epoch_{epoch+1}_masked_{k}.png')
                        torchvision.utils.save_image(denorm_transform(original_images_val[k].cpu()), f'./val_samples/epoch_{epoch+1}_original_{k}.png')
                    first_val_batch_saved_this_epoch = True


        avg_val_loss = val_loss / len(val_dataloader)
        tqdm.write(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")

        # Save the model checkpoint periodically
        if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
            torch.save(model.state_dict(), f'./unet_inpainting_epoch_{epoch+1}.pth')
            tqdm.write(f"Model saved at epoch {epoch+1}")

    print("Training finished.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")