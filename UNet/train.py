import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from multiprocessing import freeze_support #
from tqdm import tqdm

from data_loader import OxfordPetInpaintingDataset 
from UNet import UNet as UNetDepth4      
from UNet import UNet3 as UNetDepth3 
from UNet import UNet5 as UNetDepth5 
IMG_SIZE_CONFIG = 128 
BATCH_SIZE_CONFIG = 32 
DATA_DIR_CONFIG = '../data_pets' 
NUM_EPOCHS_CONFIG = 25 
LEARNING_RATE_CONFIG = 1e-4 
DEVICE_CONFIG = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
BASE_MODEL_SAVE_DIR = './models_unet_depth_ablation' 
BASE_VAL_SAMPLES_DIR = './val_samples_unet_depth_ablation'


if __name__ == '__main__':
    freeze_support() #
    unet_configurations = [
        ("Depth3", UNetDepth3),
        ("Depth4", UNetDepth4),   
        ("Depth5", UNetDepth5), 
    ]

    all_experiment_results_unet = []

    print(f"Loading datasets from {DATA_DIR_CONFIG}...")
    train_pet_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR_CONFIG, split='trainval', transform=transform, download=True
    ) 
    train_dataloader = DataLoader(
        train_pet_dataset, batch_size=BATCH_SIZE_CONFIG, shuffle=True, num_workers=0
    ) 

    val_pet_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR_CONFIG, split='test', transform=transform, download=True
    ) 
    val_dataloader = DataLoader(
        val_pet_dataset, batch_size=BATCH_SIZE_CONFIG, shuffle=False, num_workers=0
    ) 

    for depth_name, ModelClass in unet_configurations:
        print(f"\n\n===== Starting U-Net Training for {depth_name} =====")
        print(f"Using device: {DEVICE_CONFIG}")

        CURRENT_MODEL_SAVE_DIR = os.path.join(BASE_MODEL_SAVE_DIR, depth_name)
        CURRENT_VAL_SAMPLES_DIR = os.path.join(BASE_VAL_SAMPLES_DIR, depth_name)
        MODEL_SAVE_PATH_FINAL = os.path.join(CURRENT_MODEL_SAVE_DIR, f'unet_inpainting_final_{depth_name}.pth')

        if not os.path.exists(CURRENT_MODEL_SAVE_DIR):
            os.makedirs(CURRENT_MODEL_SAVE_DIR)
        if not os.path.exists(CURRENT_VAL_SAMPLES_DIR):
            os.makedirs(CURRENT_VAL_SAMPLES_DIR)

        model = ModelClass(n_channels_in=3, n_channels_out=3).to(DEVICE_CONFIG)
        criterion = nn.L1Loss() #
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_CONFIG) #

        for epoch in range(NUM_EPOCHS_CONFIG):
            model.train()
            running_loss = 0.0
            train_progress_bar = tqdm(train_dataloader, desc=f"{depth_name} - Epoch {epoch+1}/{NUM_EPOCHS_CONFIG} [Train]", leave=False)
            for i, (masked_images, original_images, masks) in enumerate(train_progress_bar): 
                masked_images = masked_images.to(DEVICE_CONFIG) 
                original_images = original_images.to(DEVICE_CONFIG) 

                optimizer.zero_grad() 
                outputs = model(masked_images) 
                loss = criterion(outputs, original_images) 
                loss.backward() 
                optimizer.step() 

                running_loss += loss.item() 
                train_progress_bar.set_postfix(loss=f"{loss.item():.4f}") 

            avg_train_loss = running_loss / len(train_dataloader)
            tqdm.write(f"{depth_name} - Epoch [{epoch+1}/{NUM_EPOCHS_CONFIG}] completed. Avg Training Loss: {avg_train_loss:.4f}")

            model.eval() 
            val_loss = 0.0
            first_val_batch_saved_this_epoch = False
            val_progress_bar = tqdm(val_dataloader, desc=f"{depth_name} - Epoch {epoch+1}/{NUM_EPOCHS_CONFIG} [Val]", leave=False)
            with torch.no_grad(): 
                for masked_images_val, original_images_val, _ in val_progress_bar: 
                    masked_images_val = masked_images_val.to(DEVICE_CONFIG) 
                    original_images_val = original_images_val.to(DEVICE_CONFIG) 

                    outputs_val = model(masked_images_val) 
                    loss_val_batch = criterion(outputs_val, original_images_val) 
                    val_loss += loss_val_batch.item() 
                    val_progress_bar.set_postfix(loss=f"{loss_val_batch.item():.4f}") 

                    if not first_val_batch_saved_this_epoch and (epoch + 1) % 5 == 0 : 
                        denorm_transform = transforms.Compose([
                            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                        ]) #
                        for k in range(min(4, masked_images_val.size(0))): 
                            torchvision.utils.save_image(denorm_transform(outputs_val[k].cpu()), os.path.join(CURRENT_VAL_SAMPLES_DIR,f'epoch_{epoch+1}_output_{k}.png'))
                            torchvision.utils.save_image(denorm_transform(masked_images_val[k].cpu()), os.path.join(CURRENT_VAL_SAMPLES_DIR,f'epoch_{epoch+1}_masked_{k}.png'))
                            torchvision.utils.save_image(denorm_transform(original_images_val[k].cpu()), os.path.join(CURRENT_VAL_SAMPLES_DIR,f'epoch_{epoch+1}_original_{k}.png'))
                        first_val_batch_saved_this_epoch = True 
            
            avg_val_loss = val_loss / len(val_dataloader)
            tqdm.write(f"{depth_name} - Epoch [{epoch+1}/{NUM_EPOCHS_CONFIG}], Validation Loss: {avg_val_loss:.4f}")

            if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS_CONFIG: 
                torch.save(model.state_dict(), os.path.join(CURRENT_MODEL_SAVE_DIR, f'unet_inpainting_epoch_{epoch+1}.pth'))
                tqdm.write(f"Model for {depth_name} saved at epoch {epoch+1} to {CURRENT_MODEL_SAVE_DIR}")

        print(f"Training finished for {depth_name}.")
        torch.save(model.state_dict(), MODEL_SAVE_PATH_FINAL) 
        print(f"Final model for {depth_name} saved to {MODEL_SAVE_PATH_FINAL}")

        all_experiment_results_unet.append({
            'depth_name': depth_name,
            'final_avg_train_loss': avg_train_loss, 
            'final_avg_val_loss': avg_val_loss,     
            'model_path': MODEL_SAVE_PATH_FINAL,
            'val_samples_dir': CURRENT_VAL_SAMPLES_DIR
        })

    print("\n\n===== All U-Net Depth Ablation Experiments Finished =====")
    print("Summary of U-Net depth experiments:")
    for res in all_experiment_results_unet:
        print(f"  Depth: {res['depth_name']}, Final Train Loss: {res['final_avg_train_loss']:.4f}, Final Val Loss: {res['final_avg_val_loss']:.4f}, Model: {res['model_path']}")

    import json
    summary_file_path = os.path.join(BASE_MODEL_SAVE_DIR, 'unet_depth_ablation_summary.json')
    with open(summary_file_path, 'w') as f:
        json.dump(all_experiment_results_unet, f, indent=4)
    print(f"U-Net depth ablation summary saved to {summary_file_path}")