# gan/train.py (Modified for Optimizer Ablation Study)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import json # For saving summary

# Assuming your model and data loader scripts are in the same 'gan' directory or accessible
from model import PatchGANDiscriminator #
from UNet import UNet # Generator
from data_loader import OxfordPetInpaintingDataset, transform #

# --- Static Configuration (can be overridden by loop if needed) ---
IMG_SIZE_CONFIG = 128
BATCH_SIZE_CONFIG = 8
DATA_DIR_CONFIG = '../data_pets'
NUM_EPOCHS_CONFIG = 50 # Start with this, GANs can take longer
# Default LRs, will be overridden by the loop configurations
LR_G_DEFAULT = 1e-4
LR_D_DEFAULT = 1e-4
BETA1_CONFIG = 0.5 # Adam optimizer beta1
LAMBDA_L1_CONFIG = 100.0 # Weight for L1 reconstruction loss
DEVICE_CONFIG = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_OUTPUT_DIR_OPTIM = './gan_optimizer_ablation_results'

# --- Utility to save sample images --- (Copied from your original script)
def denormalize_imagenet(tensor): #
    denorm_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),])
    return torch.clamp(denorm_transform(tensor), 0, 1)

# --- Main Ablation Loop ---
if __name__ == '__main__':
    if DEVICE_CONFIG.type == 'cuda' and os.name == 'nt': #
        from multiprocessing import freeze_support
        freeze_support()

    # Define Optimizer configurations: (name, G_optim_class, D_optim_class, G_lr, D_lr, G_params, D_params)
    optimizer_configurations = [
        ("Adam_Adam", torch.optim.Adam, torch.optim.Adam, LR_G_DEFAULT, LR_D_DEFAULT, {'betas':(BETA1_CONFIG, 0.999)}, {'betas':(BETA1_CONFIG, 0.999)}),
        ("SGD_SGD", torch.optim.SGD, torch.optim.SGD, 1e-3, 1e-3, {'momentum':0.9}, {'momentum':0.9}), # SGD might need different LRs
        ("RMSprop_RMSprop", torch.optim.RMSprop, torch.optim.RMSprop, LR_G_DEFAULT, LR_D_DEFAULT, {}, {}),
    ]
    # Note: Learning rates (G_lr, D_lr) might need careful tuning for each optimizer pair.

    all_gan_experiment_results = []

    # DataLoaders (can be defined once if they don't change per experiment)
    print(f"Loading dataset from {DATA_DIR_CONFIG}...")
    train_dataset = OxfordPetInpaintingDataset(
        root_dir=DATA_DIR_CONFIG, split='trainval', transform=transform, download=True
    ) #
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_CONFIG, shuffle=True, num_workers=0
    ) #

    for optim_name, G_OptimClass, D_OptimClass, current_g_lr, current_d_lr, g_optim_params, d_optim_params in optimizer_configurations:
        print(f"\n\n===== Starting GAN Training with Optimizers: {optim_name} (G_LR:{current_g_lr}, D_LR:{current_d_lr}) =====")

        # --- Per-Experiment Setup ---
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR_OPTIM, optim_name)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # --- Initialize Generator (G) and Discriminator (D) ---
        netG = UNet(n_channels_in=3, n_channels_out=3).to(DEVICE_CONFIG)
        netD = PatchGANDiscriminator(input_channels=3).to(DEVICE_CONFIG) #

        # --- Loss Functions ---
        criterion_GAN = nn.BCEWithLogitsLoss() #
        criterion_L1 = nn.L1Loss() #

        # --- Optimizers (Initialize for current configuration) ---
        optimizerG = G_OptimClass(netG.parameters(), lr=current_g_lr, **g_optim_params)
        optimizerD = D_OptimClass(netD.parameters(), lr=current_d_lr, **d_optim_params)
        print(f"Using Generator Optimizer: {optimizerG}")
        print(f"Using Discriminator Optimizer: {optimizerD}")

        # --- Training Loop (inner loop for epochs) ---
        print(f"Starting GAN training on {DEVICE_CONFIG} for optimizers: {optim_name}...")
        avg_loss_g_final_epoch = 0
        avg_loss_d_final_epoch = 0

        for epoch in range(NUM_EPOCHS_CONFIG):
            netG.train()
            netD.train()
            total_loss_g_epoch = 0
            total_loss_d_epoch = 0

            progress_bar = tqdm(train_dataloader, desc=f"{optim_name} - Epoch {epoch+1}/{NUM_EPOCHS_CONFIG}")
            for i, (masked_images, original_images, masks) in enumerate(progress_bar): #
                masked_images = masked_images.to(DEVICE_CONFIG)
                original_images = original_images.to(DEVICE_CONFIG)
                
                # --- Train Discriminator (D) --- (Copied from your original script)
                optimizerD.zero_grad()
                real_output_D = netD(original_images)
                target_real = torch.ones_like(real_output_D, device=DEVICE_CONFIG)
                loss_D_real = criterion_GAN(real_output_D, target_real)
                fake_images = netG(masked_images).detach()
                fake_output_D = netD(fake_images)
                target_fake = torch.zeros_like(fake_output_D, device=DEVICE_CONFIG)
                loss_D_fake = criterion_GAN(fake_output_D, target_fake)
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizerD.step()
                total_loss_d_epoch += loss_D.item()

                # --- Train Generator (G) --- (Copied from your original script)
                optimizerG.zero_grad()
                generated_images = netG(masked_images)
                output_G_fake_for_D = netD(generated_images)
                loss_G_GAN = criterion_GAN(output_G_fake_for_D, target_real)
                loss_G_L1 = criterion_L1(generated_images, original_images) * LAMBDA_L1_CONFIG
                loss_G = loss_G_GAN + loss_G_L1
                loss_G.backward()
                optimizerG.step()
                total_loss_g_epoch += loss_G.item()

                progress_bar.set_postfix({
                    'Loss_D': f"{loss_D.item():.4f}", 'Loss_G': f"{loss_G.item():.4f}",
                    'Loss_G_GAN': f"{loss_G_GAN.item():.4f}", 'Loss_G_L1': f"{loss_G_L1.item()/LAMBDA_L1_CONFIG:.4f}"
                }) #

            avg_loss_g_final_epoch = total_loss_g_epoch / len(train_dataloader)
            avg_loss_d_final_epoch = total_loss_d_epoch / len(train_dataloader)
            tqdm.write(f"{optim_name} - Epoch [{epoch+1}/{NUM_EPOCHS_CONFIG}] Avg Loss_G: {avg_loss_g_final_epoch:.4f}, Avg Loss_D: {avg_loss_d_final_epoch:.4f}")

            # Save some sample images periodically (Copied from your original script)
            if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS_CONFIG - 1: #
                with torch.no_grad():
                    netG.eval()
                    # Using first few images from the last batch for samples
                    sample_masked = masked_images[0:min(4, BATCH_SIZE_CONFIG)].cpu()
                    sample_generated = netG(masked_images[0:min(4, BATCH_SIZE_CONFIG)]).cpu()
                    sample_original = original_images[0:min(4, BATCH_SIZE_CONFIG)].cpu()
                    
                    img_masked_denorm = denormalize_imagenet(sample_masked)
                    img_generated_denorm = denormalize_imagenet(sample_generated)
                    img_original_denorm = denormalize_imagenet(sample_original)

                    comparison_strip = []
                    for k in range(img_masked_denorm.size(0)):
                        comparison_strip.extend([img_masked_denorm[k], img_generated_denorm[k], img_original_denorm[k]])
                    
                    grid = torchvision.utils.make_grid(comparison_strip, nrow=3, padding=5, normalize=False)
                    torchvision.utils.save_image(grid, os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}_sample.png"))
                    tqdm.write(f"Saved sample images for epoch {epoch+1} in {OUTPUT_DIR}")
                    netG.train()

        print(f"GAN Training Finished for optimizers: {optim_name}.")
        final_g_model_path = os.path.join(OUTPUT_DIR, 'netG_final.pth')
        final_d_model_path = os.path.join(OUTPUT_DIR, 'netD_final.pth')
        torch.save(netG.state_dict(), final_g_model_path) #
        torch.save(netD.state_dict(), final_d_model_path) #
        print(f"Final models for {optim_name} saved to {OUTPUT_DIR}")

        all_gan_experiment_results.append({
            'optimizer_name': optim_name,
            'generator_lr': current_g_lr,
            'discriminator_lr': current_d_lr,
            'final_avg_loss_g': avg_loss_g_final_epoch,
            'final_avg_loss_d': avg_loss_d_final_epoch,
            'generator_model_path': final_g_model_path,
            'output_dir': OUTPUT_DIR
        })
        # IMPORTANT: Evaluate netG on the test set here using logic from gan/test.py
        # to get PSNR, SSIM and add them to all_gan_experiment_results.

    print("\n\n===== All GAN Optimizer Ablation Experiments Finished =====")
    print("Summary of GAN optimizer experiments:")
    for res in all_gan_experiment_results:
        print(f"  Optimizers: {res['optimizer_name']}, G_LR: {res['generator_lr']}, D_LR: {res['discriminator_lr']}, "
              f"Final G_Loss: {res['final_avg_loss_g']:.4f}, Final D_Loss: {res['final_avg_loss_d']:.4f}, Output: {res['output_dir']}")
    
    summary_file_path = os.path.join(BASE_OUTPUT_DIR_OPTIM, 'gan_optimizer_ablation_summary.json')
    with open(summary_file_path, 'w') as f:
        json.dump(all_gan_experiment_results, f, indent=4)
    print(f"GAN optimizer ablation summary saved to {summary_file_path}")