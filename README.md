# Image Inpainting with U-Net and GANs

**Project for Artificial Intelligence II: Deep Learning Methods (BIOSINF, ETTI, UPB)**
**Year: 2024-2025**

This project explores image inpainting techniques using two main deep learning approaches: a standalone U-Net architecture and a U-Net based Generative Adversarial Network (GAN). The goal is to reconstruct missing regions in images from the Oxford-IIIT Pet dataset.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [Dataset](#dataset)
4.  [Models Implemented](#models-implemented)
    * [Standalone U-Net](#standalone-u-net)
    * [Generative Adversarial Network (GAN)](#generative-adversarial-network-gan)
5.  [Directory Structure](#directory-structure)
6.  [Setup and Installation](#setup-and-installation)
7.  [Usage](#usage)
    * [Data Preparation](#data-preparation)
    * [Training](#training)
    * [Testing](#testing)
    * [Ablation Studies](#ablation-studies)
8.  [Ablation Studies Conducted](#ablation-studies-conducted)
9.  [Results](#results)
10. [Future Work](#future-work)
11. [Dependencies](#dependencies)

## Project Overview

Image inpainting is the task of filling in missing or corrupted parts of an image. This project implements and evaluates two deep learning models for this task:
* A U-Net model trained directly with L1 reconstruction loss.
* A Conditional GAN (cGAN) where the generator is a U-Net and the discriminator is a PatchGAN, trained with a combination of adversarial loss and L1 reconstruction loss.

The project aims to compare these methods and explore the impact of various architectural choices and training parameters through ablation studies.

## Features
* Implementation of a U-Net architecture for image inpainting.
* Implementation of a GAN with a U-Net generator and PatchGAN discriminator.
* Custom data loader for the Oxford-IIIT Pet dataset with on-the-fly random mask generation.
* Training scripts for both U-Net and GAN models.
* Testing scripts to evaluate models using L1 loss, PSNR, and SSIM metrics.
* Functionality to save and compare visual results (original, masked, inpainted, mask).
* Scripts structured to facilitate ablation studies (e.g., by looping through parameters or manual configuration).

## Dataset
The project uses the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
* Images are resized to 128x128 pixels.
* Normalization is applied using ImageNet statistics (mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`).
* Random rectangular masks are generated on-the-fly during data loading.

The data loader scripts will automatically download the dataset if it's not found in the specified `DATA_DIR` (default: `../data_pets` relative to the script, or `./data_pets` for U-Net data loader if run from `UNet` directory).

## Models Implemented

### Standalone U-Net
* **Architecture**: A standard U-Net architecture with an encoder-decoder structure and skip connections. The depth and use of bilinear upsampling can be configured.
* **Loss Function**: Trained using L1 Loss (Mean Absolute Error) between the inpainted image and the original image.
* **Files**:
    * Model: `UNet/UNet.py`
    * Training: `UNet/train.py`
    * Testing: `UNet/test.py`
    * Data Loader: `UNet/data_loader.py`

### Generative Adversarial Network (GAN)
* **Generator (netG)**: A U-Net architecture, similar to the standalone U-Net.
* **Discriminator (netD)**: A PatchGAN discriminator which classifies NxN patches of an image as real or fake, rather than the entire image.
* **Loss Functions**:
    * Generator: Adversarial Loss (BCEWithLogitsLoss, aiming to fool the discriminator) + L1 Reconstruction Loss.
    * Discriminator: Adversarial Loss (BCEWithLogitsLoss, aiming to correctly classify real and fake images).
* **Files**:
    * Generator Model: `gan/UNet.py`
    * Discriminator Model: `gan/model.py`
    * Training: `gan/train.py`
    * Testing (Generator): `gan/test.py`
    * Data Loader: `gan/data_loader.py`

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd ImageInpaintingIA2
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    The primary dependency is PyTorch. You will also need `torchvision`, `numpy`, `scikit-image`, `matplotlib`, and `tqdm`.
    ```bash
    pip install torch torchvision torchaudio
    pip install numpy scikit-image matplotlib tqdm Pillow
    ```
    (Ensure you install a PyTorch version compatible with your CUDA version if using GPU.)

## Usage

### Data Preparation
The dataset will be downloaded automatically by the `data_loader.py` scripts if not present.
To prepare fixed samples for consistent visual testing across ablation studies (recommended):
1.  You might need to create a helper script (e.g., `prepare_fixed_samples.py` as discussed previously) to generate and save a few masked images, original images, and masks.
    ```python
    # Example (conceptual - place in a new script e.g. prepare_fixed_samples.py at root)
    # import torch
    # from UNet.data_loader import OxfordPetInpaintingDataset, transform # or gan.data_loader
    # import os
    # FIXED_SAMPLES_DIR = './fixed_test_samples'
    # DATA_DIR_CONFIG = './data_pets' # Adjust if your data_pets is elsewhere relative to this script
    # if not os.path.exists(FIXED_SAMPLES_DIR): os.makedirs(FIXED_SAMPLES_DIR)
    # dataset = OxfordPetInpaintingDataset(root_dir=DATA_DIR_CONFIG, split='test', transform=transform, download=True)
    # # ... (logic to select specific indices, generate masks consistently, and save tensors) ...
    # # torch.save(masked_tensor_batch, os.path.join(FIXED_SAMPLES_DIR, 'fixed_masked_images.pt'))
    # # torch.save(original_tensor_batch, os.path.join(FIXED_SAMPLES_DIR, 'fixed_original_images.pt'))
    # # torch.save(mask_tensor_batch, os.path.join(FIXED_SAMPLES_DIR, 'fixed_masks.pt'))
    ```
    Run this script once: `python prepare_fixed_samples.py`

### Training
Navigate to the respective model directory (`UNet/` or `gan/`).

**Standalone U-Net:**
```bash
cd UNet
python train.py
