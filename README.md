# Image Inpainting with U-Net, GANs, and OpenCV Comparison

This project explores image inpainting techniques using deep learning models (U-Net and Generative Adversarial Networks - GANs) and compares their performance with classical computer vision algorithms available in OpenCV. The primary dataset used is the Oxford-IIIT Pet Dataset.

## Table of Contents
1.  [Overview](#overview)
2.  [Features](#features)
3.  [Setup and Installation](#setup-and-installation)
4.  [Dataset](#dataset)
5.  [Usage](#usage)
    * [U-Net Models](#u-net-models)
    * [GAN Models](#gan-models)
    * [OpenCV Inpainting Comparison](#opencv-inpainting-comparison)
6.  [Models Architecture](#models-architecture)
    * [U-Net](#u-net)
    * [GAN](#gan)
7.  [Ablation Studies and Results](#ablation-studies-and-results)
8.  [Contributing](#contributing)
9.  [License](#license)

## Overview

The project implements and evaluates different approaches for image inpainting, the task of filling in missing or corrupted parts of an image. It includes:
* A U-Net based model with explorations into different network depths.
* A GAN-based model with a U-Net generator and a PatchGAN discriminator, including studies on L1 loss weighting and optimizer choices.
* A comparative analysis with traditional inpainting methods such as Telea and Navier-Stokes algorithms from OpenCV, as well as a simple mean color fill baseline.

The models are trained and tested on the Oxford-IIIT Pet Dataset. Performance is evaluated using L1 loss, Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).

## Features

* **U-Net Implementation**: Customizable U-Net architecture with varying depths (Depth3, Depth4, Depth5).
* **GAN Implementation**: Conditional GAN with a U-Net generator and a PatchGAN discriminator.
* **Data Loading**: Custom data loader for the Oxford-IIIT Pet Dataset, including image transformations and random mask generation.
* **Training Scripts**:
    * U-Net training with depth ablation capabilities.
    * GAN training with L1 lambda and optimizer ablation capabilities.
* **Testing Scripts**:
    * Evaluation scripts for trained U-Net and GAN models, calculating L1, PSNR, and SSIM.
    * Generation of visual comparisons for qualitative assessment.
* **OpenCV Comparison**: Script to evaluate and compare results with OpenCV's inpainting methods (Telea, Navier-Stokes) and a mean fill algorithm.
* **Ablation Studies**:
    * Effect of U-Net depth on performance.
    * Effect of L1 loss weight (lambda) in GAN training.
    * Effect of different optimizers in GAN training.
* **Results Tracking**: JSON files summarizing training and testing metrics for different experiments.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/diaconuadr13/ImageInpaintingIA2
    cd https://github.com/diaconuadr13/ImageInpaintingIA2
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```
3.  **Install dependencies:**
    The project requires Python and libraries such as PyTorch, TorchVision, OpenCV, scikit-image, NumPy, and Matplotlib.
    ```bash
    pip install torch torchvision torchaudio
    pip install opencv-python scikit-image numpy matplotlib tqdm
    ```
    Ensure you have a version of PyTorch compatible with your CUDA version if you plan to use a GPU.

## Dataset

The project uses the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
The `data_loader.py` scripts in both the `UNet/` and `gan/` directories are configured to automatically download the dataset if it's not found in the specified `DATA_DIR` (default is `./data_pets` relative to the script, so likely `../data_pets` if running from `UNet/` or `gan/` subdirectories, or `./data_pets` if `DATA_DIR` is adjusted in scripts).

The data loaders perform the following preprocessing steps:
* Resizing images to a fixed size (default 128x128).
* Converting images to PyTorch tensors.
* Normalizing images using ImageNet statistics.
* Generating random rectangular masks for inpainting.

## Usage

All training and testing scripts are configured with default parameters. You might need to adjust paths or parameters within the scripts if your setup differs.

### U-Net Models

**1. Training U-Net (with Depth Ablation):**
* Navigate to the `UNet/` directory.
* The `train.py` script is set up to train U-Net models of different depths (Depth3, Depth4, Depth5).
* Key configurations in `UNet/train.py`:
    * `IMG_SIZE_CONFIG`: Image size (default: 128)
    * `BATCH_SIZE_CONFIG`: Batch size (default: 32)
    * `DATA_DIR_CONFIG`: Dataset directory (default: `../data_pets`)
    * `NUM_EPOCHS_CONFIG`: Number of epochs (default: 25)
    * `LEARNING_RATE_CONFIG`: Learning rate (default: 1e-4)
    * `BASE_MODEL_SAVE_DIR`: Directory to save trained models and summary (default: `./models_unet_depth_ablation`)
* Run the training script:
    ```bash
    cd UNet
    python train.py
    ```
    Models and a `unet_depth_ablation_summary.json` file will be saved in `UNet/models_unet_depth_ablation/`.

**2. Testing U-Net Models:**
* The `UNet/test.py` script is used for evaluation.
* **Before running, you need to edit `UNet/test.py`** to specify the model to test:
    * `MODEL_PATH_TO_TEST`: Path to the trained U-Net model (e.g., `'./models_unet_depth_ablation/Depth4/unet_inpainting_final_Depth4.pth'`).
    * `MODEL_CLASS_NAME_TO_TEST`: The class name of the model (e.g., `"UNet"`, `"UNet3"`, `"UNet5"`). This must match the model type loaded.
    * `VARIANT_NAME_FOR_OUTPUTS`: A name for the output directory for this test run (e.g., `"Depth4_Default"`).
    * `RESULTS_BASE_DIR`: Base directory for test results (default: `'./test_results_unet_ablations_no_parser'`).
* Run the test script:
    ```bash
    cd UNet
    python test.py
    ```
    Metrics (L1, PSNR, SSIM) will be printed and saved in a JSON file within the specified results directory (e.g., `UNet/test_results_unet_ablations_no_parser/Depth4_Default/metrics.json`). Visual comparison images will also be saved.

### GAN Models

**1. Training GAN (with Optimizer Ablation):**
* Navigate to the `gan/` directory.
* The `train.py` script is configured for an optimizer ablation study (Adam, SGD, RMSprop).
    It can also be adapted for L1 lambda ablation or other GAN training by modifying configurations.
* Key configurations in `gan/train.py`:
    * `IMG_SIZE_CONFIG`: Image size (default: 128)
    * `BATCH_SIZE_CONFIG`: Batch size (default: 8)
    * `DATA_DIR_CONFIG`: Dataset directory (default: `../data_pets`)
    * `NUM_EPOCHS_CONFIG`: Number of epochs (default: 50)
    * `LR_G_DEFAULT`, `LR_D_DEFAULT`: Default learning rates for generator and discriminator.
    * `LAMBDA_L1_CONFIG`: Weight for L1 reconstruction loss (default: 100.0).
    * `BASE_OUTPUT_DIR_OPTIM`: Directory for optimizer ablation results (default: `'./gan_optimizer_ablation_results'`).
* Run the training script:
    ```bash
    cd gan
    python train.py
    ```
    Models (generator and discriminator) and a `gan_optimizer_ablation_summary.json` will be saved in `gan/gan_optimizer_ablation_results/`.
    *Note: The script structure also implies previous experiments for L1 lambda ablation, results of which are in `gan/gan_inpainting_results/`. You might need a different version of `train.py` or modify the current one to reproduce those specific lambda experiments.*

**2. Testing GAN Models (Generator):**
* The `gan/test.py` script is used for evaluating the trained GAN generator.
* **Before running, you need to edit `gan/test.py`** to specify the generator model to test:
    * `GENERATOR_MODEL_PATH_TO_TEST`: Path to the trained generator model (e.g., `'./gan_optimizer_ablation_results/Adam_Adam/netG_final.pth'` or `'./gan_inpainting_results/lambda_100.0/netG_final.pth'`).
    * `GENERATOR_CLASS_NAME_TO_TEST`: Class name of the generator, typically `"UNet"`.
    * `VARIANT_NAME_FOR_OUTPUTS`: A name for the output directory for this test run (e.g., `"GAN_Adam_Adam_DefaultLR_Best"`).
    * `RESULTS_BASE_DIR`: Base directory for GAN test results (default: `'./gan_test_results_ablations_no_parser'`).
* Run the test script:
    ```bash
    cd gan
    python test.py
    ```
    Metrics (L1, PSNR, SSIM) for the generator will be printed and saved in a JSON file within the specified results directory. Visual comparisons will also be saved.

### OpenCV Inpainting Comparison

* The `inpaint.py` script compares classical OpenCV inpainting methods (Telea, Navier-Stokes) and a Mean Color Fill baseline with the dataset.
* Key configurations in `inpaint.py`:
    * `DATA_DIR`: Dataset directory (default: `'./data_pets'`). Ensure this path is correct relative to the root project directory when running `inpaint.py`.
    * `RESULTS_SAVE_DIR`: Directory to save visual comparisons (default: `'./opencv_inpainting_results'`).
    * `INPAINT_RADIUS`: Radius for OpenCV inpainting algorithms.
* Run the script from the project's root directory:
    ```bash
    python inpaint.py
    ```
    Average metrics for each OpenCV method will be printed, and visual samples saved to `opencv_inpainting_results/`.

## Models Architecture

### U-Net

The U-Net architecture is implemented in `UNet/UNet.py`. It consists of:
* `UNetConvBlock`: A block of two convolutional layers, each followed by Batch Normalization and ReLU activation.
* `Down`: A downsampling block using MaxPool2d followed by a `UNetConvBlock`.
* `Up`: An upsampling block that uses either bilinear upsampling or a transposed convolution, followed by concatenation with skip connections from the encoder path, and a `UNetConvBlock`.
* `OutConv`: A final 1x1 convolution to produce the output image.
* The main U-Net classes are `UNet` (typically 4 down/up stages), `UNet3` (3 down/up stages), and `UNet5` (5 down/up stages).

### GAN

The GAN model consists of a generator and a discriminator:
* **Generator**: A U-Net model, similar to the one described above. The implementation is in `gan/UNet.py`. It takes a masked image as input and outputs the inpainted image.
* **Discriminator**: A PatchGAN discriminator, implemented in `gan/model.py`. It takes an image as input and outputs a map of predictions, where each value classifies a patch of the input image as real or fake.

The GAN is trained with a combination of adversarial loss (BCEWithLogitsLoss) and L1 reconstruction loss.

## Ablation Studies and Results

The project includes several ablation studies to evaluate the impact of different architectural choices and training parameters.

* **U-Net Depth Ablation**:
    * Training summary: `UNet/models_unet_depth_ablation/unet_depth_ablation_summary.json`
    * Test metrics: Stored in subdirectories under `UNet/test_results_unet_ablations_no_parser/` (e.g., `Depth3_Default/metrics3.json`, `Depth4_Default/metrics4.json`, `Depth5_Default/metrics5.json`). *Note: `metrics.json` also exists for Depth4.*

* **GAN L1 Lambda Ablation**:
    * Training summary: `gan/gan_inpainting_results/ablation_summary.json` (shows training losses for lambda values 10, 50, 100, 150).
    * Test metrics: Stored in subdirectories under `gan/gan_test_results_ablations_no_parser/` (e.g., `GAN_lambda10_DefaultLR_Best/lambda10.json`, `GAN_lambda50_DefaultLR_Best/lambda50.json`, `GAN_lambda100_DefaultLR_Best/lambda100.json`, `GAN_lambda150_DefaultLR_Best/lambda150.json`).

* **GAN Optimizer Ablation**:
    * Training summary: `gan/gan_optimizer_ablation_results/gan_optimizer_ablation_summary.json` (Adam, SGD, RMSprop).
    * Test metrics: Stored in subdirectories under `gan/gan_test_results_ablations_no_parser/` (e.g., `GAN_Adam_Adam_DefaultLR_Best/metrics2.json`, `GAN_SGD_SGD_DefaultLR_Best/metrics_sgd.json`, `GAN_rmsprop_rmsprop_DefaultLR_Best/metrics_rms.json`).

The JSON files contain metrics such as average L1 loss, PSNR, and SSIM, which can be used to compare the performance of different model configurations.
