
### README Template for Your Diffusion Model Project

---

# Diffusion Model with Classifier-Free Guidance

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** using a **UNet architecture**, with both **classifier-free guidance** and **unguided training**. The model is trained on a Gaussian Mixture Model (GMM) dataset and visualizes the denoising process over multiple timesteps.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Visualization](#visualization)
- [Changes and Improvements](#changes-and-improvements)
- [Known Issues](#known-issues)
- [Contact](#contact)

## Project Overview

This project implements a diffusion model that learns to reverse the diffusion process and generate clean samples from noisy data. The model uses **classifier-free guidance** to provide conditional control over the denoising process without requiring a separate classifier. The trained model can generate new samples and visualize their denoising trajectory over a specified number of timesteps.

The model is evaluated by:
- Visualizing the denoising trajectory of a particular sample over multiple timesteps (t=10, 20, 30, 40, 50).
- Comparing the generated sample distribution with the ground truth GMM distribution.
- Saving the visualizations as images.

## Installation

### Prerequisites

Make sure you have the following installed:
- Python 3.8+
- PyTorch
- Matplotlib (for visualization)
- SciPy (for KDE estimation)
- tqdm (for progress bars)

### Install the required packages:

```bash
pip install torch scipy matplotlib tqdm
```

## Usage

### 1. **Cloning the Repository**

First, clone the repository or download the project files.

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

### 2. **Running the Training Script**

To train the diffusion model, use the following command:

```bash
python train.py
```

This will:
- Train the model on a GMM dataset.
- Print out the loss for both guided and unguided training.
- After training, it will generate and save visualizations of the denoising process at various timesteps.

### 3. **Viewing the Output**

Once the training is complete, check the `denoising_images/` directory for the output images. Each image represents a sample's denoising trajectory over time, compared with the ground-truth GMM distribution.

## Training

The training process involves:
- A **UNet model** that learns to denoise the images.
- The **forward diffusion process**, where noise is added to the original data.
- The **reverse process**, where the model learns to predict the noise and remove it.
- The **loss function**, which measures the difference between the actual noise and the model's predicted noise.

### Parameters:
- **Epochs**: 100
- **Batch Size**: 64
- **Learning Rate**: 5e-4
- **Guidance Weight**: 0.5

These parameters can be adjusted in the `train.py` script.

## Visualization

At the end of the training, the denoising process is visualized by:
- Sampling noisy data at various timesteps.
- Applying the reverse diffusion process.
- Plotting the denoising trajectory of the sample overlaid on the estimated probability density function (PDF).

The results are saved as `.png` images in the `denoising_images/` folder.

## Changes and Improvements

- **Dense Layer Reshaping**: Adjustments were made to the dense layers to ensure proper reshaping and broadcast of time embeddings in the UNet architecture.
- **Guidance Mechanism**: Classifier-free guidance was incorporated to provide conditional control over the generated samples.
- **Visualization**: Added functionality to visualize the denoising trajectory of samples over time.

## Known Issues

- **Shape Mismatch**: During early stages of development, shape mismatches occurred in the UNet architecture when adding the time embeddings to the feature maps. These were resolved by correctly reshaping the dense layer outputs and broadcasting the time embeddings.
- **Denoising Process Stability**: The denoising process can sometimes be unstable at very high or low timesteps. This is managed by adjusting the diffusion parameters (`betas` and `alphas`).

# Portfolio
# eecs398portfolio
