# Retardance2Retardance (R2R)

**Unsupervised Denoising for Polarization-Sensitive Optical Coherence Tomography**

This repository contains the PyTorch implementation of the Retardance2Retardance (R2R) framework, a physics-constrained, self-supervised deep learning architecture designed for robust polarimetric speckle suppression in PS-OCT.

## Abstract

Polarization-sensitive optical coherence tomography (PS-OCT) enables advanced micron-scale characterization of biological tissues. However, its quantitative diagnostic utility is fundamentally limited by the presence of speckle and associated polarimetric noise. Conventional spatial filtering techniques suppress speckle effects yet compromise spatial resolution and can introduce severe coordinate-transformation artifacts in catheter-based applications. To address these challenges, we present the Retardance2Retardance (R2R) network, a physics-constrained, self-supervised deep learning framework for robust polarimetric speckle suppression. R2R exploits spectral binning to obtain independent speckle realizations from existing data and train the model in a Noise2Noise fashion without ground truth. A physics-informed regularization in the loss function enforces that the denoised polarization effects correspond to retardance. Additionally, a physics-informed data augmentation (PIDA) strategy prevents the network from overfitting to the static polarimetric system effects inherent to the training data. Applied to independent human coronary artery pullbacks acquired in vivo, R2R effectively suppresses speckle-induced polarimetric noise while preserving fine polarimetric features.

## Prerequisites

Ensure your Python environment is configured with the necessary dependencies. The following package versions are recommended for optimal compatibility, though other recent versions may also work:

* `numpy==2.4.3`
* `scipy==1.17.1`
* `tensorboard==2.20.0`
* `torch==2.11.0`
* `torchsummary==1.5.1`
* `torchvision==0.26.0`
* `matplotlib==3.10.8`

## Repository Structure

* **`train.py`**: The primary entry point for training the network. It handles argument parsing for hyperparameters and manages the training pipeline.
* **`inference.py`**: The testing script utilized to evaluate the model and reconstruct denoised images using a pre-trained checkpoint.
* **`Network.py`**: Defines the U-Net-style convolutional neural network architecture with encoder-decoder blocks and global skip connections. It manages the training loop, residual learning mechanics, and the hybrid objective function that balances data fidelity with physics-informed scale-invariant orthogonality regularization.
* **`MultiEpochsDataLoader.py`**: Contains custom PyTorch Dataset and DataLoader classes tailored for reading 7-channel `.bin` files, pairing independent source-target speckle realizations (aligned spectral bins), and normalizing OCT intensity data.
* **`PSProcessing.py`**: A utility module for physical and mathematical operations, including generating 3D rotation matrices for physics-informed data augmentation (PIDA) and calculating the Degree of Polarization (DOP).

## Usage

### Training

To train the R2R network using independent speckle realizations with dataset augmentation enabled, use the following command:

```bash
python train.py --train-dir ./train/ --valid-dir ./valid/ --nb-epochs 100 --batch-size 16 --cuda --Trueaugmentation

```
During training, model checkpoints and TensorBoard logs will be automatically saved in the **`./runs/`** directory.

### Sample Data & Inference

To facilitate immediate testing, we have provided a sample single B-scan transmission vector dataset in the **`./test/`** directory/. This sample data contains the aligned source and target spectral bin **`.bin`** files (e.g., **`Train_Sym_Vol001_Frame121_Bin1_source.bin`**). Each spectral bin file utilizes a data structure of 1024 &times; 512 &times; 7 (Depth &times; A-lines &times; Channels) and is saved in **`numpy.float32`** precision.

To run evaluations and denoise your testing data utilizing a saved model checkpoint, run:

```bash
python inference.py --test-dir ./test/ --load-ckpt ./runs/R2R_Pretrained/R2R_Pretrained.pt --cuda

```
After inference is complete, the denoised results will be automatically saved in the **`./Results/`** directory. The output filenames will be explicitly marked with the word **`output`** (e.g., **`Train_Sym_Vol001_Frame121_Bin1_output.bin`**) so they can be easily identified.

The denoised inference results will be automatically saved in the **`./Results/`** directory.
