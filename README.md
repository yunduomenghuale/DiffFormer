
# DiffFormer: A Transformer-based Diffusion Model for Deconvolution of Bulk RNA-seq Data


## Overview

DiffFormer is a deep learning-based framework for deconvolving bulk RNA-seq data to infer cell type proportions. The project leverages a conditional diffusion model with a Transformer or MLP backbone to learn the complex relationships between bulk gene expression and cell type fractions.

The core idea is to train a model that can reverse a diffusion process, starting from pure noise and a conditioning bulk expression profile, to generate a verisimilar cell type proportion vector.

## Features

- **Diffusion Model Core**: Utilizes a Denoising Diffusion Probabilistic Model (DDPM) to generate accurate cell type proportions.
- **Conditional Generation**: Conditions the generation process on bulk RNA-seq expression data.
- **Flexible Backbones**: Implements both a `Transformer` and a simple `MLP` as the denoising network, allowing for a comparison between attention-based and standard architectures.
- **Multiple Datasets**: Includes data processing, training, and inference pipelines for four public datasets:
    - PBMC3k
    - PBMC68k
    - Pancreas
    - Liver
- **Benchmarking**: Compares the model's performance against established deconvolution methods like MuSiC, ADAPTS, and CPM.

## Project Structure

```
├── data/
│   ├── pbmc68k/
│   │   ├── processed/
│   │   └── raw/
│   ├── ... (other datasets)
├── results/
│   ├── pbmc68k/
│   │   ├── checkpoints/
│   │   └── ... (predicted proportions, plots)
│   ├── ... (other datasets)
├── src/
│   ├── model/
│   │   ├── network.py       # Defines the Transformer and MLP models
│   │   └── dataset.py       # PyTorch dataset for loading data
│   ├── training/
│   │   └── run_pbmc68k_training.py # Training script for a dataset
│   ├── inference/
│   │   └── run_pbmc68k_deconvolution.py # Inference and benchmarking script
│   └── comparison/
│       ├── run_music.py     # Scripts to run comparison methods
│       └── ...
└── environment.yml
```

## How It Works

### 1. Data Processing

The raw single-cell and bulk RNA-seq data (not included in the repo to save space, but assumed to be acquired) are processed into a format suitable for the model. This involves:
- Gene filtering and normalization.
- Simulation of pseudo-bulk samples by aggregating single-cell profiles.
- Creation of signature matrices.
- Saving processed data (`bulk_samples.npy`, `proportions.npy`, etc.) in the `data/<dataset_name>/processed` directory.

### 2. Model Architecture

The core of the project is the conditional denoising model, implemented in `src/model/network.py`. Two architectures are provided:
- `ConditionalDenoisingTransformer`: Uses a Transformer encoder to process the concatenated noisy proportions, timestep embedding, and bulk expression embedding.
- `ConditionalDenoisingMLP`: A simpler feed-forward network for the same task.

### 3. Training

The training process is handled by scripts in `src/training/`. For a given dataset:
1.  A `DeconvolutionDataset` (`src/model/dataset.py`) loads the processed bulk samples and their corresponding ground-truth proportions.
2.  The `DDPMScheduler` from the `diffusers` library is used to manage the forward (noising) process.
3.  In each training step:
    - A `clean_proportion` vector is selected.
    - Noise is added at a random `timestep`.
    - The model (`ConditionalDenoisingTransformer` or `MLP`) is tasked to predict the added `noise` based on the `noisy_proportion`, the `timestep`, and the conditioning `bulk_expression` profile.
4.  The Mean Squared Error between the predicted noise and the actual noise is used as the loss function.
5.  Model checkpoints are saved periodically to the `results/<dataset_name>/checkpoints/` directory.

### 4. Inference and Deconvolution

The inference scripts in `src/inference/` perform the deconvolution:
1.  The script loads a trained model checkpoint.
2.  It takes a real or simulated bulk expression sample as input.
3.  It runs the reverse diffusion process: starting with pure Gaussian noise, the model iteratively denoises the sample for a fixed number of timesteps, guided by the bulk expression profile.
4.  The final denoised output is a predicted cell type proportion vector.
5.  This output is post-processed (scaled to [0, 1] and normalized) to ensure it represents a valid probability distribution.
6.  The script also runs other deconvolution methods (`MuSiC`, `ADAPTS`, `CPM`) on the same data to provide a performance benchmark.
7.  Finally, it calculates and prints performance metrics (RMSE, Pearson correlation) for all methods.

## How to Run

### 1. Setup Environment

First, create the Conda environment using the provided file:
```bash
conda env create -f environment.yml
conda activate diffformer
```

### 2. Data
*Note: The processed data is provided in this repository, but the scripts for data fetching and processing were part of the original project structure and can be recreated if needed.*

Ensure the processed data files are located in the `data/<dataset-name>/processed/` directories.

### 3. Training a Model

To train a model on a specific dataset, run the corresponding training script. For example, to train the Transformer model on the PBMC68k dataset:
```bash
python src/training/run_pbmc68k_training.py
```
Checkpoints will be saved in `results/pbmc68k/checkpoints/`.

### 4. Running Deconvolution

After training, you can run inference and compare the model against benchmarks. Make sure the checkpoint path in the inference script matches the one saved during training.
```bash
python src/inference/run_pbmc68k_deconvolution.py
```
This will:
- Load the trained DiffFormer model.
- Run deconvolution on the test set.
- Run MuSiC, ADAPTS, and CPM for comparison.
- Save all predicted proportions to `.npy` files in `results/pbmc68k/`.
- Print a summary of the performance metrics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
