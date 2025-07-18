# DiffFormer: A Diffusion-based Transformer for Cell Type Deconvolution

## Project Overview

This study introduces DiffFormer, a novel deep learning model for cell type deconvolution in bulk RNA-seq data. DiffFormer leverages a diffusion-based transformer architecture to accurately estimate cell type proportions from complex tissue samples. By modeling the inherent noise and variability in gene expression, our approach demonstrates superior performance and generalization across diverse biological samples, from peripheral blood to solid organs.

## Datasets

This study utilized four publicly available scRNA-seq datasets to comprehensively evaluate the performance of our model.

### Peripheral Blood Mononuclear Cell (PBMC) Datasets
- **pbmc3k**: Sourced from 10x Genomics, this dataset contains approximately 2,700 cells, which we annotated into 8 major immune cell types. ([Link](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k))
- **pbmc68k**: Also from 10x Genomics, this dataset includes about 68,000 cells, with 5 major pre-annotated cell clusters used as a reference. ([Link](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc68k))

### Organ Tissue Datasets
- **Liver Dataset**: Originating from a study by Camp et al. (2017), this dataset comprises scRNA-seq of human liver tissue. After quality control, approximately 2,100 cells across 6 cell types were used. (GEO Accession: [GSE81252](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81252))
- **Pancreas Dataset**: From a study by Muraro et al. (2016) on human pancreatic tissue, this dataset includes about 1,900 cells after filtering, with a complex composition of 9 cell types. (GEO Accession: [GSE85241](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241))

## Results

### Training Loss Curve

![Training Loss Curve](results/pbmc3k/transformer_loss_curve.png)

**Figure 1: Training Loss Curve of DiffFormer on the pbmc3k Dataset.** The figure shows the raw average loss values for each training epoch, along with a trend line smoothed by a moving average, visually depicting the convergence dynamics during model training.

## Data Availability Statement

The code developed for this study is openly available in this GitHub repository. The public datasets analyzed in this study can be found at:
- **PBMC Datasets**: Available from the 10x Genomics website ([pbmc3k](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k) and [pbmc68k](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc68k)).
- **Liver Dataset**: Available from the Gene Expression Omnibus (GEO) under accession number [GSE81252](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81252).
- **Pancreas Dataset**: Available from the Gene Expression Omnibus (GEO) under accession number [GSE85241](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241).

## Getting Started

### 1. Create the Conda Environment

This project's dependencies are managed with Conda. The `environment.yml` file contains all necessary packages for a fully reproducible environment. Create the environment using the following command. This will create a new conda environment named `DiffFormer`.

```bash
conda env create -f environment.yml
```

**2. Activate the Environment**

Before running any scripts, activate the newly created environment:
```bash
conda activate DiffFormer
```

## Repository Structure

The project is organized to ensure clarity and reproducibility:

```
├── data/                   # Raw and processed datasets (pbmc3k, pbmc68k, liver, pancreas)
├── environment.yml         # Conda environment configuration for reproducibility
├── README.md               # This README file
├── results/                # All outputs: model checkpoints, predictions, and final plots
└── src/                    # All Python source code
    ├── comparison/         # Scripts for running baseline methods (MuSiC, ADAPTS, CPM)
    ├── data_processing/    # Scripts for data fetching, QC, and pseudo-bulk generation
    ├── inference/          # Scripts to run deconvolution with trained models
    ├── model/              # PyTorch model implementations (DiffFormer, DiffMLP)
    ├── training/           # Scripts for training the models
    └── visualization/      # Scripts to generate final plots
```

## Reproducibility Guide

To reproduce all results, follow these steps in order. All commands should be run from the root of the project repository.

### 1. Data Processing

First, process the raw data for all four datasets. This involves quality control, normalization, and generating the pseudo-bulk samples for training and testing.

```bash
# Process PBMC datasets
python src/data_processing/run_pbmc3k_processing.py
python src/data_processing/run_pbmc68k_processing.py

# Process Liver dataset
python src/data_processing/run_liver_processing.py

# Process Pancreas dataset
python src/data_processing/run_pancreas_processing.py
```

### 2. Model Training

Next, train the `DiffFormer` and `DiffMLP` models on all four datasets. Checkpoints will be saved to the `results/{dataset_name}/` directory.

```bash
# --- Train on pbmc3k ---
python src/training/run_pbmc3k_training.py       # DiffFormer
python src/training/run_pbmc3k_training_mlp.py  # DiffMLP

# --- Train on pbmc68k ---
python src/training/run_pbmc68k_training.py       # DiffFormer
python src/training/run_pbmc68k_training_mlp.py  # DiffMLP

# --- Train on liver ---
python src/training/run_liver_training.py       # DiffFormer
python src/training/run_liver_training_mlp.py  # DiffMLP

# --- Train on pancreas ---
python src/training/run_pancreas_training.py       # DiffFormer
python src/training/run_pancreas_training_mlp.py  # DiffMLP
```

### 3. Deconvolution / Inference

With the trained models, run inference to generate the predicted cell type proportions for all models (DiffFormer, DiffMLP, and baselines). The main inference scripts automatically trigger the baseline models (`ADAPTS`, `MuSiC`, `CPM`). Predictions are saved as `.npy` files in the `results/{dataset_name}/` directory.

```bash
# --- Run inference on pbmc3k ---
python src/inference/run_pbmc3k_deconvolution.py
python src/inference/run_pbmc3k_deconvolution_mlp.py

# --- Run inference on pbmc68k ---
python src/inference/run_pbmc68k_deconvolution.py
python src/inference/run_pbmc68k_deconvolution_mlp.py

# --- Run inference on liver ---
python src/inference/run_liver_deconvolution.py
python src/inference/run_liver_deconvolution_mlp.py

# --- Run inference on pancreas ---
python src/inference/run_pancreas_deconvolution.py
python src/inference/run_pancreas_deconvolution_mlp.py
```