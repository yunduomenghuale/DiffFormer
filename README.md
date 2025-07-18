
**1. Create the Conda Environment**

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

### 4. Final Analysis and Visualization

Finally, generate all the plots and statistical analyses. The final figures will be saved in the `results/` and `results/{dataset_name}` directories.

```bash
# Generate individual plots for each dataset
python src/visualization/run_pbmc3k_plotting.py
python src/visualization/run_pbmc68k_plotting.py
python src/visualization/run_liver_plotting.py
python src/visualization/run_pancreas_plotting.py

# Generate the main summary plot comparing RMSE across all datasets
python src/visualization/run_final_comparison_plot.py
``` 