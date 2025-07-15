import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import json
import sys

def plot_for_dataset():
    """
    Generates and saves a detailed comparison plot for the pbmc68k dataset.
    """
    dataset_name = "pbmc68k"

    # --- 1. Load Data ---
    try:
        # This works when running the script directly
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
    except NameError:
        # This works when running in an interactive environment
        project_root = Path('.').resolve()

    data_dir = project_root / "data" / dataset_name / "processed"
    results_dir = project_root / "results" / dataset_name

    print(f"--- Loading data for '{dataset_name}' plotting ---")
    try:
        ground_truth = np.load(data_dir / "test_proportions.npy")
        with open(data_dir / "cell_types.json", 'r') as f:
            cell_types = json.load(f)

        predictions = {
            'DiffFormer': np.load(results_dir / "diffusion_predicted_proportions.npy"),
            'DiffMLP': np.load(results_dir / "mlp_predicted_proportions.npy"),
            'ADAPTS': np.load(results_dir / "adapts_predicted_proportions.npy"),
            'MuSiC': np.load(results_dir / "music_predicted_proportions.npy"),
            'CPM': np.load(results_dir / 'cpm_predicted_proportions.npy')
        }
    except FileNotFoundError as e:
        print(f"Error: A required data file is missing: {e.filename}")
        print(f"Please ensure 'run_{dataset_name}_deconvolution.py' has been run successfully.")
        return

    # --- 2. Calculate Metrics ---
    rmse_data = []
    for model_name, preds in predictions.items():
        per_sample_rmse = np.sqrt(np.mean((preds - ground_truth)**2, axis=1))
        for rmse in per_sample_rmse:
            rmse_data.append({'Model': model_name, 'RMSE': rmse})
    rmse_df = pd.DataFrame(rmse_data)

    pcc_data = []
    for model_name, preds in predictions.items():
        for i, cell_type in enumerate(cell_types):
            with np.errstate(invalid='ignore'):
                corr, _ = pearsonr(ground_truth[:, i], preds[:, i])
            pcc_data.append({'Model': model_name, 'Cell Type': cell_type, 'PCC': corr})
    pcc_df = pd.DataFrame(pcc_data).pivot(index='Cell Type', columns='Model', values='PCC')

    # --- Enforce a consistent, logical order for all plots ---
    model_order = ['DiffFormer', 'DiffMLP', 'ADAPTS', 'CPM', 'MuSiC']
    pcc_df = pcc_df[model_order]

    # --- 3. Plotting ---
    print(f"--- Generating comparison plots for '{dataset_name}' ---")
    sns.set_theme(style="whitegrid")
    
    # Plot A: RMSE Boxplot
    fig_rmse, ax_rmse = plt.subplots(figsize=(8, 7), dpi=120)
    sns.boxplot(x='Model', y='RMSE', data=rmse_df, ax=ax_rmse, palette='viridis', 
                width=0.6, hue='Model', legend=False, order=model_order)
    ax_rmse.set_title(f'RMSE Distribution per Sample ({dataset_name})', fontsize=16, fontweight='bold')
    ax_rmse.set_xlabel('Model', fontsize=12)
    ax_rmse.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
    ax_rmse.tick_params(axis='x', rotation=15)
    fig_rmse.tight_layout()
    rmse_save_path = results_dir / f"FINAL_{dataset_name}_RMSE_BOXPLOT.png"
    fig_rmse.savefig(rmse_save_path)
    print(f"RMSE boxplot saved to: {rmse_save_path}")

    # Plot B: PCC Heatmap
    fig_pcc, ax_pcc = plt.subplots(figsize=(12, 8), dpi=120)
    sns.heatmap(pcc_df, ax=ax_pcc, annot=True, fmt=".3f", cmap="vlag", center=0, linewidths=.5)
    ax_pcc.set_title(f'Pearson Correlation (PCC) per Cell Type ({dataset_name})', fontsize=16, fontweight='bold')
    ax_pcc.set_xlabel('Model', fontsize=12)
    ax_pcc.set_ylabel('Cell Type', fontsize=12)
    ax_pcc.tick_params(axis='y', rotation=0)
    fig_pcc.tight_layout()
    pcc_save_path = results_dir / f"FINAL_{dataset_name}_PCC_HEATMAP.png"
    fig_pcc.savefig(pcc_save_path)
    print(f"PCC heatmap saved to: {pcc_save_path}")

if __name__ == "__main__":
    plot_for_dataset() 