import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys

def get_rmse_df_for_dataset(dataset_name: str, project_root: Path) -> pd.DataFrame:
    """
    Loads data for a given dataset and returns a DataFrame with RMSE values.
    """
    data_dir = project_root / "data" / dataset_name / "processed"
    results_dir = project_root / "results" / dataset_name

    print(f"--- Loading data for '{dataset_name}' ---")
    try:
        ground_truth = np.load(data_dir / "test_proportions.npy")
        
        # Consistent model naming for different datasets
        is_new_dataset = dataset_name in ['liver', 'pancreas']
        mlp_pred_file = 'diffmlp_predicted_proportions.npy' if is_new_dataset else 'mlp_predicted_proportions.npy'
        
        predictions = {
            'DiffFormer': np.load(results_dir / "diffusion_predicted_proportions.npy"),
            'DiffMLP': np.load(results_dir / mlp_pred_file),
            'ADAPTS': np.load(results_dir / "adapts_predicted_proportions.npy"),
            'MuSiC': np.load(results_dir / "music_predicted_proportions.npy"),
            'CPM': np.load(results_dir / 'cpm_predicted_proportions.npy')
        }
    except FileNotFoundError as e:
        print(f"Error: A required data file is missing for dataset '{dataset_name}': {e.filename}", file=sys.stderr)
        print("Please ensure all deconvolution scripts have been run successfully.", file=sys.stderr)
        sys.exit(1) # Exit if data is missing

    # Calculate RMSE
    rmse_data = []
    for model_name, preds in predictions.items():
        # Ensure predictions and ground truth have the same shape
        if preds.shape != ground_truth.shape:
             print(f"Warning: Shape mismatch for model '{model_name}' in dataset '{dataset_name}'. Skipping.", file=sys.stderr)
             print(f"Preds shape: {preds.shape}, Ground truth shape: {ground_truth.shape}", file=sys.stderr)
             continue
        per_sample_rmse = np.sqrt(np.mean((preds - ground_truth)**2, axis=1))
        for rmse in per_sample_rmse:
            rmse_data.append({'Model': model_name, 'RMSE': rmse})
            
    df = pd.DataFrame(rmse_data)
    df['Dataset'] = dataset_name
    return df

def plot_combined_rmse():
    """
    Generates and saves a combined RMSE boxplot for both pbmc3k and pbmc68k datasets.
    """
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
    except NameError:
        project_root = Path('.').resolve()
        
    results_dir = project_root / "results"

    # Get RMSE data for all datasets
    rmse_df_3k = get_rmse_df_for_dataset("pbmc3k", project_root)
    rmse_df_68k = get_rmse_df_for_dataset("pbmc68k", project_root)
    rmse_df_liver = get_rmse_df_for_dataset("liver", project_root)
    rmse_df_pancreas = get_rmse_df_for_dataset("pancreas", project_root)

    # Combine dataframes
    combined_df = pd.concat([rmse_df_3k, rmse_df_68k, rmse_df_liver, rmse_df_pancreas], ignore_index=True)
    
    # Define the consistent model order
    model_order = ['DiffFormer', 'DiffMLP', 'ADAPTS', 'CPM', 'MuSiC']

    # --- Plotting ---
    print("\n--- Generating final combined comparison plot ---")
    sns.set_theme(style="whitegrid")

    # Create a single figure and axes for plotting
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

    # Use boxplot on the combined data, with 'hue' to differentiate datasets
    sns.boxplot(
        data=combined_df,
        x='Model',
        y='RMSE',
        hue='Dataset',
        palette='viridis',
        ax=ax,
        width=0.8,
        order=model_order
    )

    # Customize titles and labels
    ax.set_title('Model Performance Comparison across Datasets', fontsize=18, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=15, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(title='Dataset', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()

    # Save the final plot
    save_path = results_dir / "FINAL_RMSE_BOXPLOT.png"
    fig.savefig(save_path)
    print(f"Successfully saved combined RMSE boxplot to: {save_path}")


if __name__ == "__main__":
    plot_combined_rmse() 