import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Matplotlib settings for English fonts ---
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Specify default font
# plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of the negative sign '-' displaying as a square

def plot_comparison(sample_index=0):
    """
    Load the prediction results of all models and plot a comparison bar chart for the specified sample.
    """
    # --- Path settings ---
    project_root = Path(__file__).resolve().parent.parent.parent
    results_dir = project_root / 'results' / 'pbmc3k'
    data_dir = project_root / 'data' / 'pbmc3k' / 'processed'
    
    # --- Define models and corresponding result file names ---
    models = {
        'DiffFormer': 'diffusion_predicted_proportions.npy',
        'MuSiC': 'music_predicted_proportions.npy',
        'ADAPTS': 'adapts_predicted_proportions.npy',
        'CPM': 'cpm_predicted_proportions.npy'
    }

    print("--- Loading data required for plotting ---")
    try:
        # Load ground truth proportions and cell type names
        ground_truth = np.load(data_dir / 'test_proportions.npy')
        with open(data_dir / 'cell_types.json', 'r') as f:
            cell_types = json.load(f)

        # Load prediction results for all models
        predictions = {}
        for model_name, file_name in models.items():
            predictions[model_name] = np.load(results_dir / file_name)

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        print("Please ensure you have successfully run the deconvolution scripts.")
        return

    # --- Prepare data for plotting ---
    true_sample_props = ground_truth[sample_index]
    plot_data = {
        'Cell Type': cell_types,
        'Ground Truth': true_sample_props
    }
    for model_name, preds in predictions.items():
        plot_data[model_name] = preds[sample_index]
    df = pd.DataFrame(plot_data)

    # --- Start plotting ---
    print(f"--- Generating comparison chart for sample #{sample_index + 1} ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    n_models_total = len(models) + 1  # +1 for ground truth
    bar_width = 0.15
    index = np.arange(len(cell_types))
    
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd'] # Green for Ground Truth
    all_series = ['Ground Truth'] + list(models.keys())

    for i, series_name in enumerate(all_series):
        bar_positions = index + (i - (n_models_total - 1) / 2) * bar_width
        ax.bar(bar_positions, df[series_name] * 100, bar_width, label=series_name, color=colors[i])

    # --- Beautify the chart ---
    ax.set_title(f'Cell Type Proportion Prediction Comparison on pbmc3k Test Sample #{sample_index + 1}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Cell Type', fontsize=14)
    ax.set_ylabel('Cell Proportion (%)', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(cell_types, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12, title='Algorithm/Model')
    ax.grid(axis='x')

    plt.tight_layout()

    # --- Save the chart ---
    output_path = results_dir / f'pbmc3k_sample_{sample_index + 1}_comparison.png'
    plt.savefig(output_path, dpi=300)
    
    print(f"\nChart successfully saved to: {output_path}")

if __name__ == '__main__':
    # You can modify the sample_index here to view results for other test samples (e.g., 1, 2, ...)
    plot_comparison(sample_index=0)