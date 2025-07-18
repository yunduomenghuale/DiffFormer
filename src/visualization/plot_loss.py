import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

def setup_matplotlib_english_font():
    """
    Sets up matplotlib for English font display.
    """
    plt.rcParams['axes.unicode_minus'] = False

def plot_loss_curve(dataset_name="pbmc3k", model_type="transformer"):
    """
    Reads the training log and plots the original loss curve.
    
    Args:
        dataset_name (str): The name of the dataset.
        model_type (str): The type of the model (e.g., 'transformer').
    """
    # --- 1. Path and Font Setup ---
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
    except NameError:
        project_root = Path('.').resolve()
    
    results_dir = project_root / 'results' / dataset_name
    log_file_path = results_dir / f'{model_type}_training_log.csv'

    setup_matplotlib_english_font()

    # --- 2. Load Data ---
    print(f"--- Loading log file: {log_file_path} ---")
    try:
        log_df = pd.read_csv(log_file_path)
    except FileNotFoundError:
        print(f"Error: Log file not found!")
        print(f"Please run 'src/training/run_{dataset_name}_training.py' first to generate the log file.")
        return

    # --- 3. Plotting ---
    print("--- Generating loss curve plot ---")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    sns.set_theme(style="whitegrid")

    # Plot the original loss curve
    sns.lineplot(data=log_df, x='epoch', y='avg_loss', marker='.', linestyle='-', ax=ax, label='Average Loss', color='darkblue')

    # --- 4. Beautify the Chart ---
    ax.set_title(f'{model_type.capitalize()} Model Training Loss Curve on {dataset_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average MSE Loss', fontsize=12)
    ax.legend()
    
    if len(log_df['epoch']) > 20:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=15))
    
    plt.tight_layout()

    # --- 5. Save the Chart ---
    output_path = results_dir / f'{model_type}_loss_curve.png'
    fig.savefig(output_path)
    
    print(f"\nLoss curve plot successfully saved to: {output_path}")

if __name__ == '__main__':
    plot_loss_curve()