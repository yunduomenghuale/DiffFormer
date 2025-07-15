import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy.stats import ttest_ind

def get_rmse_per_sample(dataset_name: str, project_root: Path) -> dict:
    """
    Loads data for a given dataset and returns a dictionary of per-sample RMSE values for each model.
    """
    data_dir = project_root / "data" / dataset_name / "processed"
    results_dir = project_root / "results" / dataset_name

    print(f"--- Loading data for '{dataset_name}' ---")
    try:
        ground_truth = np.load(data_dir / "test_proportions.npy")
        
        predictions = {
            'DiffFormer': np.load(results_dir / "diffusion_predicted_proportions.npy"),
            'DiffMLP': np.load(results_dir / "mlp_predicted_proportions.npy"),
            'ADAPTS': np.load(results_dir / "adapts_predicted_proportions.npy"),
            'MuSiC': np.load(results_dir / "music_predicted_proportions.npy"),
            'CPM': np.load(results_dir / 'cpm_predicted_proportions.npy')
        }
    except FileNotFoundError as e:
        print(f"Error: A required data file is missing for dataset '{dataset_name}': {e.filename}", file=sys.stderr)
        sys.exit(1)

    # Calculate per-sample RMSE for each model
    rmse_distributions = {}
    for model_name, preds in predictions.items():
        if preds.shape != ground_truth.shape:
             print(f"Warning: Shape mismatch for model '{model_name}' in dataset '{dataset_name}'. Skipping.", file=sys.stderr)
             continue
        per_sample_rmse = np.sqrt(np.mean((preds - ground_truth)**2, axis=1))
        rmse_distributions[model_name] = per_sample_rmse
            
    return rmse_distributions

def run_statistical_tests_for_dataset(dataset_name: str, rmse_df: pd.DataFrame):
    """Runs t-tests and prints the results for a single dataset."""
    
    print(f"\n--- Statistical Analysis for: {dataset_name.upper()} ---")

    our_model = "DiffFormer"
    
    # Define the consistent model order
    model_order = ['DiffFormer', 'DiffMLP', 'ADAPTS', 'CPM', 'MuSiC']
    
    # --- Descriptive Statistics ---
    print("\n--- Descriptive Statistics (RMSE) ---")
    
    # Calculate and print mean and std for each model in the defined order
    for model in model_order:
        if model in rmse_df['Model'].unique():
            model_rmse = rmse_df[rmse_df['Model'] == model]['RMSE']
            mean_rmse = model_rmse.mean()
            std_rmse = model_rmse.std()
            print(f"  - {model:<12}: Mean={mean_rmse:.4f}, Std={std_rmse:.4f}")
        
    print("\n" + "="*50)
    
    # --- T-Tests ---
    print(f"\n--- T-Tests: Comparing {our_model} against other models ---")
    
    if our_model not in rmse_df['Model'].unique():
        print(f"Our model '{our_model}' not found in the data for this dataset.")
        return
        
    our_model_rmse = rmse_df[rmse_df['Model'] == our_model]['RMSE']
    other_models = [m for m in model_order if m != our_model and m in rmse_df['Model'].unique()]
    
    results = []
    mean_our_model = our_model_rmse.mean()
    
    for model_to_compare in other_models:
        rmse_to_compare = rmse_df[rmse_df['Model'] == model_to_compare]['RMSE']
        mean_to_compare = rmse_to_compare.mean()
        
        # Perform independent two-sample t-test
        t_stat, p_value = ttest_ind(our_model_rmse, rmse_to_compare, equal_var=False) # Welch's t-test
        
        results.append({
            "Comparison": f"DiffFormer vs. {model_to_compare}",
            f"Mean RMSE (DiffFormer)": f"{mean_our_model:.4f}",
            f"Mean RMSE ({model_to_compare})": f"{mean_to_compare:.4f}",
            "T-statistic": f"{t_stat:.2f}",
            "P-value": f"{p_value:.2e}"
        })
        
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*50)

def main():
    """Main function to run the analysis."""
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
    except NameError:
        project_root = Path('.').resolve()

    datasets = ["pbmc3k", "pbmc68k"]
    
    print("--- Starting Statistical Significance Analysis ---")
    
    for dataset in datasets:
        # We need to load the data for each dataset to run the tests
        results_dir = project_root / "results" / dataset
        predictions = {
            'DiffFormer': np.load(results_dir / "diffusion_predicted_proportions.npy"),
            'DiffMLP': np.load(results_dir / "mlp_predicted_proportions.npy"),
            'ADAPTS': np.load(results_dir / "adapts_predicted_proportions.npy"),
            'MuSiC': np.load(results_dir / "music_predicted_proportions.npy"),
            'CPM': np.load(results_dir / 'cpm_predicted_proportions.npy')
        }
        ground_truth = np.load(project_root / "data" / dataset / "processed" / "test_proportions.npy")
        
        rmse_data = []
        for model_name, preds in predictions.items():
            per_sample_rmse = np.sqrt(np.mean((preds - ground_truth)**2, axis=1))
            for rmse in per_sample_rmse:
                rmse_data.append({'Model': model_name, 'RMSE': rmse})
        
        rmse_df = pd.DataFrame(rmse_data)
        
        run_statistical_tests_for_dataset(dataset, rmse_df)

if __name__ == "__main__":
    main() 