import torch
import numpy as np
import json
from pathlib import Path
import sys
import pandas as pd
from scipy.stats import pearsonr
from tqdm.auto import tqdm

# --- Path setup ---
try:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
except NameError:
    project_root = Path('.').resolve()
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from model.network import ConditionalDenoisingMLP  # Changed import
from diffusers import DDPMScheduler

def deconvolve(model, scheduler, bulk_sample, device, num_inference_steps, output_dim):
    model.eval()
    sample = torch.randn(1, output_dim).to(device)
    bulk_sample = bulk_sample.unsqueeze(0).to(device)
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        with torch.no_grad():
            model_input_time = t.unsqueeze(0).to(device)
            noise_pred = model(sample, model_input_time, bulk_sample)
        sample = scheduler.step(noise_pred, t, sample).prev_sample
    return sample.squeeze(0).cpu()

def postprocess_proportions(raw_output):
    # Post-processing to ensure proportions are valid (sum to 1, non-negative)
    proportions = torch.clamp(raw_output, 0)
    if proportions.sum() > 0:
        proportions = proportions / proportions.sum()
    else:
        # Handle the edge case where all outputs are zero or negative
        proportions = torch.ones_like(raw_output) / raw_output.numel()
    return proportions.numpy()

def main():
    """
    Main function to run deconvolution for the pbmc68k dataset using the MLP model.
    """
    dataset_name = "pbmc68k"
    MODEL_TYPE = "MLP"
    CHECKPOINT_EPOCH = 150
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running MLP Deconvolution for '{dataset_name}' on {device} ---")

    # --- Setup Paths ---
    data_dir = project_root / "data" / dataset_name / "processed"
    results_dir = project_root / "results" / dataset_name
    checkpoints_dir = results_dir / f"checkpoints_{MODEL_TYPE.lower()}"
    results_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = checkpoints_dir / f"{MODEL_TYPE.lower()}_epoch_{CHECKPOINT_EPOCH}.pth"

    # --- Load Data ---
    print("\n--- Loading Data ---")
    try:
        with open(data_dir / "cell_types.json", 'r') as f: cell_types = json.load(f)
        with open(data_dir / "genes.json", 'r') as f: genes = json.load(f)
        test_bulk_samples_np = np.load(data_dir / "test_bulk_samples.npy")
        ground_truth = np.load(data_dir / "test_proportions.npy")
    except FileNotFoundError as e:
        print(f"Error: Missing data file {e.filename}. Please run data processing first.")
        sys.exit(1)

    # --- Load Trained MLP Model ---
    print("\n--- Loading Trained MLP Model ---")
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}.")
        print(f"Please ensure the MLP model has been trained by running 'run_{dataset_name}_training_mlp.py'")
        sys.exit(1)

    mlp_params = {"time_emb_dim": 32, "hidden_dims": [256, 512, 256]}
    model = ConditionalDenoisingMLP(
        proportion_dim=len(cell_types),
        bulk_expr_dim=len(genes),
        **mlp_params
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    print("MLP model loaded successfully.")

    # --- Run Deconvolution using the MLP model ---
    print("\n--- Running MLP Diffusion Model ---")
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    mlp_preds = []
    for i in tqdm(range(len(test_bulk_samples_np)), desc=f"Deconvolving with {MODEL_TYPE} Model"):
        bulk_sample = torch.from_numpy(test_bulk_samples_np[i]).float()
        raw_output = deconvolve(model, scheduler, bulk_sample, device, 1000, len(cell_types))
        mlp_preds.append(postprocess_proportions(raw_output))
    
    predictions = np.array(mlp_preds)
    output_path = results_dir / "mlp_predicted_proportions.npy"
    np.save(output_path, predictions)
    print(f"MLP model has finished running. Predictions saved to {output_path}")

    # --- Final Performance Metrics ---
    print("\n--- MLP Model Performance Metrics ---")
    rmse = np.sqrt(np.mean((predictions - ground_truth)**2))
    overall_corr, _ = pearsonr(predictions.flatten(), ground_truth.flatten())
    
    metrics_df = pd.DataFrame([{
        'Model': MODEL_TYPE, 
        'Overall RMSE': rmse, 
        'Overall Pearson R': overall_corr
    }]).round(6)
    
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main() 