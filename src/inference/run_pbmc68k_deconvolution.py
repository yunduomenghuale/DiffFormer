import torch
import numpy as np
import json
from pathlib import Path
import sys
import time
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

from model.network import ConditionalDenoisingTransformer
from diffusers import DDPMScheduler
from comparison.run_music import run_music
from comparison.run_adapts import run_adapts
from comparison.run_cpm import run_cpm

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
    proportions = (raw_output + 1) / 2
    proportions = torch.clamp(proportions, 0, 1)
    proportions = proportions / proportions.sum()
    return proportions.numpy()

def main():
    """
    Main function to run deconvolution for the pbmc68k dataset.
    """
    dataset_name = "pbmc68k"
    MODEL_TYPE = "Transformer"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Deconvolution Pipeline for '{dataset_name}' on {device} ---")

    data_dir = project_root / "data" / dataset_name / "processed"
    results_dir = project_root / "results" / dataset_name
    checkpoints_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = checkpoints_dir / f"{MODEL_TYPE.lower()}_epoch_150.pth"

    print("\n--- Loading Data ---")
    try:
        with open(data_dir / "cell_types.json", 'r') as f: cell_types = json.load(f)
        with open(data_dir / "genes.json", 'r') as f: genes = json.load(f)
        test_bulk_samples_np = np.load(data_dir / "test_bulk_samples.npy")
        ground_truth = np.load(data_dir / "test_proportions.npy")
        signature_matrix_df = pd.read_csv(data_dir / "signature_matrix.csv", index_col=0)
    except FileNotFoundError as e:
        print(f"Error: Missing data file {e.filename}. Please run data processing first.")
        sys.exit(1)

    bulk_samples_df = pd.DataFrame(test_bulk_samples_np.T, index=genes)
    
    print("\n--- Loading Our Trained Model ---")
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}. Please train the model first.")
        sys.exit(1)

    transformer_params = {"model_dim": 128, "nhead": 4, "num_encoder_layers": 3, "dim_feedforward": 256}
    model = ConditionalDenoisingTransformer(
        proportion_dim=len(cell_types),
        bulk_expr_dim=len(genes),
        **transformer_params
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)

    all_predictions = {}

    print("\n--- Running Comparison Models ---")
    all_predictions['MuSiC'] = run_music(bulk_samples_df, signature_matrix_df, dataset_name).loc[cell_types].values.T
    np.save(results_dir / "music_predicted_proportions.npy", all_predictions['MuSiC'])
    
    all_predictions['ADAPTS'] = run_adapts(bulk_samples_df, signature_matrix_df).loc[cell_types].values.T
    np.save(results_dir / "adapts_predicted_proportions.npy", all_predictions['ADAPTS'])

    all_predictions['CPM'] = run_cpm(bulk_samples_df, signature_matrix_df).loc[cell_types].values.T
    np.save(results_dir / "cpm_predicted_proportions.npy", all_predictions['CPM'])
    print("Comparison models have finished running.")

    print("\n--- Running Our Diffusion Model ---")
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    our_preds = []
    for i in tqdm(range(len(test_bulk_samples_np)), desc="Deconvolving with Our Model"):
        bulk_sample = torch.from_numpy(test_bulk_samples_np[i]).float()
        raw_output = deconvolve(model, scheduler, bulk_sample, device, 1000, len(cell_types))
        our_preds.append(postprocess_proportions(raw_output))
    all_predictions['Diffusion'] = np.array(our_preds)
    np.save(results_dir / "diffusion_predicted_proportions.npy", all_predictions['Diffusion'])
    print("Our model has finished running and predictions are saved.")

    print("\n--- Final Performance Metrics ---")
    metrics = []
    for name, preds in all_predictions.items():
        rmse = np.sqrt(np.mean((preds - ground_truth)**2))
        overall_corr, _ = pearsonr(preds.flatten(), ground_truth.flatten())
        metrics.append({'Model': name, 'Overall RMSE': rmse, 'Overall Pearson R': overall_corr})
    
    metrics_df = pd.DataFrame(metrics).round(6)
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main() 