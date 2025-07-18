import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import pandas as pd

# --- Path setup ---
try:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
except NameError:
    project_root = Path('.').resolve()
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from model.dataset import DeconvolutionDataset
from model.network import ConditionalDenoisingTransformer
from diffusers import DDPMScheduler

def run_training():
    """
    Main training function for the pbmc3k dataset.
    """
    dataset_name = "pbmc3k"
    
    # --- 1. Configuration ---
    config = {
        "epochs": 150,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "num_diffusion_timesteps": 1000,
        "save_checkpoint_every": 25,
        "model_type": "Transformer",
        "transformer_params": {
            "model_dim": 128,
            "nhead": 4,
            "num_encoder_layers": 3,
            "dim_feedforward": 256
        },
        "scheduler_params": {
            "T_max": 150,
            "eta_min": 1e-6
        }
    }
    
    # --- 2. Setup Directories and Device ---
    results_dir = project_root / "results" / dataset_name
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting training for '{dataset_name}' on {device} ---")

    # --- 3. Load Data ---
    data_dir = project_root / "data" / dataset_name / "processed"
    dataset = DeconvolutionDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    
    x_sample, y_sample = dataset[0]
    bulk_expr_dim = x_sample.shape[0]
    proportion_dim = y_sample.shape[0]

    # --- 4. Initialize Model, Scheduler, and Optimizer ---
    model = ConditionalDenoisingTransformer(
        proportion_dim=proportion_dim,
        bulk_expr_dim=bulk_expr_dim,
        **config["transformer_params"]
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_diffusion_timesteps"])
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, **config["scheduler_params"])

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Prepare log file
    log_file_path = results_dir / f'{config["model_type"].lower()}_training_log.csv'
    log_df = pd.DataFrame(columns=['epoch', 'avg_loss', 'learning_rate'])
    
    # --- 5. Training Loop ---
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for step, (bulk_expr, clean_proportions) in enumerate(progress_bar):
            bulk_expr = bulk_expr.to(device)
            clean_proportions = clean_proportions.to(device)
            noise = torch.randn_like(clean_proportions)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_proportions.shape[0],), device=device).long()
            noisy_proportions = noise_scheduler.add_noise(clean_proportions, noise, timesteps)
            noise_pred = model(noisy_proportions, timesteps, bulk_expr)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # Log metrics
        new_log_entry = pd.DataFrame([{'epoch': epoch + 1, 'avg_loss': avg_loss, 'learning_rate': current_lr}])
        log_df = pd.concat([log_df, new_log_entry], ignore_index=True)
        log_df.to_csv(log_file_path, index=False)

        if (epoch + 1) % config["save_checkpoint_every"] == 0 or epoch == config["epochs"] - 1:
            model_name = config["model_type"].lower()
            checkpoint_path = checkpoints_dir / f"{model_name}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        scheduler.step()

    print(f"--- Training for '{dataset_name}' complete! ---")

if __name__ == "__main__":
    run_training() 