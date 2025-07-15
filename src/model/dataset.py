import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class DeconvolutionDataset(Dataset):
    """
    PyTorch Dataset for loading the simulated bulk RNA-seq data and
    corresponding cell type proportions.
    """
    def __init__(self, data_dir=None):
        """
        Args:
            data_dir (str, optional): Directory where the .npy files are stored.
                                      If None, it defaults to '<project_root>/data/processed'.
        """
        if data_dir is None:
            # Automatically determine the project root and data directory
            # Assumes the script is in <project_root>/src/model/
            script_path = Path(__file__).resolve()
            project_root = script_path.parent.parent.parent
            data_dir = project_root / "data" / "processed"

        data_path = Path(data_dir)
        bulk_samples_path = data_path / "bulk_samples.npy"
        proportions_path = data_path / "proportions.npy"

        if not bulk_samples_path.exists() or not proportions_path.exists():
            raise FileNotFoundError(
                f"Required data files not found in {data_path}. "
                "Please run the data processing scripts first."
            )
            
        print("Loading data into memory...")
        # Load the data and convert to float32, the standard for PyTorch
        self.bulk_samples = torch.from_numpy(np.load(bulk_samples_path)).float()
        self.proportions = torch.from_numpy(np.load(proportions_path)).float()
        print("Data loaded successfully.")
        
        # We need to normalize proportions from [0, 1] to [-1, 1]
        # This is a common practice for diffusion models to better match the noise distribution
        self.proportions = self.proportions * 2 - 1

        print(f"Dataset size: {len(self.bulk_samples)} samples.")
        print(f"Bulk sample tensor shape: {self.bulk_samples.shape}")
        print(f"Proportions tensor shape: {self.proportions.shape}")


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.bulk_samples)

    def __getitem__(self, idx):
        """
        Fetches the sample and its corresponding proportions at the given index.

        Returns:
            tuple: (bulk_sample, proportion)
        """
        bulk_sample = self.bulk_samples[idx]
        proportion = self.proportions[idx]
        return bulk_sample, proportion

if __name__ == '__main__':
    # You can run this script directly to test if the dataset loads correctly.
    print("Testing the DeconvolutionDataset...")
    try:
        dataset = DeconvolutionDataset()
        # Fetch the first sample to test __getitem__
        sample_x, sample_y = dataset[0]
        print("\\nSuccessfully loaded dataset.")
        print("Shape of one bulk sample (X):", sample_x.shape)
        print("Shape of one proportion vector (y):", sample_y.shape)
        print("Value of the first proportion vector (y[0]):", sample_y)
        print("Note: Proportions are scaled to [-1, 1].")
    except Exception as e:
        print(f"An error occurred: {e}") 