import scanpy as sc
from pathlib import Path

def fetch_and_save_pbmc3k():
    """
    Downloads the pbmc3k dataset and saves it as a raw .h5ad file.
    """
    # Define the output path for the raw data
    output_dir = Path("../../data/raw")
    output_file = output_dir / "pbmc3k_raw.h5ad"

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the data has already been downloaded
    if output_file.exists():
        print(f"Dataset already found at: {output_file}")
        print("Skipping download.")
        return

    print("Loading PBMC 3k dataset from scanpy...")
    # The pbmc3k() function automatically downloads the dataset
    adata = sc.datasets.pbmc3k()
    
    print(f"Dataset loaded successfully. AnnData object summary:")
    print(adata)

    print(f"\nSaving raw data to {output_file}...")
    adata.write(output_file)
    print("Data saved.")

if __name__ == "__main__":
    fetch_and_save_pbmc3k() 