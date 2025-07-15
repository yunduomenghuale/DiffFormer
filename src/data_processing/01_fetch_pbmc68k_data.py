import scanpy as sc
from pathlib import Path

def download_pbmc68k():
    """
    Downloads the 10x Genomics PBMC 68k dataset using the scanpy API.
    This is a reliable alternative to direct web downloads.
    The dataset is saved in the .h5ad format.
    """
    # Define paths
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        project_root = Path('.').resolve()
    
    output_dir = project_root / "data" / "pbmc68k" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = "pbmc68k_raw.h5ad"
    output_path = output_dir / file_name

    if output_path.exists():
        print(f"Dataset already exists at: {output_path}")
        return

    print("Downloading PBMC 68k dataset via scanpy...")
    
    try:
        # This function downloads and caches the data
        adata = sc.datasets.pbmc68k_reduced()
        
        # Save the downloaded data to our project structure
        adata.write(output_path)
        
        print(f"\nSuccessfully downloaded and saved to: {output_path}")
        print(f"Dataset details: {adata.n_obs} cells x {adata.n_vars} genes")

    except Exception as e:
        print(f"An error occurred while fetching the data with scanpy: {e}")

if __name__ == "__main__":
    download_pbmc68k() 