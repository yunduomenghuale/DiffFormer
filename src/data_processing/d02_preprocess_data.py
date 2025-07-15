import scanpy as sc
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import numpy as np

def preprocess_data(dataset_name: str):
    """
    Loads the raw data for a given dataset, performs standard preprocessing, 
    and saves the result.
    
    Args:
        dataset_name (str): The name of the dataset to process (e.g., 'pbmc3k', 'pbmc68k').
    """
    # --- Define dynamic paths based on dataset name ---
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        project_root = Path('.').resolve()

    data_dir = project_root / "data" / dataset_name
    raw_filename = f"{dataset_name}_raw.h5ad"
    processed_filename = f"{dataset_name}_processed.h5ad"
    
    raw_data_path = data_dir / "raw" / raw_filename
    processed_data_path = data_dir / "processed" / processed_filename
    
    # Create a results directory specific to this script's outputs
    script_results_dir = project_root / "results" / "data_processing"
    script_results_dir.mkdir(exist_ok=True)

    if not raw_data_path.exists():
        print(f"Error: Raw data file not found at {raw_data_path}")
        print("Please ensure the corresponding data fetching script has been run.")
        return

    print(f"Loading raw data for '{dataset_name}' from {raw_data_path}...")
    adata = sc.read(raw_data_path)
    
    # --- Preprocessing Steps ---
    print("\nStarting preprocessing...")

    # --- Data Inspector and Heuristic Check ---
    # Check if the data is already log-transformed.
    # Raw counts typically have high max values, log-transformed data doesn't.
    is_logged = np.max(adata.X) < 100
    if is_logged:
        print("Data appears to be already log-transformed. Skipping normalization and log1p.")
    else:
        print("Data appears to be raw counts. Proceeding with normalization and log1p.")

    # 1. Basic filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # After filtering genes, some cells might have 0 counts. Filter them out.
    sc.pp.filter_cells(adata, min_counts=1)

    # 2. Annotate mitochondrial genes
    # The 'MT-' prefix is standard for human data, so this should work for both PBMC datasets.
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Let's save a QC plot
    qc_plot_path = script_results_dir / f"qc_violin_{dataset_name}_before_filtering.png"
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True, show=False)
    plt.savefig(qc_plot_path)
    print(f"Saved QC violin plot to {qc_plot_path}")

    # 3. Filter cells based on QC metrics
    # These thresholds might need adjustment for different datasets.
    # For now, we use the same as for pbmc3k.
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    print(f"  Cells after filtering n_genes < 2500: {adata.n_obs}")
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    print(f"  Cells after filtering MT% < 5: {adata.n_obs}")

    # --- Add a safeguard ---
    if adata.n_obs == 0:
        raise ValueError("All cells were filtered out by QC metrics. Please check the filtering thresholds for this dataset.")

    # 4. Conditionally Normalize and log-transform
    if not is_logged:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # 5. Identify highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable] # Subset to highly variable genes

    # --- Perform Clustering to generate 'louvain' column ---
    print("\nRunning PCA, Neighbors, and Louvain clustering...")
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.louvain(adata)

    # Optional: Scale data
    sc.pp.scale(adata, max_value=10)
    
    print("\nPreprocessing complete.")
    print("Final AnnData object shape:", adata.shape)

    print(f"\nSaving processed data to {processed_data_path}...")
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(processed_data_path)
    print("Processed data saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw single-cell data.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=['pbmc3k', 'pbmc68k'],
        help="The name of the dataset to process (e.g., 'pbmc3k', 'pbmc68k')."
    )
    args = parser.parse_args()
    
    preprocess_data(args.dataset) 