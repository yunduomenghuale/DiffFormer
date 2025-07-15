import sys
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy.io import mmread

# Add the 'src' directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the reusable simulation function
from src.data_processing.d03_annotate_and_simulate import annotate_and_simulate

def preprocess_pancreas_data(dataset_name: str):
    """
    Loads pancreas data from .mtx/.tsv files, performs preprocessing, and saves the result.
    This version uses an in-memory approach to fix the MTX file to avoid I/O issues.
    """
    print(f"--- Starting {dataset_name.capitalize()} Data Preprocessing ---")
    
    # --- 1. Define Paths ---
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / dataset_name
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    processed_data_dir.mkdir(exist_ok=True)
    
    processed_filename = f"{dataset_name}_processed.h5ad"
    processed_data_path = processed_data_dir / processed_filename
    
    script_results_dir = project_root / "results" / "data_processing"
    script_results_dir.mkdir(exist_ok=True)

    # --- 2. Load Data from MTX and TSV files using in-memory fix ---
    print("Loading data from MTX and TSV files using robust in-memory method...")
    
    matrix_path = raw_data_dir / 'matrix.mtx'
    genes_path = raw_data_dir / 'genes.tsv'
    barcodes_path = raw_data_dir / 'barcodes.tsv'
    
    with open(matrix_path, 'r') as f:
        data_lines = [line for line in f if not line.startswith('%')]

    num_genes = pd.read_csv(genes_path, header=None, sep='\\t', engine='python').shape[0]
    num_cells = pd.read_csv(barcodes_path, header=None, sep='\\t', engine='python').shape[0]
    num_entries = len(data_lines)
    print(f"Inferred dimensions: {num_genes} genes, {num_cells} cells, {num_entries} non-zero entries.")

    banner = '%%MatrixMarket matrix coordinate real general\n'
    dimension_line = f"{num_genes} {num_cells} {num_entries}\n"
    full_content = banner + dimension_line + "".join(data_lines)
    string_io_file = io.StringIO(full_content)

    sparse_matrix = mmread(string_io_file).astype(np.float32)
    
    adata = sc.AnnData(sparse_matrix)
    adata = adata.T
    adata.X = adata.X.tocsr()

    barcodes = pd.read_csv(barcodes_path, header=None, sep='\\t', engine='python')
    adata.obs_names = barcodes[0]

    genes = pd.read_csv(genes_path, header=None, sep='\\t', names=['gene_id', 'gene_symbol'], engine='python')
    adata.var_names = genes['gene_id']
    adata.var['gene_symbol'] = genes['gene_symbol'].values

    cell_types = pd.read_csv(raw_data_dir / 'cell_type.tsv', header=None, sep='\\t', engine='python')
    adata.obs['cell_type'] = cell_types[0].values

    adata.var_names_make_unique()

    print("Raw data loaded successfully.")
    print(f"Initial AnnData object shape: {adata.shape}")

    # --- 3. Preprocessing Steps ---
    print("\\nStarting preprocessing...")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_counts=1)

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    qc_plot_path = script_results_dir / f"qc_violin_{dataset_name}_before_filtering.png"
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True, show=False)
    plt.savefig(qc_plot_path)
    plt.close()
    print(f"Saved QC violin plot to {qc_plot_path}")

    adata = adata[adata.obs.n_genes_by_counts < 4000, :]
    print(f"  Cells after filtering n_genes < 4000: {adata.n_obs}")
    adata = adata[adata.obs.pct_counts_mt < 10, :]
    print(f"  Cells after filtering MT% < 10: {adata.n_obs}")

    if adata.n_obs == 0:
        raise ValueError("All cells were filtered out. Check QC thresholds.")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]

    print("\\nRunning PCA, Neighbors, and Louvain clustering...")
    
    n_pcs = min(40, adata.n_obs - 1)
    n_neighbors = min(10, n_pcs)

    if n_pcs < 40:
        print(f"Warning: Low number of cells ({adata.n_obs}). Adjusting n_pcs to {n_pcs} and n_neighbors to {n_neighbors}.")

    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.louvain(adata)
    
    sc.pp.scale(adata, max_value=10)
    
    print("\\nPreprocessing complete.")
    print("Final AnnData object shape:", adata.shape)

    print(f"\\nSaving processed data to {processed_data_path}...")
    adata.write(processed_data_path)
    print("Processed data saved.")


def main():
    """
    Runs the full data processing pipeline for the pancreas dataset.
    """
    dataset_name = "pancreas"
    
    print(f"--- Starting full data processing pipeline for '{dataset_name}' ---")
    
    preprocess_pancreas_data(dataset_name)

    print("\\n[Step 2/3] Generating training set...")
    annotate_and_simulate(dataset_name, mode='train')
    print("[Step 2/3] Training set generation complete.")

    print("\\n[Step 3/3] Generating test set...")
    annotate_and_simulate(dataset_name, mode='test')
    print("[Step 3/3] Test set generation complete.")
    
    print(f"\\n--- Pipeline for '{dataset_name}' finished successfully! ---")

if __name__ == "__main__":
    main() 