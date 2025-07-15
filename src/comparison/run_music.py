import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import nnls
from pathlib import Path

def run_music(bulk_samples_df, signature_matrix_df, dataset_name: str):
    """
    Performs deconvolution using the traditional MuSiC algorithm, including
    the gene weighting step based on cross-subject variance.

    NOTE: This function needs to load the original single-cell data to calculate
    the gene weights, which is a core part of the MuSiC algorithm.

    Args:
        bulk_samples_df (pd.DataFrame): DataFrame of bulk gene expression, with genes
                                        as rows and samples as columns.
        signature_matrix_df (pd.DataFrame): DataFrame of the cell-type signature matrix,
                                            with genes as rows and cell types as columns.
        dataset_name (str): The name of the dataset (e.g., 'pbmc3k', 'pbmc68k')
                            to load the correct single-cell reference.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted cell type proportions,
                      with cell types as rows and samples as columns.
    """
    print("Performing TRADITIONAL MuSiC deconvolution with gene weighting...")

    # --- Load single-cell data to calculate weights ---
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        project_root = Path('.').resolve()
    
    # Use the provided dataset_name to build the path dynamically
    sc_data_path = project_root / "data" / dataset_name / "processed" / f"{dataset_name}_processed.h5ad"
    if not sc_data_path.exists():
        raise FileNotFoundError(f"Single-cell data for MuSiC weighting not found at: {sc_data_path}")
    
    print("  Loading single-cell reference to calculate gene weights...")
    adata = sc.read(sc_data_path)

    # Use the correct annotation logic based on the dataset
    if 'cell_type' not in adata.obs.columns:
        print(f"  'cell_type' column not found in {dataset_name}, attempting to create it from 'louvain'.")
        # Generic fallback for pbmc3k-like data
        if dataset_name == 'pbmc3k':
            louvain_to_celltype = {
                '0': 'CD4 T-cell', '1': 'CD14+ Monocyte', '2': 'B-cell', '3': 'CD8 T-cell',
                '4': 'NK cell', '5': 'FCGR3A+ Monocyte', '6': 'Dendritic cell', '7': 'Megakaryocyte'
            }
            adata.obs['cell_type'] = adata.obs['louvain'].map(louvain_to_celltype).astype('category')
        else:
            # For other datasets like pbmc68k, we assume louvain clusters are the types
            adata.obs['cell_type'] = adata.obs['louvain'].astype('category')
    
    # Use louvain clusters as a proxy for subjects for variance calculation
    # This part is a heuristic and might need adjustment for real multi-subject data.
    adata.obs['subject_id'] = adata.obs['louvain'].astype(str)

    # --- Calculate MuSiC Weights ---
    print("  Calculating gene weights based on cross-cluster variance...")
    # Get relative expression (proportions within each cell)
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1)
    
    # Calculate variance for each gene within each cell type across "subjects"
    # Handle both sparse and dense matrix formats
    expression_matrix = adata_norm.X.toarray() if hasattr(adata_norm.X, "toarray") else adata_norm.X
    relative_expr_df = pd.DataFrame(expression_matrix, index=adata_norm.obs.index, columns=adata_norm.var.index)
    relative_expr_df['cell_type'] = adata_norm.obs['cell_type'].values
    relative_expr_df['subject'] = adata_norm.obs['subject_id'].values
    
    var_by_cell_type = relative_expr_df.groupby(['cell_type', 'subject'], observed=True)[adata_norm.var_names].mean().groupby('cell_type', observed=True).var()
    
    epsilon = 1e-10
    weights_df = 1 / (var_by_cell_type + epsilon)
    weights_df = weights_df.T.reindex(signature_matrix_df.index)

    # --- Perform Weighted Deconvolution ---
    # Align genes
    common_genes = signature_matrix_df.index.intersection(bulk_samples_df.index)
    M = signature_matrix_df.loc[common_genes].values
    bulk_aligned = bulk_samples_df.loc[common_genes].values
    
    # Create diagonal weight matrix from the mean weight of each gene
    # Fill NaN with 0 in case some genes have no variance info
    mean_weights = weights_df.loc[common_genes].mean(axis=1).fillna(0)
    W = np.diag(mean_weights)

    # Apply weights
    M_w = W @ M
    
    all_proportions = []

    for i in range(bulk_aligned.shape[1]):
        y = bulk_aligned[:, i]
        y_w = W @ y

        # Solve the W-NNLS problem
        result, _ = nnls(M_w, y_w)

        # Normalize proportions to sum to 1
        total = np.sum(result)
        if total > 0:
            result /= total
        else:
            num_cell_types = M.shape[1]
            result = np.full(num_cell_types, 1.0 / num_cell_types)
            
        all_proportions.append(result)

    # Create the results DataFrame
    proportions_df = pd.DataFrame(
        data=np.array(all_proportions).T,
        index=signature_matrix_df.columns,
        columns=bulk_samples_df.columns
    )

    return proportions_df 