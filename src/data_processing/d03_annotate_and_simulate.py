import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse

def annotate_and_simulate(dataset_name: str, mode: str):
    """
    Annotates cell types (if necessary) and generates pseudo-bulk samples.
    Can generate 'train' or 'test' sets.
    
    Args:
        dataset_name (str): The name of the dataset ('pbmc3k', 'pbmc68k').
        mode (str): The set to generate ('train' or 'test').
    """
    # --- 1. Define Paths and Load Data ---
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        project_root = Path('.').resolve()

    processed_data_path = project_root / "data" / dataset_name / "processed" / f"{dataset_name}_processed.h5ad"
    output_dir = project_root / "data" / dataset_name / "processed"
    
    if not processed_data_path.exists():
        print(f"Error: Processed data not found at {processed_data_path}")
        print("Please run '02_preprocess_data.py' first.")
        return

    print(f"Loading processed data for '{dataset_name}' from {processed_data_path}...")
    adata = sc.read(processed_data_path)

    # --- 2. Annotate Cell Types (if needed) ---
    if dataset_name == 'pbmc3k':
        print("\nAnnotating cell types for pbmc3k...")
        # Mapping for pbmc3k based on louvain clusters
        louvain_to_celltype = {
            '0': 'CD4 T-cell', '1': 'CD14+ Monocyte', '2': 'B-cell', 
            '3': 'CD8 T-cell', '4': 'NK cell', '5': 'FCGR3A+ Monocyte',
            '6': 'Dendritic cell', '7': 'Megakaryocyte'
        }
        adata.obs['cell_type'] = adata.obs['louvain'].map(louvain_to_celltype).astype('category')
    elif dataset_name == 'pbmc68k':
        print("\nChecking for pre-annotated cell types for pbmc68k...")
        # If 'cell_type' isn't present, fall back to using 'louvain' clusters.
        if 'cell_type' not in adata.obs.columns:
            print("  Warning: 'cell_type' column not found.")
            print("  Falling back to using 'louvain' clusters as cell type labels.")
            if 'louvain' not in adata.obs.columns:
                raise ValueError(
                    "Fatal: Neither 'cell_type' nor 'louvain' column found. "
                    "Please ensure the preprocessing script is run correctly and generates clusters."
                )
            adata.obs['cell_type'] = adata.obs['louvain'].astype('category')
        else:
            print("  Found 'cell_type' column. Using pre-annotated labels.")
            # Make sure it's a categorical type for consistency
            adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    
    # Drop cells with unassigned/NaN cell types
    adata = adata[~adata.obs['cell_type'].isna()]

    print("\nCell types in use. Value counts:")
    print(adata.obs['cell_type'].value_counts())
    
    cell_types = adata.obs['cell_type'].cat.categories.tolist()

    # --- 3. Construct and Save Signature Matrix (only for training mode) ---
    if mode == 'train':
        print("\nConstructing and saving the signature matrix...")
        # Use a more robust way to create the signature matrix
        signature_matrix = pd.DataFrame(
            index=adata.var_names,
            columns=cell_types,
            dtype=np.float64
        )
        for cell_type in cell_types:
            # Calculate mean expression, handling both sparse and dense matrices
            mean_expr = adata[adata.obs.cell_type == cell_type].X.mean(axis=0)
            # Ensure it's a flattened numpy array
            if hasattr(mean_expr, 'A1'):
                signature_matrix[cell_type] = mean_expr.A1
            else:
                signature_matrix[cell_type] = np.ravel(mean_expr)

        sig_matrix_path = output_dir / "signature_matrix.csv"
        signature_matrix.to_csv(sig_matrix_path)
        print(f"Signature matrix saved to {sig_matrix_path}")

    # --- 4. Simulate Pseudo-bulk Data ---
    n_samples = 5000 if mode == 'train' else 500
    n_cells_per_sample = 2000
    print(f"\nStarting pseudo-bulk simulation for '{mode}' set ({n_samples} samples)...")

    bulk_samples = []
    proportions = []
    
    grouped_cells = adata.obs.groupby('cell_type')
    cell_indices_by_type = {cell_type: group.index for cell_type, group in grouped_cells}

    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"  Generating sample {i+1}/{n_samples}...")
            
        random_props = np.random.dirichlet(np.ones(len(cell_types)), size=1)[0]
        n_cells_by_type = (random_props * n_cells_per_sample).astype(int)
        
        # Adjust proportions to sum to the total number of cells due to rounding
        diff = n_cells_per_sample - n_cells_by_type.sum()
        if diff > 0:
            n_cells_by_type[np.random.choice(len(cell_types))] += diff

        chosen_indices = []
        actual_counts = []
        for j, cell_type in enumerate(cell_types):
            n_to_sample = n_cells_by_type[j]
            actual_counts.append(n_to_sample)
            available_indices = cell_indices_by_type[cell_type]
            if len(available_indices) == 0:
                continue # Skip if no cells of this type exist
            sampled_indices = np.random.choice(available_indices, n_to_sample, replace=True)
            chosen_indices.extend(sampled_indices)
        
        if not chosen_indices:
            continue

        pseudo_bulk_adata = adata[chosen_indices, :]
        pseudo_bulk_expression = pseudo_bulk_adata.X.mean(axis=0)
        
        bulk_samples.append(pseudo_bulk_expression)
        
        # Store the actual proportions based on the sampled cells
        proportions.append(np.array(actual_counts) / sum(actual_counts))


    X_bulk = np.array(bulk_samples)
    y_proportions = np.array(proportions)
    
    print("\nSimulation complete.")
    print("Shape of bulk samples (X):", X_bulk.shape)
    print("Shape of proportions (y):", y_proportions.shape)

    # --- 5. Save Data ---
    prefix = "" if mode == 'train' else "test_"
    output_X_path = output_dir / f"{prefix}bulk_samples.npy"
    output_y_path = output_dir / f"{prefix}proportions.npy"
    
    print(f"\nSaving '{mode}' data...")
    np.save(output_X_path, X_bulk)
    np.save(output_y_path, y_proportions)
    
    if mode == 'train':
        print("Saving metadata (genes and cell types)...")
        with open(output_dir / "genes.json", 'w') as f:
            json.dump(adata.var_names.tolist(), f)
        with open(output_dir / "cell_types.json", 'w') as f:
            json.dump(cell_types, f)
        
    print(f"'{mode}' data saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate cell types and simulate pseudo-bulk data.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=['pbmc3k', 'pbmc68k'],
        help="The name of the dataset to process."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['train', 'test'],
        help="Whether to generate the 'train' or 'test' set."
    )
    args = parser.parse_args()
    
    annotate_and_simulate(args.dataset, args.mode) 