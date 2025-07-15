import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

def run_adapts(bulk_samples_df, signature_matrix_df):
    """
    Deconvolves bulk gene expression samples using an Elastic Net regression approach,
    inspired by the ADAPTS framework.

    This implementation uses ElasticNetCV to find the best alpha and l1_ratio
    through cross-validation. It enforces a non-negativity constraint on the
    predicted cell type proportions.

    Args:
        bulk_samples_df (pd.DataFrame): DataFrame of bulk gene expression, with genes
                                        as rows and samples as columns.
        signature_matrix_df (pd.DataFrame): DataFrame of the cell-type signature matrix,
                                            with genes as rows and cell types as columns.

    Returns:
        pd.DataFrame: A DataFrame containing the predicted cell type proportions,
                      with cell types as rows and samples as columns.
    """
    # Align genes between bulk samples and signature matrix
    common_genes = signature_matrix_df.index.intersection(bulk_samples_df.index)
    sig_matrix_aligned = signature_matrix_df.loc[common_genes]
    bulk_samples_aligned = bulk_samples_df.loc[common_genes]

    # Prepare data for sklearn
    X = sig_matrix_aligned.values
    y_bulk = bulk_samples_aligned.values

    # Set up ElasticNetCV with cross-validation
    # We use a custom CV splitter to ensure stability
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0] # Range of L1 ratios to test

    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        cv=cv_splitter,
        n_jobs=-1,
        random_state=42,
        positive=True # Enforces non-negative coefficients
    )

    all_proportions = []

    # Deconvolve each sample
    for i in range(y_bulk.shape[1]):
        sample_expression = y_bulk[:, i]
        model.fit(X, sample_expression)

        # Get coefficients (proportions) and ensure non-negativity
        proportions = model.coef_
        proportions[proportions < 0] = 0 # Redundant due to positive=True, but safe

        # Normalize proportions to sum to 1
        total = np.sum(proportions)
        if total > 0:
            proportions /= total
        else:
            # If all proportions are zero, assign equal weight to all cell types
            num_cell_types = len(signature_matrix_df.columns)
            proportions = np.full(num_cell_types, 1.0 / num_cell_types)


        all_proportions.append(proportions)

    # Create the results DataFrame
    proportions_df = pd.DataFrame(
        data=np.array(all_proportions).T,
        index=signature_matrix_df.columns,
        columns=bulk_samples_df.columns
    )

    return proportions_df 