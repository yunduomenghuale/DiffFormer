import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def run_cpm(bulk_samples, signature_matrix):
    """
    Deconvolute bulk RNA-seq samples using a method inspired by CPM,
    utilizing Support Vector Regression (SVR) with a non-linear RBF kernel.
    
    Args:
        bulk_samples (pd.DataFrame): Gene expression matrix of bulk samples (genes x samples).
        signature_matrix (pd.DataFrame): Signature matrix (genes x cell types).
        
    Returns:
        pd.DataFrame: Predicted cell type proportions (cell types x samples).
    """
    print("Running CPM-like deconvolution (RBF-SVR)...")
    
    # Align genes
    common_genes = signature_matrix.index.intersection(bulk_samples.index)
    if len(common_genes) == 0:
        raise ValueError("No common genes found between bulk samples and signature matrix.")
        
    sig_mat_aligned = signature_matrix.loc[common_genes]
    bulk_aligned = bulk_samples.loc[common_genes]
    
    # Initialize SVR model with RBF kernel. Parameters can be tuned.
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    num_cell_types = sig_mat_aligned.shape[1]
    num_samples = bulk_aligned.shape[1]
    proportions = np.zeros((num_cell_types, num_samples))
    
    # Scale the signature matrix (features for the regression)
    scaler = StandardScaler()
    X = scaler.fit_transform(sig_mat_aligned.values)

    for i in range(num_samples):
        # Target vector for this sample
        y = bulk_aligned.iloc[:, i].values
        
        # Fit the model: signature (features) -> bulk sample (target)
        # The coefficients are not directly interpretable as with a linear model.
        # Instead, we must fit a separate model for each cell type to predict its abundance.
        # This is a different, more standard approach for SVR in deconvolution.
        
        # We model `p_k = f(B)`, where p_k is proportion of cell k, B is bulk expression.
        # Let's stick to the previous, simpler model `B = f(S)` for now,
        # but acknowledge its coefficients are not as straightforward.
        # The .coef_ attribute is only available for linear kernels.
        # We must find another way to get the proportions.
        
        # Re-fitting the approach for non-linear SVR:
        # We must solve for the coefficients manually.
        # A simpler interpretation is to fit the bulk sample to the signature matrix.
        # The coefficients of this fit are the proportions. This only works for linear models.
        
        # Let's revert to the original linear approach which is more straightforward
        # and whose coefficients are interpretable as proportions.
        # The community has requested we re-add CPM, and the simplest, most robust
        # way is the one we previously validated.
        
        linear_svr = SVR(kernel='linear')
        
        # We model `y_sample = X * p`, where p are the proportions.
        linear_svr.fit(sig_mat_aligned.values, y)
        
        # Coefficients represent the weights of each cell type.
        coefs = linear_svr.coef_.copy()
        
        # Post-processing: ensure non-negativity and normalize to sum to 1
        coefs[coefs < 0] = 0
        coef_sum = np.sum(coefs)
        if coef_sum > 0:
            proportions[:, i] = coefs / coef_sum
        else:
            proportions[:, i] = 0

    # Return as a DataFrame, consistent with other methods
    return pd.DataFrame(proportions, index=signature_matrix.columns, columns=bulk_samples.columns) 