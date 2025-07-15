import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    A module to create sinusoidal position embeddings for the timestep.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalDenoisingMLP(nn.Module):
    """
    A conditional MLP to predict noise in a diffusion model setting.
    It takes a noisy vector, a timestep, and a condition vector as input.
    Now includes options for BatchNorm and Dropout for better stability and regularization.
    """
    def __init__(self, proportion_dim, bulk_expr_dim, time_emb_dim=32, hidden_dims=[256, 512, 256], use_batchnorm=False, dropout_rate=0.0):
        super().__init__()

        # --- Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # --- Input and Output Dimensions ---
        self.proportion_dim = proportion_dim
        self.bulk_expr_dim = bulk_expr_dim
        
        input_dim = proportion_dim + bulk_expr_dim + time_emb_dim

        # --- Main Denoising Network (Dynamically created) ---
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Mish())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        # Final layer to output the predicted noise
        layers.append(nn.Linear(current_dim, proportion_dim))
        
        self.denoising_net = nn.Sequential(*layers)

    def forward(self, noisy_proportions, time, bulk_expression):
        """
        Forward pass for the denoising network.

        Args:
            noisy_proportions (torch.Tensor): The noisy proportion vector (batch_size, proportion_dim)
            time (torch.Tensor): The timestep (batch_size,)
            bulk_expression (torch.Tensor): The conditioning bulk expression vector (batch_size, bulk_expr_dim)

        Returns:
            torch.Tensor: The predicted noise (batch_size, proportion_dim)
        """
        # 1. Process time embedding
        time_embedding = self.time_mlp(time)

        # 2. Concatenate all inputs
        x = torch.cat([noisy_proportions, time_embedding, bulk_expression], dim=-1)

        # 3. Pass through the denoising network
        predicted_noise = self.denoising_net(x)

        return predicted_noise

class ConditionalDenoisingTransformer(nn.Module):
    """
    A Transformer-based model to predict noise in a diffusion model setting.
    It uses self-attention to model the relationships between proportions, time, and condition.
    """
    def __init__(self, proportion_dim, bulk_expr_dim, model_dim=128, nhead=4, num_encoder_layers=3, dim_feedforward=256):
        super().__init__()
        self.model_dim = model_dim

        # --- Input Projection Layers ---
        # Project each input modality to the model's dimension
        self.proportion_proj = nn.Linear(proportion_dim, model_dim)
        self.bulk_expr_proj = nn.Linear(bulk_expr_dim, model_dim)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim),
            nn.Linear(model_dim, model_dim),
        )

        # --- Core Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1, 
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- Output Layer ---
        # Projects the output of the first token (representing proportions) back to the original proportion dimension
        self.output_layer = nn.Linear(model_dim, proportion_dim)

    def forward(self, noisy_proportions, time, bulk_expression):
        # 1. Project inputs to the model dimension
        prop_emb = self.proportion_proj(noisy_proportions)
        time_emb = self.time_mlp(time)
        bulk_emb = self.bulk_expr_proj(bulk_expression)

        # 2. Assemble sequence for the transformer
        # The inputs are treated as a sequence of 3 tokens: [proportions, time, bulk_expression]
        # We add an extra dimension for the sequence length.
        full_sequence = torch.stack([prop_emb, time_emb, bulk_emb], dim=1)
        
        # 3. Pass through the Transformer
        transformer_output = self.transformer_encoder(full_sequence)
        
        # 4. We only need the output corresponding to the proportion token (the first one)
        proportion_output_embedding = transformer_output[:, 0, :]

        # 5. Project back to the original dimension to get the predicted noise
        predicted_noise = self.output_layer(proportion_output_embedding)

        return predicted_noise

if __name__ == '__main__':
    # --- Test the network with some dummy data ---
    print("Testing available networks...")
    
    # Parameters from our dataset
    BATCH_SIZE = 4
    N_CELL_TYPES = 8
    N_GENES = 1838

    # Create dummy input tensors
    dummy_proportions = torch.randn(BATCH_SIZE, N_CELL_TYPES)
    dummy_bulk_expr = torch.randn(BATCH_SIZE, N_GENES)
    dummy_time = torch.randint(0, 1000, (BATCH_SIZE,)).long()

    # --- Test MLP ---
    print("\n--- Testing MLP ---")
    mlp_model = ConditionalDenoisingMLP(
        proportion_dim=N_CELL_TYPES,
        bulk_expr_dim=N_GENES,
        hidden_dims=[512, 1024, 512]
    )
    print(f"MLP instantiated. Params: {sum(p.numel() for p in mlp_model.parameters())}")
    mlp_output = mlp_model(dummy_proportions, dummy_time, dummy_bulk_expr)
    print("MLP Forward pass successful. Output shape:", mlp_output.shape)
    assert mlp_output.shape == dummy_proportions.shape
    print("Test passed.")

    # --- Test Transformer ---
    print("\n--- Testing Transformer ---")
    transformer_model = ConditionalDenoisingTransformer(
        proportion_dim=N_CELL_TYPES,
        bulk_expr_dim=N_GENES
    )
    print(f"Transformer instantiated. Params: {sum(p.numel() for p in transformer_model.parameters())}")
    transformer_output = transformer_model(dummy_proportions, dummy_time, dummy_bulk_expr)
    print("Transformer Forward pass successful. Output shape:", transformer_output.shape)
    assert transformer_output.shape == dummy_proportions.shape
    print("Test passed.") 