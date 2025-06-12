import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GeneExpressionGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[1024, 512, 256], library_size=10000, 
                 use_batch_norm=True, dropout_rate=0.1, use_attention=True, residual_blocks=2):
        """
        Generator network for translating between gene expression domains
        
        Args:
            input_dim: Dimension of input gene expression data
            output_dim: Dimension of output gene expression data
            hidden_dims: List of hidden layer dimensions
            library_size: Target sum for the generated expression values (default: 10000)
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0 to disable)
            use_attention: Whether to use self-attention mechanism
            residual_blocks: Number of residual blocks to use
        """
        super(GeneExpressionGenerator, self).__init__()
        
        # Store parameters
        self.library_size = library_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Encoder part - compresses to latent representation
        self.encoder = nn.ModuleList()
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim) if use_batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            )
            self.encoder.append(block)
            current_dim = h_dim
        
        # Bottleneck dimension
        self.bottleneck_dim = hidden_dims[-1]
        
        # Residual blocks at bottleneck level for better information flow
        self.residual_blocks = nn.ModuleList()
        for _ in range(residual_blocks):
            self.residual_blocks.append(ResidualBlock(
                self.bottleneck_dim, 
                use_batch_norm=use_batch_norm, 
                dropout_rate=dropout_rate
            ))
        
        # Self-attention mechanism at the bottleneck
        if use_attention:
            self.attention = SelfAttention(self.bottleneck_dim)
        
        # Decoder part - reconstructs from latent to target domain
        self.decoder = nn.ModuleList()
        current_dim = self.bottleneck_dim
        
        # Reverse the hidden dimensions for decoder
        for h_dim in reversed(hidden_dims[:-1]):
            block = nn.Sequential(
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            )
            self.decoder.append(block)
            current_dim = h_dim
        
        # Output projection with additional flexibility
        self.output_layer = nn.Sequential(
            nn.Linear(current_dim, current_dim),  # Extra layer for flexibility
            nn.ReLU(),
            nn.Linear(current_dim, output_dim)
        )


class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow"""
    def __init__(self, dim, use_batch_norm=True, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity()
        )
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class SelfAttention(nn.Module):
    """Self-attention mechanism to capture global dependencies"""
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        self.scale = torch.sqrt(torch.tensor(dim, dtype=torch.float32))
        
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(dim, dim)
    
    def forward(self, x):
        # Input is batch_size x feature_dim
        # To apply attention, we treat each feature as a "position"
        # Reshape to batch_size x 1 x feature_dim
        x_reshaped = x.unsqueeze(1)
        
        # Compute query, key, value
        q = self.query(x_reshaped)  # batch_size x 1 x dim
        k = self.key(x_reshaped)    # batch_size x 1 x dim
        v = self.value(x_reshaped)  # batch_size x 1 x dim
        
        # Transpose key for dot product
        k_t = k.transpose(1, 2)  # batch_size x dim x 1
        
        # Compute attention scores and apply scale
        scores = torch.bmm(q, k_t) / self.scale  # batch_size x 1 x 1
        
        # Apply softmax to get attention weights
        attn_weights = self.softmax(scores)  # batch_size x 1 x 1
        
        # Apply attention weights to values
        context = torch.bmm(attn_weights, v)  # batch_size x 1 x dim
        
        # Reshape and project back
        context = context.squeeze(1)  # batch_size x dim
        output = self.output(context)  # batch_size x dim
        
        # Residual connection
        return x + output

    def forward(self, x):
        """
        Forward pass through the generator
        
        Args:
            x: Input gene expression data
            
        Returns:
            Normalized gene expression data in the target domain
        """
        # Pass through encoder blocks
        features = x
        for encoder_block in self.encoder:
            features = encoder_block(features)
        
        # Pass through residual blocks
        for res_block in self.residual_blocks:
            features = res_block(features)
        
        # Apply self-attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Pass through decoder blocks
        for decoder_block in self.decoder:
            features = decoder_block(features)
        
        # Final output projection
        output = self.output_layer(features)
        
        # Normalize to library size
        normalized_output = self.normalize_to_library_size(output)
        
        return normalized_output
    
    def normalize_to_library_size(self, x):
        """
        Normalize the output tensor so the sum equals the target library size
        while ensuring all values are non-negative
        """
        # Apply ReLU to ensure non-negative values
        x = torch.relu(x)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Calculate current sum for each sample
        current_sums = x.sum(dim=1, keepdim=True) + epsilon
        
        # Scale to target library size
        normalized = x * (self.library_size / current_sums)
        
        return normalized
    
    def get_negative_penalty(self, generated_data):
        """Calculate penalty for negative values"""
        negative_mask = (generated_data < 0).float()
        negative_proportion = negative_mask.mean()
        negative_magnitude = (generated_data * negative_mask).abs().mean()
        return negative_magnitude, negative_proportion


class GeneExpressionDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], use_spectral_norm=True, 
                 leaky_relu_slope=0.2, dropout_rate=0.3, use_neg_detector=False):
        """
        Discriminator network for gene expression domain
        
        Args:
            input_dim: Dimension of input gene expression data
            hidden_dims: List of hidden layer dimensions
            use_spectral_norm: Whether to use spectral normalization
            leaky_relu_slope: Slope for LeakyReLU activation
            dropout_rate: Dropout rate
            use_neg_detector: Whether to use negative value detection
        """
        super(GeneExpressionDiscriminator, self).__init__()
        
        # Store use_neg_detector flag
        self.use_neg_detector = use_neg_detector
        
        # Build the main network
        layers = []
        current_dim = input_dim
        
        # Add hidden layers
        for h_dim in hidden_dims:
            if use_spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Linear(current_dim, h_dim)))
            else:
                layers.append(nn.Linear(current_dim, h_dim))
            
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            current_dim = h_dim
        
        # Output layer
        if use_spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(current_dim, 1)))
        else:
            layers.append(nn.Linear(current_dim, 1))
        
        # Combine all layers
        self.main_network = nn.Sequential(*layers)
        
        # Add negative value detection branch if enabled
        if use_neg_detector:
            self.negative_detector = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Linear(hidden_dims[0], 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        """
        Forward pass through the discriminator
        
        Args:
            x: Input gene expression data
            
        Returns:
            Discrimination score
        """
        # Main discrimination score
        validity = self.main_network(x)
        
        # Add negative value detection if enabled
        if self.use_neg_detector:
            neg_score = self.negative_detector(torch.relu(-x))  # Only pass negative values
            return validity - 0.1 * neg_score  # Penalize negative values
        
        return validity

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Calculate gradient penalty for WGAN-GP
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Calculate discriminator output for interpolated samples
    d_interpolates = discriminator(interpolates)
    
    # Create fake gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty