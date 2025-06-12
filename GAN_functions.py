import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import os
import json
from datetime import datetime
from tqdm import tqdm
import scipy.sparse
import scanpy as sc
from scanpy.experimental.pp import normalize_pearson_residuals
import umap

# Optional wandb integration
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

# Self-Attention Mechanism
class GeneSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(GeneSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)

# Sparse Self-Attention Mechanism
class LearnedSparseSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity_target=0.9, temperature=0.5, dropout_rate=0.1):
        super(LearnedSparseSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.gate = nn.Linear(embed_dim, 1)  # Output a scalar per token
        self.norm = nn.LayerNorm(embed_dim)
        self.sparsity_target = sparsity_target
        self.temperature = temperature
    
    def forward(self, x):
        gates = torch.sigmoid(self.gate(x) / self.temperature)  # (batch_size, seq_len, 1)
        attn_output, attn_weights = self.attention(x, x, x)  # attn_weights: (batch_size, seq_len, seq_len)
        sparse_weights = attn_weights * gates  # Broadcasting: (batch_size, seq_len, seq_len) * (batch_size, seq_len, 1)
        return self.norm(x + attn_output)

# Residual Block with BatchNorm for 3D input
class ResidualBlock(nn.Module):
    def __init__(self, embed_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        x_reshaped = x.view(batch_size * seq_len, embed_dim)
        residual = self.block(x_reshaped)
        residual = residual.view(batch_size, seq_len, embed_dim)
        return x + residual

# Generator with VAE-like Latent Space and BatchNorm
class GeneExpressionGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, vocab_sizes, config):
        super(GeneExpressionGenerator, self).__init__()
        self.library_size = config['library_size']
        self.use_attention = config['use_self_attention']
        self.preprocess_type = config['preprocess_type']
        self.latent_dim = config['latent_dim']
        self.seq_len = 16  # Define sequence length
        self.embed_dim = self.latent_dim // self.seq_len
        assert self.embed_dim * self.seq_len == self.latent_dim, "latent_dim must be divisible by seq_len"

        self.embeddings = nn.ModuleList([nn.Embedding(vs, min(100, vs)) for vs in vocab_sizes])
        self.embedding_dim = sum(min(100, vs) for vs in vocab_sizes)

        # Encoder: First layer matches input_dim
        encoder_layers = [nn.Linear(input_dim, config['hidden_dims'][0]), nn.BatchNorm1d(config['hidden_dims'][0]), nn.ReLU(inplace=True)]
        in_dim = config['hidden_dims'][0]
        for h_dim in config['hidden_dims'][1:]:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(config['hidden_dims'][-1], self.latent_dim)
        self.log_var = nn.Linear(config['hidden_dims'][-1], self.latent_dim)

        attention_class = LearnedSparseSelfAttention if config['use_sparse_attention'] else GeneSelfAttention
        self.residual_attention_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(self.embed_dim),
                attention_class(
                    self.embed_dim,
                    config['num_attention_heads'],
                    sparsity_target=config['sparsity_target'],
                    temperature=config['temperature']
                ) if config['use_sparse_attention'] else GeneSelfAttention(
                    self.embed_dim,
                    config['num_attention_heads']
                )
            ) for _ in range(config['residual_blocks'])
        ])

        # Decoder: Last layer matches output_dim
        decoder_layers = [nn.Linear(self.latent_dim + self.embedding_dim, config['hidden_dims'][0]), nn.BatchNorm1d(config['hidden_dims'][0]), nn.ReLU(inplace=True)]
        in_dim = config['hidden_dims'][0]
        for h_dim in config['hidden_dims'][1:]:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, output_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cat_covs=None):
        if cat_covs is not None:
            cat_covs = cat_covs.to(x.device)
            embeddings = torch.cat([emb(cat_covs[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        else:
            embeddings = torch.zeros(x.size(0), self.embedding_dim, device=x.device)
        features = self.encoder(x)
        mu, log_var = self.mu(features), self.log_var(features)
        z = self.reparameterize(mu, log_var)  # (batch_size, latent_dim)
        batch_size = z.size(0)
        z = z.view(batch_size, self.seq_len, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        for block in self.residual_attention_blocks:
            z = block(z)
        z = z.view(batch_size, self.latent_dim)  # (batch_size, latent_dim)
        z = torch.cat([z, embeddings], dim=1)
        output = self.decoder(z)
        if self.preprocess_type == 'counts':
            output = self.normalize_to_library_size(output)
        return output, mu, log_var

    def normalize_to_library_size(self, x):
        x = torch.relu(x)
        current_sums = x.sum(dim=1, keepdim=True) + 1e-10
        return x * (self.library_size / current_sums)

# Discriminator with BatchNorm
class GeneExpressionDiscriminator(nn.Module):
    def __init__(self, input_dim, vocab_sizes, config):
        super(GeneExpressionDiscriminator, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vs, min(100, vs)) for vs in vocab_sizes])
        self.embedding_dim = sum(min(100, vs) for vs in vocab_sizes)
        
        layers = [
            nn.Linear(input_dim + self.embedding_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, cat_covs=None, return_features=False):
        if cat_covs is not None:
            cat_covs = cat_covs.to(x.device)
            embeddings = torch.cat([emb(cat_covs[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        else:
            embeddings = torch.zeros(x.size(0), self.embedding_dim, device=x.device)
        x = torch.cat([x, embeddings], dim=1)
        features = x
        for layer in self.model[:-1]:
            features = layer(features)
        output = self.model[-1](features)
        if return_features:
            return output, features
        return output

# Gradient Penalty
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device, cat_covs=None):
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates, _ = discriminator(interpolates, cat_covs, return_features=True)
    fake = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Compute Centroids
def compute_cell_type_centroids(adata, tensor, device):
    centroids = {}
    for ct in adata.obs['myannotations'].unique():
        mask = adata.obs['myannotations'] == ct
        if mask.sum() > 0:
            centroids[ct] = tensor[mask].mean(dim=0).clone().detach().to(device)
    return centroids

# Poisson Loss
def poisson_loss(pred, target):
    return torch.mean(pred - target * torch.log(pred + 1e-10))

# Training Function with Test Figures
def train_enhanced_cycle_gan(g_m2h, g_h2m, d_m, d_h, mouse_adata, human_adata, 
                            mouse_tensor, human_tensor, cat_tensor, human_umap_coords, umap_model, 
                            idx_to_cell_type, config, device, pca_model=None, scaler=None):
    g_optimizer = optim.Adam(list(g_m2h.parameters()) + list(g_h2m.parameters()), lr=config['lr']*1.5, betas=(0.5, 0.9), amsgrad=True)
    d_m_optimizer = optim.Adam(d_m.parameters(), lr=config['lr'], betas=(0.5, 0.9), amsgrad=True)
    d_h_optimizer = optim.Adam(d_h.parameters(), lr=config['lr'], betas=(0.5, 0.9), amsgrad=True)
    
    class LRDecayScheduler:
        def __init__(self, optimizers, initial_lr, decay_start_epoch, total_epochs, decay_factor, decay_type):
            self.optimizers = optimizers
            self.initial_lr = initial_lr
            self.decay_start_epoch = decay_start_epoch
            self.total_epochs = total_epochs
            self.decay_factor = decay_factor
            self.decay_type = decay_type
        
        def step(self, epoch):
            if epoch < self.decay_start_epoch:
                return self.initial_lr
            progress = (epoch - self.decay_start_epoch) / (self.total_epochs - self.decay_start_epoch)
            if self.decay_type == 'cosine':
                new_lr = self.initial_lr * (1 + np.cos(np.pi * progress)) / 2 * self.decay_factor
            else:
                new_lr = self.initial_lr * (1 - progress * self.decay_factor)
            for opt in self.optimizers:
                for pg in opt.param_groups:
                    pg['lr'] = new_lr
            return new_lr
    
    lr_scheduler = LRDecayScheduler(
        [g_optimizer, d_m_optimizer, d_h_optimizer], config['lr'], 
        config['decay_start_epoch'], config['epochs'], config['decay_factor'], config['decay_type']
    ) if config['use_lr_decay'] else None
    
    n_mouse = mouse_tensor.size(0)
    mouse_cat_tensor = torch.tensor(cat_tensor[:n_mouse], dtype=torch.long).to(device)
    human_cat_tensor = torch.tensor(cat_tensor[n_mouse:], dtype=torch.long).to(device)
    
    mouse_dataset = TensorDataset(mouse_tensor, mouse_cat_tensor)
    human_dataset = TensorDataset(human_tensor, human_cat_tensor)
    
    mouse_loader = DataLoader(mouse_dataset, batch_size=config['batch_size'], shuffle=True)
    human_loader = DataLoader(human_dataset, batch_size=config['batch_size'], shuffle=True)
    
    mouse_centroids = compute_cell_type_centroids(mouse_adata, mouse_tensor, device)
    human_centroids = compute_cell_type_centroids(human_adata, human_tensor, device)
    
    metrics = {
        'g_loss': [], 'd_m_loss': [], 'd_h_loss': [], 'fake_human_var': [], 'fake_mouse_var': [],
        'cycle_loss': [], 'celltype_loss': [], 'fm_loss': [], 'kl_loss': [], 'current_lr': []
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"results/enhanced_cyclegan_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    if config['use_wandb'] and has_wandb:
        wandb.init(project="Grok_gene_cycleGAN_lr", config=config, name=f"run_{timestamp}")
    
    # Initialize moving averages for dynamic scaling
    moving_avg_adv = 0
    moving_avg_cycle = 0
    moving_avg_celltype = 0
    moving_avg_fm = 0
    moving_avg_kl = 0
    alpha = 0.3  # Smoothing factor for moving average
    
    for epoch in tqdm(range(config['epochs']), desc="Training"):
        g_losses, d_m_losses, d_h_losses = [], [], []
        cycle_losses, celltype_losses, fm_losses, kl_losses = [], [], [], []
        fake_human_vars, fake_mouse_vars = [], []

        mouse_iter = iter(mouse_loader)
        human_iter = iter(human_loader)

        for i in range(max(len(mouse_loader), len(human_loader))):
            try:
                mouse_data, mouse_cat = next(mouse_iter)
            except StopIteration:
                mouse_iter = iter(mouse_loader)
                mouse_data, mouse_cat = next(mouse_iter)

            try:
                human_data, human_cat = next(human_iter)
            except StopIteration:
                human_iter = iter(human_loader)
                human_data, human_cat = next(human_iter)

            mouse_data, human_data = mouse_data.to(device), human_data.to(device)
            mouse_cat, human_cat = mouse_cat.to(device), human_cat.to(device)

            # Synchronize batch sizes by taking the minimum
            min_batch_size = min(mouse_data.size(0), human_data.size(0))
            mouse_data = mouse_data[:min_batch_size]
            human_data = human_data[:min_batch_size]
            mouse_cat = mouse_cat[:min_batch_size]
            human_cat = human_cat[:min_batch_size]

            # Train Discriminators
            d_m_optimizer.zero_grad()
            d_h_optimizer.zero_grad()
            fake_human, _, _ = g_m2h(mouse_data, mouse_cat)
            fake_mouse, _, _ = g_h2m(human_data, human_cat)

            real_m_validity, real_m_features = d_m(mouse_data, mouse_cat, return_features=True)
            fake_m_validity, fake_m_features = d_m(fake_mouse.detach(), human_cat, return_features=True)
            gp_m = compute_gradient_penalty(d_m, mouse_data, fake_mouse, device, human_cat)
            d_m_loss = -torch.mean(real_m_validity) + torch.mean(fake_m_validity) + config['lambda_gp'] * gp_m

            real_h_validity, real_h_features = d_h(human_data, human_cat, return_features=True)
            fake_h_validity, fake_h_features = d_h(fake_human.detach(), mouse_cat, return_features=True)
            gp_h = compute_gradient_penalty(d_h, human_data, fake_human, device, mouse_cat)
            d_h_loss = -torch.mean(real_h_validity) + torch.mean(fake_h_validity) + config['lambda_gp'] * gp_h

            d_m_loss.backward()
            d_h_loss.backward()
            d_m_optimizer.step()
            d_h_optimizer.step()

            if i % config['nb_critic'] == 0:
                g_optimizer.zero_grad()
                fake_human, mu_h, log_var_h = g_m2h(mouse_data, mouse_cat)
                fake_mouse, mu_m, log_var_m = g_h2m(human_data, human_cat)

                g_adv = -torch.mean(d_h(fake_human, human_cat)) - torch.mean(d_m(fake_mouse, mouse_cat))

                recov_mouse, _, _ = g_h2m(fake_human, human_cat)
                recov_human, _, _ = g_m2h(fake_mouse, mouse_cat)
                #g_cycle = poisson_loss(recov_mouse, mouse_data) + poisson_loss(recov_human, human_data)
                # Forward cycle consistency: ||F(G(x)) - x||1
                g_cycle_m = torch.mean(torch.abs(recov_mouse - mouse_data))
                # Backward cycle consistency: ||G(F(y)) - y||1
                g_cycle_h = torch.mean(torch.abs(recov_human - human_data))
                # Combined cycle consistency
                g_cycle = g_cycle_m + g_cycle_h

                g_celltype = torch.tensor(0.0, device=device)
                if mouse_cat.size(0) > 0 and human_cat.size(0) > 0:
                    mouse_ct_idx = [idx_to_cell_type[idx.item()] for idx in mouse_cat]
                    human_ct_idx = [idx_to_cell_type[idx.item()] for idx in human_cat]
                    g_celltype += F.mse_loss(fake_human, torch.stack([human_centroids[ct] for ct in mouse_ct_idx]))
                    g_celltype += F.mse_loss(fake_mouse, torch.stack([mouse_centroids[ct] for ct in human_ct_idx]))

                _, real_h_fm = d_h(human_data, human_cat, return_features=True)
                _, fake_h_fm = d_h(fake_human, human_cat, return_features=True)
                fm_loss_h = F.mse_loss(real_h_fm.mean(dim=0), fake_h_fm.mean(dim=0))

                _, real_m_fm = d_m(mouse_data, mouse_cat, return_features=True)
                _, fake_m_fm = d_m(fake_mouse, mouse_cat, return_features=True)
                fm_loss_m = F.mse_loss(real_m_fm.mean(dim=0), fake_m_fm.mean(dim=0))
                g_fm = (fm_loss_h + fm_loss_m) / 2

                kl_loss = -0.5 * torch.mean(1 + log_var_h - mu_h.pow(2) - log_var_h.exp()) - \
                          0.5 * torch.mean(1 + log_var_m - mu_m.pow(2) - log_var_m.exp())
                          
                ### Scale loss
                # Update moving averages with current loss values
                moving_avg_adv = alpha * g_adv.item() + (1 - alpha) * moving_avg_adv
                moving_avg_cycle = alpha * g_cycle.item() + (1 - alpha) * moving_avg_cycle
                moving_avg_celltype = alpha * g_celltype.item() + (1 - alpha) * moving_avg_celltype
                moving_avg_fm = alpha * g_fm.item() + (1 - alpha) * moving_avg_fm
                moving_avg_kl = alpha * kl_loss.item() + (1 - alpha) * moving_avg_kl

                # Compute dynamic scaling factors based on inverse magnitudes
                scale_adv = 1 / (abs(moving_avg_adv) + 1e-8)  # Avoid division by zero
                scale_cycle = 1 / (moving_avg_cycle + 1e-8)
                scale_celltype = 1 / (moving_avg_celltype + 1e-8)
                scale_fm = 1 / (moving_avg_fm + 1e-8)
                scale_kl = 1 / (moving_avg_kl + 1e-8)

                # Normalize scales so the smallest scale is 1
                min_scale = min(scale_adv, scale_cycle, scale_celltype, scale_fm, scale_kl)
                scale_adv /= min_scale
                scale_cycle /= min_scale
                scale_celltype /= min_scale
                scale_fm /= min_scale
                scale_kl /= min_scale

                # Calculate final generator loss with initial lambdas and dynamic scales
                g_loss = (config['lambda_adv'] * scale_adv * g_adv +
                    config['lambda_cycle'] * scale_cycle * g_cycle +
                    config['lambda_celltype'] * scale_celltype * g_celltype +
                    config['lambda_fm'] * scale_fm * g_fm +
                    config['lambda_kl'] * scale_kl * kl_loss)

                g_loss.backward()
                g_optimizer.step()
                
                g_losses.append(g_loss.item())
                d_m_losses.append(d_m_loss.item())
                d_h_losses.append(d_h_loss.item())
                cycle_losses.append(g_cycle.item())
                celltype_losses.append(g_celltype.item())
                fm_losses.append(g_fm.item())
                kl_losses.append(kl_loss.item())
                fake_human_vars.append(torch.var(fake_human, dim=0).mean().item())
                fake_mouse_vars.append(torch.var(fake_mouse, dim=0).mean().item())
        
        current_lr = lr_scheduler.step(epoch) if lr_scheduler else config['lr']
        
        metrics['g_loss'].append(np.mean(g_losses))
        metrics['d_m_loss'].append(np.mean(d_m_losses))
        metrics['d_h_loss'].append(np.mean(d_h_losses))
        metrics['cycle_loss'].append(np.mean(cycle_losses))
        metrics['celltype_loss'].append(np.mean(celltype_losses))
        metrics['fm_loss'].append(np.mean(fm_losses))
        metrics['kl_loss'].append(np.mean(kl_losses))
        metrics['fake_human_var'].append(np.mean(fake_human_vars))
        metrics['fake_mouse_var'].append(np.mean(fake_mouse_vars))
        metrics['current_lr'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{config['epochs']}: G Loss: {metrics['g_loss'][-1]:.4f}, "
              f"D_M Loss: {metrics['d_m_loss'][-1]:.4f}, D_H Loss: {metrics['d_h_loss'][-1]:.4f}, "
              f"KL Loss: {metrics['kl_loss'][-1]:.4f}, FM Loss: {metrics['fm_loss'][-1]:.4f}, Cycle Loss: {metrics['cycle_loss'][-1]:.4f}, "
              f"celltype Loss: {metrics['celltype_loss'][-1]:.4f}")
              
        if config['use_wandb'] and has_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'G Loss': metrics['g_loss'][-1],
                'D_M Loss': metrics['d_m_loss'][-1],
                'D_H Loss': metrics['d_h_loss'][-1],
                'KL Loss': metrics['kl_loss'][-1],
                'FM Loss': metrics['fm_loss'][-1],
                'Cycle Loss': metrics['cycle_loss'][-1],
                'Celltype Loss': metrics['celltype_loss'][-1],
                'Fake Human Variance': metrics['fake_human_var'][-1],
                'Fake Mouse Variance': metrics['fake_mouse_var'][-1],
                'Scaledadv': config['lambda_adv'] * scale_adv * g_adv,
                'Scaled cycle': config['lambda_cycle'] * scale_cycle * g_cycle,
                'Scaled celltype': config['lambda_celltype'] * scale_celltype * g_celltype,
                'Scaled fm': config['lambda_fm'] * scale_fm * g_fm,
                'Scaled kl': config['lambda_kl'] * scale_kl * kl_loss,
                'Current LR': metrics['current_lr']
            })

        
        if (epoch + 1) % 10 == 0 or epoch == config['epochs'] - 1:
            g_m2h.eval()
            with torch.no_grad():
                subsample_size = min(3000, mouse_tensor.size(0))
                subsample_indices = np.random.choice(mouse_tensor.size(0), subsample_size, replace=False)
                mouse_subsample = mouse_tensor[subsample_indices].to(device)
                cat_covs_subsample = mouse_cat_tensor[subsample_indices].to(device)
                
                fake_human_subsample, _, _ = g_m2h(mouse_subsample, cat_covs_subsample)
                fake_human_np = fake_human_subsample.cpu().numpy()
                
                if human_umap_coords is not None and pca_model is not None and scaler is not None:
                    # Scale the translated data using the provided scaler
                    fake_human_scaled = scaler.transform(fake_human_np)
                    fake_human_pca = pca_model.transform(fake_human_scaled)
                    fake_human_umap_coords = umap_model.transform(fake_human_pca)
                    subsample_cell_types = [idx_to_cell_type[idx.item()] for idx in cat_covs_subsample]
                    
                    # Plot UMAP
                    plt.figure(figsize=(10, 8))
                    # Define color_dict for consistent coloring
                    unique_cell_types = list(set(human_adata.obs['myannotations']) | set(subsample_cell_types))
                    color_dict = {ct: plt.cm.tab20(i) for i, ct in enumerate(unique_cell_types)}
                    # Real human data
                    for ct in set(human_adata.obs['myannotations']):
                        mask = human_adata.obs['myannotations'] == ct
                        sns.scatterplot(
                            x=human_umap_coords[mask, 0], 
                            y=human_umap_coords[mask, 1], 
                            color=color_dict[ct], 
                            label=f"Real {ct}", 
                            s=10, 
                            alpha=0.5
                        )
                    # Translated mouse data
                    for ct in set(subsample_cell_types):
                        mask = [ct == sct for sct in subsample_cell_types]
                        sns.scatterplot(
                            x=fake_human_umap_coords[mask, 0], 
                            y=fake_human_umap_coords[mask, 1], 
                            color=color_dict[ct], 
                            label=f"Translated {ct}", 
                            marker='^', 
                            s=50, 
                            alpha=0.8
                        )
                    
                    plt.title(f"UMAP: Real Human vs Translated Mouse (Epoch {epoch+1})")
                    plt.xlim(human_umap_coords[:, 0].min() - 1, human_umap_coords[:, 0].max() + 1)
                    plt.ylim(human_umap_coords[:, 1].min() - 1, human_umap_coords[:, 1].max() + 1)
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                    plt.savefig(os.path.join(output_dir, f"umap_epoch_{epoch+1}.png"), bbox_inches='tight')
                    plt.close()
                
                translated_centroids = {}
                for ct in set(subsample_cell_types):
                    mask = [ct == sct for sct in subsample_cell_types]
                    if sum(mask) > 0:
                        translated_centroids[ct] = torch.tensor(scaler.transform(fake_human_np[mask])).mean(dim=0).cpu().numpy()
                
                real_human_centroids_np = {ct: scaler.transform(centroid.cpu().numpy().reshape(1, -1))[0] 
                                         for ct, centroid in human_centroids.items()}
                common_cell_types = set(real_human_centroids_np.keys()) & set(translated_centroids.keys())
                correlations = []
                for gene_idx in range(fake_human_np.shape[1]):
                    real_vals = [real_human_centroids_np[ct][gene_idx] for ct in common_cell_types]
                    trans_vals = [translated_centroids[ct][gene_idx] for ct in common_cell_types]
                    if len(real_vals) > 1:
                        corr = np.corrcoef(real_vals, trans_vals)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                plt.figure(figsize=(8, 6))
                sns.histplot(correlations, bins=50, kde=True)
                plt.title(f"Centroid Correlation Histogram (Epoch {epoch+1}) - Scaled")
                plt.xlabel("Pearson Correlation Coefficient")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(output_dir, f"centroid_correlations_epoch_{epoch+1}_scaled.png"))
                plt.close()
            
            g_m2h.train()
        
        if (epoch + 1) % 20 == 0 or epoch == config['epochs'] - 1:
            torch.save(g_m2h.state_dict(), os.path.join(output_dir, f'g_m2h_epoch_{epoch+1}.pt'))
            torch.save(g_h2m.state_dict(), os.path.join(output_dir, f'g_h2m_epoch_{epoch+1}.pt'))
            torch.save(d_m.state_dict(), os.path.join(output_dir, f'd_m_epoch_{epoch+1}.pt'))
            torch.save(d_h.state_dict(), os.path.join(output_dir, f'd_h_epoch_{epoch+1}.pt'))
    
    if config['use_wandb'] and has_wandb:
        wandb.finish()
    return metrics, output_dir

# Plotting Metrics
def plot_metrics(metrics, output_dir):
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    axs[0, 0].plot(metrics['g_loss'], label='Generator Loss')
    axs[0, 0].plot(metrics['d_m_loss'], label='Mouse Disc Loss')
    axs[0, 0].plot(metrics['d_h_loss'], label='Human Disc Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title('Losses')
    
    axs[0, 1].plot(metrics['cycle_loss'], label='Cycle Loss')
    axs[0, 1].plot(metrics['celltype_loss'], label='Celltype Loss')
    axs[0, 1].legend()
    axs[0, 1].set_title('Component Losses')
    
    axs[1, 0].plot(metrics['fake_human_var'], label='Fake Human Variance')
    axs[1, 0].plot(metrics['fake_mouse_var'], label='Fake Mouse Variance')
    axs[1, 0].legend()
    axs[1, 0].set_title('Output Variance')
    
    axs[1, 1].plot(metrics['fm_loss'], label='Feature Matching Loss')
    axs[1, 1].plot(metrics['kl_loss'], label='KL Divergence Loss')
    axs[1, 1].legend()
    axs[1, 1].set_title('Additional Losses')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

# Preprocessing Function
def preprocess_data(adata, layer_type='counts', library_size=None):
    """
    Preprocess AnnData object to extract gene expression data from a specified layer.
    
    Parameters:
    - adata: AnnData object containing gene expression data
    - layer_type: str, either 'counts' or 'log_normalized' to specify the layer to use
    - library_size: float or None, target sum for normalization (only applied if layer_type='counts')
    
    Returns:
    - torch.FloatTensor: Processed gene expression data ready for model training
    """
    if layer_type not in ['counts', 'log_normalized']:
        raise ValueError("layer_type must be 'counts' or 'log_normalized'")
    
    # Extract the specified layer
    if layer_type == 'counts':
        data = adata.layers['counts']
        if library_size is not None:
            # Normalize total counts per cell to the specified library size
            sc.pp.normalize_total(adata, target_sum=library_size, layer='counts')
            data = adata.layers['counts']  # Re-extract after normalization
    elif layer_type == 'log_normalized':
        data = adata.layers['log_normalized']
        # No additional normalization applied to log_normalized data
    
    # Convert to dense format if sparse, then to PyTorch tensor
    if scipy.sparse.issparse(data):
        data = data.toarray()
    return torch.tensor(data, dtype=torch.float32)