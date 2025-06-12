import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def rbf_kernel(x, y, sigma=1.0):
    """RBF kernel for MMD calculation"""
    x_expanded = x.unsqueeze(1)
    y_expanded = y.unsqueeze(0)
    dist = torch.sum((x_expanded - y_expanded) ** 2, dim=2)
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd_rbf(x, y, sigma_list=[1, 5, 10]):
    """MMD using multiple RBF kernels with different bandwidths"""
    mmd_value = 0
    for sigma in sigma_list:
        x_kernel = rbf_kernel(x, x, sigma)
        y_kernel = rbf_kernel(y, y, sigma)
        xy_kernel = rbf_kernel(x, y, sigma)
        mmd_value += torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd_value

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculate gradient penalty for WGAN-GP"""
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def biological_consistency_loss(real_data, translated_data, n_pairs=50):
    """Calculate loss based on preserving gene-gene correlations"""
    n_genes = real_data.shape[1]
    gene_indices = torch.randperm(n_genes)[:min(n_pairs*2, n_genes)]
    
    if len(gene_indices) < 4:
        return torch.tensor(0.0, device=real_data.device)
    
    gene_pairs = gene_indices.reshape(-1, 2)
    loss = 0.0
    valid_pairs = 0
    
    for pair in gene_pairs:
        if pair.shape[0] < 2:
            continue
            
        i, j = pair
        
        # Real data correlation
        real_i, real_j = real_data[:, i], real_data[:, j]
        real_i_mean, real_j_mean = real_i.mean(), real_j.mean()
        real_cov = ((real_i - real_i_mean) * (real_j - real_j_mean)).mean()
        real_std_i = torch.sqrt(((real_i - real_i_mean) ** 2).mean() + 1e-8)
        real_std_j = torch.sqrt(((real_j - real_j_mean) ** 2).mean() + 1e-8)
        real_corr = real_cov / (real_std_i * real_std_j + 1e-8)
        
        # Translated data correlation
        trans_i, trans_j = translated_data[:, i], translated_data[:, j]
        trans_i_mean, trans_j_mean = trans_i.mean(), trans_j.mean()
        trans_cov = ((trans_i - trans_i_mean) * (trans_j - trans_j_mean)).mean()
        trans_std_i = torch.sqrt(((trans_i - trans_i_mean) ** 2).mean() + 1e-8)
        trans_std_j = torch.sqrt(((trans_j - trans_j_mean) ** 2).mean() + 1e-8)
        trans_corr = trans_cov / (trans_std_i * trans_std_j + 1e-8)
        
        pair_loss = (real_corr - trans_corr) ** 2
        
        if not torch.isnan(pair_loss) and not torch.isinf(pair_loss):
            loss += pair_loss
            valid_pairs += 1
    
    return loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0, device=real_data.device)

def get_negative_penalty(data, weight):
    """Calculate penalty for negative values in generated data"""
    negative_mask = (data < 0).float()
    negative_magnitude = (data * negative_mask).abs().mean()
    return negative_magnitude * weight, negative_mask.mean()

def train_cycle_gan(g_m2h, g_h2m, d_m, d_h, mouse_loader, human_loader, config, device, save_fn=None):
    """
    Train the cycle-consistent GAN for translating between mouse and human gene expression data
    with options for self-attention and cross-attention mechanisms.
    
    Args:
        g_m2h: Generator for mouse to human translation
        g_h2m: Generator for human to mouse translation
        d_m: Mouse discriminator
        d_h: Human discriminator
        mouse_loader: DataLoader for mouse gene expression data
        human_loader: DataLoader for human gene expression data
        config: Dictionary of training parameters and options
        device: Device to run training on (cuda, mps, or cpu)
        save_fn: Optional function to save models during training
    """
    # Extract config parameters
    lambda_gp = config.get('lambda_gp', 10)
    lambda_cycle = config.get('lambda_cycle', 10)
    lambda_consistency = config.get('lambda_consistency', 2.0)
    lambda_mmd = config.get('lambda_mmd', 5.0)
    neg_penalty_start = config.get('negative_penalty_start_epoch', 50)
    neg_penalty_ramp = config.get('negative_penalty_ramp_epochs', 50)
    max_neg_penalty = config.get('max_negative_penalty', 10.0)
    
    # Attention options
    use_self_attention = config.get('use_self_attention', True)
    use_cross_attention = config.get('use_cross_attention', False)
    
    # Print attention configuration
    print(f"Using self-attention: {use_self_attention}")
    print(f"Using cross-attention: {use_cross_attention}")
    
    # Check if generators are configured for cross-attention
    if use_cross_attention:
        if not hasattr(g_m2h, 'use_cross_attention') or not hasattr(g_h2m, 'use_cross_attention'):
            print("Warning: Cross-attention enabled but generators don't have this capability.")
            print("Make sure to initialize generators with use_cross_attention=True.")
            use_cross_attention = False
        elif not g_m2h.use_cross_attention or not g_h2m.use_cross_attention:
            print("Warning: Cross-attention enabled but at least one generator has it disabled.")
            print("Setting cross-attention to False for consistency.")
            use_cross_attention = False
    
    # Optimizers
    g_optimizer = optim.Adam(list(g_m2h.parameters()) + list(g_h2m.parameters()), 
                            lr=config['lr'], betas=(config.get('beta1', 0.5), config.get('beta2', 0.9)), 
                            amsgrad=True)
    d_m_optimizer = optim.Adam(d_m.parameters(), lr=config['lr'], betas=(0.5, 0.9), amsgrad=True)
    d_h_optimizer = optim.Adam(d_h.parameters(), lr=config['lr'], betas=(0.5, 0.9), amsgrad=True)
    
    # Negative penalty weight function
    def get_negative_penalty_weight(epoch):
        if epoch < neg_penalty_start: return 0.0
        ramp_progress = min(1.0, max(0.0, (epoch - neg_penalty_start) / neg_penalty_ramp))
        return max_neg_penalty * ramp_progress
    
    # Data iterators
    mouse_iter = iter(mouse_loader)
    human_iter = iter(human_loader)
    
    def get_next_batch(loader, iterator):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        return batch, iterator
    
    total_batches = min(len(mouse_loader), len(human_loader))
    print(f"Starting training for {config['epochs']} epochs...")
    
    # Helper function to get bottleneck representations if using cross-attention
    def get_bottleneck_representation(generator, x):
        """Extract bottleneck features from generator for cross-attention"""
        features = x
        
        # Pass through encoder
        for encoder_block in generator.encoder:
            features = encoder_block(features)
        
        # Pass through residual blocks
        for res_block in generator.residual_blocks:
            features = res_block(features)
        
        # Apply self-attention if enabled
        if use_self_attention and generator.use_attention:
            features = generator.attention(features)
            
        return features
    
    # Main training loop
    for epoch in range(config['epochs']):
        # Track metrics
        d_m_losses, d_h_losses, g_losses = [], [], []
        g_adv_losses, g_cycle_losses = [], []
        g_cycle_m_losses, g_cycle_h_losses = [], []
        g_mmd_losses, g_bio_losses = [], []
        neg_penalties = []
        
        curr_neg_weight = get_negative_penalty_weight(epoch)
        progress_bar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx in range(total_batches):
            # Get data batches
            (mouse_data,), mouse_iter = get_next_batch(mouse_loader, mouse_iter)
            (human_data,), human_iter = get_next_batch(human_loader, human_iter)
            
            batch_size = min(mouse_data.size(0), human_data.size(0))
            mouse_data = mouse_data[:batch_size].to(device)
            human_data = human_data[:batch_size].to(device)
            
            # Train Discriminators
            for _ in range(config.get('nb_critic', 5)):
                d_m_optimizer.zero_grad()
                d_h_optimizer.zero_grad()
                
                # Generate fake samples (no cross-attention in discriminator phase)
                with torch.no_grad():
                    fake_human = g_m2h(mouse_data)
                    fake_mouse = g_h2m(human_data)
                
                # Mouse Discriminator
                real_mouse_validity = d_m(mouse_data)
                fake_mouse_validity = d_m(fake_mouse)
                gp_m = compute_gradient_penalty(d_m, mouse_data, fake_mouse, device)
                d_m_loss = -torch.mean(real_mouse_validity) + torch.mean(fake_mouse_validity) + lambda_gp * gp_m
                d_m_loss.backward()
                d_m_optimizer.step()
                
                # Human Discriminator
                real_human_validity = d_h(human_data)
                fake_human_validity = d_h(fake_human)
                gp_h = compute_gradient_penalty(d_h, human_data, fake_human, device)
                d_h_loss = -torch.mean(real_human_validity) + torch.mean(fake_human_validity) + lambda_gp * gp_h
                d_h_loss.backward()
                d_h_optimizer.step()
                
                d_m_losses.append(d_m_loss.item())
                d_h_losses.append(d_h_loss.item())
            
            # Train Generators
            g_optimizer.zero_grad()
            
            # For cross-attention, first get bottleneck representations
            mouse_bottleneck = None
            human_bottleneck = None
            
            if use_cross_attention:
                with torch.no_grad():
                    mouse_bottleneck = get_bottleneck_representation(g_m2h, mouse_data)
                    human_bottleneck = get_bottleneck_representation(g_h2m, human_data)
            
            # Generate fake samples (with cross-attention if enabled)
            if use_cross_attention:
                fake_human = g_m2h(mouse_data, human_bottleneck)
                fake_mouse = g_h2m(human_data, mouse_bottleneck)
            else:
                fake_human = g_m2h(mouse_data)
                fake_mouse = g_h2m(human_data)
            
            # Adversarial loss
            fake_human_validity = d_h(fake_human)
            fake_mouse_validity = d_m(fake_mouse)
            g_adv_h = -torch.mean(fake_human_validity)
            g_adv_m = -torch.mean(fake_mouse_validity)
            g_adv = g_adv_h + g_adv_m
            
            # Cycle consistency losses
            # For cross-attention in cycle path, we need new bottleneck representations
            if use_cross_attention:
                with torch.no_grad():
                    fake_human_bottleneck = get_bottleneck_representation(g_m2h, fake_mouse)
                    fake_mouse_bottleneck = get_bottleneck_representation(g_h2m, fake_human)
                
                recov_mouse = g_h2m(fake_human, fake_mouse_bottleneck)
                recov_human = g_m2h(fake_mouse, fake_human_bottleneck)
            else:
                recov_mouse = g_h2m(fake_human)
                recov_human = g_m2h(fake_mouse)
            
            # Forward cycle consistency: ||F(G(x)) - x||1
            g_cycle_m = torch.mean(torch.abs(recov_mouse - mouse_data))
            
            # Backward cycle consistency: ||G(F(y)) - y||1
            g_cycle_h = torch.mean(torch.abs(recov_human - human_data))
            
            # Combined cycle consistency
            g_cycle = g_cycle_m + g_cycle_h
            
            # MMD losses (replacing identity mapping)
            g_mmd_h = mmd_rbf(fake_human, human_data)
            g_mmd_m = mmd_rbf(fake_mouse, mouse_data)
            g_mmd = g_mmd_h + g_mmd_m
            
            # Biological consistency losses
            g_bio_consist = torch.tensor(0.0, device=device)
            if epoch >= config.get('bio_consistency_start_epoch', 50):
                g_corr_m = biological_consistency_loss(mouse_data, fake_human)
                g_corr_h = biological_consistency_loss(human_data, fake_mouse)
                g_bio_consist = g_corr_m + g_corr_h
            
            # Negative penalties
            neg_mag_h, neg_prop_h = get_negative_penalty(fake_human, curr_neg_weight)
            neg_mag_m, neg_prop_m = get_negative_penalty(fake_mouse, curr_neg_weight)
            neg_penalty = neg_mag_h + neg_mag_m
            
            # Total generator loss
            g_loss = (g_adv + lambda_cycle * g_cycle + lambda_mmd * g_mmd + 
                     lambda_consistency * g_bio_consist + neg_penalty)
            
            g_loss.backward()
            g_optimizer.step()
            
            # Track metrics
            g_losses.append(g_loss.item())
            g_adv_losses.append(g_adv.item())
            g_cycle_losses.append(g_cycle.item())
            g_cycle_m_losses.append(g_cycle_m.item())
            g_cycle_h_losses.append(g_cycle_h.item())
            g_mmd_losses.append(g_mmd.item())
            g_bio_losses.append(g_bio_consist.item())
            neg_penalties.append(neg_penalty.item())
            
            progress_bar.update(1)
            
            # Print metrics periodically
            if batch_idx % 10 == 0:
                att_msg = ""
                if use_self_attention and use_cross_attention:
                    att_msg = "[Self+Cross Attention]"
                elif use_self_attention:
                    att_msg = "[Self-Attention]"
                elif use_cross_attention:
                    att_msg = "[Cross-Attention]"
                
                tqdm.write(f"  Batch [{batch_idx}/{total_batches}] {att_msg} "
                          f"D_m: {d_m_loss.item():.4f}, "
                          f"D_h: {d_h_loss.item():.4f}, "
                          f"G: {g_loss.item():.4f}, "
                          f"G_adv: {g_adv.item():.4f}, "
                          f"G_cycle: {g_cycle.item():.4f}, "
                          f"G_mmd: {g_mmd.item():.4f}, "
                          f"G_bio: {g_bio_consist.item():.4f}")
        
        progress_bar.close()
        
        # Compute epoch averages
        avg_d_m_loss = np.mean(d_m_losses)
        avg_d_h_loss = np.mean(d_h_losses)
        avg_g_loss = np.mean(g_losses)
        avg_g_adv = np.mean(g_adv_losses)
        avg_g_cycle = np.mean(g_cycle_losses)
        avg_g_cycle_m = np.mean(g_cycle_m_losses)
        avg_g_cycle_h = np.mean(g_cycle_h_losses)
        avg_g_mmd = np.mean(g_mmd_losses)
        avg_g_bio = np.mean(g_bio_losses)
        avg_neg_penalty = np.mean(neg_penalties)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Mouse Discriminator Loss: {avg_d_m_loss:.4f}")
        print(f"  Human Discriminator Loss: {avg_d_h_loss:.4f}")
        print(f"  Generator Loss: {avg_g_loss:.4f}")
        print(f"  Adversarial Loss: {avg_g_adv:.4f}")
        print(f"  Cycle Consistency Loss: {avg_g_cycle:.4f}")
        print(f"    Forward Cycle (M→H→M): {avg_g_cycle_m:.4f}")
        print(f"    Backward Cycle (H→M→H): {avg_g_cycle_h:.4f}")
        print(f"  MMD Loss: {avg_g_mmd:.4f}")
        print(f"  Biological Consistency Loss: {avg_g_bio:.4f}")
        print(f"  Negative Penalty: {avg_neg_penalty:.4f}")
        print(f"  Attention: Self={use_self_attention}, Cross={use_cross_attention}")
        
        # Log metrics to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'd_m_loss': avg_d_m_loss,
                    'd_h_loss': avg_d_h_loss,
                    'g_loss': avg_g_loss,
                    'g_adv': avg_g_adv,
                    'g_cycle': avg_g_cycle,
                    'g_cycle_m': avg_g_cycle_m,
                    'g_cycle_h': avg_g_cycle_h,
                    'g_mmd': avg_g_mmd,
                    'g_biological': avg_g_bio,
                    'neg_penalty': avg_neg_penalty,
                    'neg_penalty_weight': curr_neg_weight,
                    'use_self_attention': use_self_attention,
                    'use_cross_attention': use_cross_attention
                })
        except ImportError:
            pass
        
        # Save models periodically
        if save_fn is not None and (epoch % config.get('save_interval', 20) == 0 or epoch == config['epochs'] - 1):
            save_fn(g_m2h, g_h2m, d_m, d_h, epoch)