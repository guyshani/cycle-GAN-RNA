import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import scipy.sparse
import anndata as ad
import h5py
import os
import time
import json
import wandb
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import our model and training functions
from cycle_gan_models import GeneExpressionGenerator, GeneExpressionDiscriminator
from cycle_gan_training import train_cycle_gan

def main(config_path=None):
    """
    Main function to set up and train the gene expression CycleGAN
    without orthology or cell type constraints
    """
    # Default configuration
    CONFIG = {
        'epochs': 300,
        'batch_size': 32,
        'lr': 1e-4,
        'beta1': 0.5,
        'beta2': 0.9,
        'eps': 1e-8,
        'nb_critic': 5,
        'lambda_gp': 10,
        'lambda_cycle': 10,
        'lambda_identity': 5,
        'negative_penalty_start_epoch': 50,
        'negative_penalty_ramp_epochs': 50,
        'max_negative_penalty': 5.0,
        'library_size': 10000,
        'save_interval': 20,
        'hidden_dims': [256, 512, 1024],
        'use_batch_norm': True,
        'dropout_rate': 0.1,
    }
    
    # Load config from file if provided
    if config_path is not None:
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            CONFIG.update(file_config)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"gene_cyclegan_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=4)
    
    # Load mouse data
    print("Loading mouse gene expression data...")
    mouse_data_path = CONFIG.get('mouse_data_path', 'path/to/mouse_data.h5ad')
    mouse_adata = ad.read_h5ad(mouse_data_path)
    
    # Extract mouse expression matrix
    if isinstance(mouse_adata.X, np.ndarray):
        mouse_expr = mouse_adata.X
    elif scipy.sparse.issparse(mouse_adata.X):
        print("Converting mouse sparse matrix to dense array...")
        mouse_expr = mouse_adata.X.toarray()
    else:
        print("Warning: Unexpected mouse data format. Attempting to convert to numpy array...")
        mouse_expr = np.array(mouse_adata.X)
    
    # Load human data
    print("Loading human gene expression data...")
    human_data_path = CONFIG.get('human_data_path', 'path/to/human_data.h5ad')
    human_adata = ad.read_h5ad(human_data_path)
    
    # Extract human expression matrix
    if isinstance(human_adata.X, np.ndarray):
        human_expr = human_adata.X
    elif scipy.sparse.issparse(human_adata.X):
        print("Converting human sparse matrix to dense array...")
        human_expr = human_adata.X.toarray()
    else:
        print("Warning: Unexpected human data format. Attempting to convert to numpy array...")
        human_expr = np.array(human_adata.X)
    
    # Print data info
    print(f"Mouse data: {mouse_expr.shape[0]} cells, {mouse_expr.shape[1]} genes")
    print(f"Human data: {human_expr.shape[0]} cells, {human_expr.shape[1]} genes")
    
    # Convert to PyTorch tensors
    mouse_tensor = torch.tensor(mouse_expr, dtype=torch.float32)
    human_tensor = torch.tensor(human_expr, dtype=torch.float32)
    
    # Create data loaders
    mouse_dataset = TensorDataset(mouse_tensor)
    human_dataset = TensorDataset(human_tensor)
    
    mouse_loader = DataLoader(
        mouse_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        drop_last=True
    )
    
    human_loader = DataLoader(
        human_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        drop_last=True
    )
    
    # Initialize models
    g_m2h = GeneExpressionGenerator(
        input_dim=mouse_expr.shape[1],
        output_dim=human_expr.shape[1],
        hidden_dims=CONFIG['hidden_dims'],
        library_size=CONFIG['library_size'],
        use_batch_norm=CONFIG['use_batch_norm'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    g_h2m = GeneExpressionGenerator(
        input_dim=human_expr.shape[1],
        output_dim=mouse_expr.shape[1],
        hidden_dims=CONFIG['hidden_dims'],
        library_size=CONFIG['library_size'],
        use_batch_norm=CONFIG['use_batch_norm'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    d_m = GeneExpressionDiscriminator(
        input_dim=mouse_expr.shape[1],
        hidden_dims=CONFIG['hidden_dims'][::-1],  # Reverse hidden dims for discriminator
        use_spectral_norm=True,
        dropout_rate=0.3
    ).to(device)
    
    d_h = GeneExpressionDiscriminator(
        input_dim=human_expr.shape[1],
        hidden_dims=CONFIG['hidden_dims'][::-1],  # Reverse hidden dims for discriminator
        use_spectral_norm=True,
        dropout_rate=0.3
    ).to(device)
    
    # Print model information
    print("\nModel architectures:")
    print(f"Generator (Mouse to Human): {sum(p.numel() for p in g_m2h.parameters()):,} parameters")
    print(f"Generator (Human to Mouse): {sum(p.numel() for p in g_h2m.parameters()):,} parameters")
    print(f"Discriminator (Mouse): {sum(p.numel() for p in d_m.parameters()):,} parameters")
    print(f"Discriminator (Human): {sum(p.numel() for p in d_h.parameters()):,} parameters")
    
    # Define save function
    def save_models(g_m2h, g_h2m, d_m, d_h, epoch):
        # Save generators
        g_m2h_path = os.path.join(output_dir, f"g_m2h_epoch_{epoch+1}.pt")
        g_h2m_path = os.path.join(output_dir, f"g_h2m_epoch_{epoch+1}.pt")
        
        torch.save(g_m2h.state_dict(), g_m2h_path)
        torch.save(g_h2m.state_dict(), g_h2m_path)
        
        # Save discriminators
        d_m_path = os.path.join(output_dir, f"d_m_epoch_{epoch+1}.pt")
        d_h_path = os.path.join(output_dir, f"d_h_epoch_{epoch+1}.pt")
        
        torch.save(d_m.state_dict(), d_m_path)
        torch.save(d_h.state_dict(), d_h_path)
        
        print(f"\nModels saved at epoch {epoch+1}")
        
        # Generate and visualize samples
        if epoch % 20 == 0:
            visualize_translations(g_m2h, g_h2m, mouse_expr, human_expr, 
                                  mouse_adata, human_adata, output_dir, epoch, device)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.save(g_m2h_path)
            wandb.save(g_h2m_path)
            wandb.save(d_m_path)
            wandb.save(d_h_path)
    
    # Initialize wandb
    if CONFIG.get('use_wandb', True):
        run_name = f"gene_cyclegan_{timestamp}"
        wandb.init(
            project='gene_expression_cyclegan',
            config=CONFIG,
            name=run_name,
            reinit=True
        )
    
    # Train the model
    train_cycle_gan(
        g_m2h=g_m2h,
        g_h2m=g_h2m,
        d_m=d_m,
        d_h=d_h,
        mouse_loader=mouse_loader,
        human_loader=human_loader,
        config=CONFIG,
        device=device,
        save_fn=save_models
    )
    
    # Final evaluation and visualization
    final_evaluation(g_m2h, g_h2m, mouse_expr, human_expr, mouse_adata, human_adata, 
                    output_dir, device)
    
    # Save final models
    final_g_m2h_path = os.path.join(output_dir, "g_m2h_final.pt")
    final_g_h2m_path = os.path.join(output_dir, "g_h2m_final.pt")
    
    torch.save(g_m2h.state_dict(), final_g_m2h_path)
    torch.save(g_h2m.state_dict(), final_g_h2m_path)
    
    print("\nTraining complete!")
    print(f"Models and outputs saved to: {output_dir}")
    
    # Close wandb
    if wandb.run is not None:
        wandb.finish()


def visualize_translations(g_m2h, g_h2m, mouse_expr, human_expr, 
                          mouse_adata, human_adata, output_dir, epoch, device):
    """
    Visualize gene expression translations between mouse and human
    """
    # Set models to evaluation mode
    g_m2h.eval()
    g_h2m.eval()
    
    # Create output directory for visualization
    viz_dir = os.path.join(output_dir, f"visualizations_epoch_{epoch+1}")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Sample some data points
    num_samples = 100
    mouse_indices = np.random.choice(mouse_expr.shape[0], num_samples, replace=False)
    human_indices = np.random.choice(human_expr.shape[0], num_samples, replace=False)
    
    mouse_samples = torch.tensor(mouse_expr[mouse_indices], dtype=torch.float32).to(device)
    human_samples = torch.tensor(human_expr[human_indices], dtype=torch.float32).to(device)
    
    # Generate translations
    with torch.no_grad():
        mouse_to_human = g_m2h(mouse_samples).cpu().numpy()
        human_to_mouse = g_h2m(human_samples).cpu().numpy()
        
        # Cycle translations
        mouse_to_human_to_mouse = g_h2m(torch.tensor(mouse_to_human, dtype=torch.float32).to(device)).cpu().numpy()
        human_to_mouse_to_human = g_m2h(torch.tensor(human_to_mouse, dtype=torch.float32).to(device)).cpu().numpy()
    
    # Get original samples in numpy format
    mouse_samples_np = mouse_samples.cpu().numpy()
    human_samples_np = human_samples.cpu().numpy()
    
    # Perform dimensionality reduction
    # PCA for mouse domain
    pca_mouse = PCA(n_components=2)
    mouse_real_pca = pca_mouse.fit_transform(mouse_samples_np)
    h2m_pca = pca_mouse.transform(human_to_mouse)
    m2h2m_pca = pca_mouse.transform(mouse_to_human_to_mouse)
    
    # PCA for human domain
    pca_human = PCA(n_components=2)
    human_real_pca = pca_human.fit_transform(human_samples_np)
    m2h_pca = pca_human.transform(mouse_to_human)
    h2m2h_pca = pca_human.transform(human_to_mouse_to_human)
    
    # Plot PCA visualizations
    plt.figure(figsize=(14, 6))
    
    # Mouse domain plot
    plt.subplot(1, 2, 1)
    plt.scatter(mouse_real_pca[:, 0], mouse_real_pca[:, 1], label='Real Mouse', alpha=0.7)
    plt.scatter(h2m_pca[:, 0], h2m_pca[:, 1], label='Human→Mouse', alpha=0.7)
    plt.scatter(m2h2m_pca[:, 0], m2h2m_pca[:, 1], label='Mouse→Human→Mouse', alpha=0.7)
    plt.title('Mouse Domain')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    
    # Human domain plot
    plt.subplot(1, 2, 2)
    plt.scatter(human_real_pca[:, 0], human_real_pca[:, 1], label='Real Human', alpha=0.7)
    plt.scatter(m2h_pca[:, 0], m2h_pca[:, 1], label='Mouse→Human', alpha=0.7)
    plt.scatter(h2m2h_pca[:, 0], h2m2h_pca[:, 1], label='Human→Mouse→Human', alpha=0.7)
    plt.title('Human Domain')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'pca_visualization.png'), dpi=300)
    plt.close()
    
    # Calculate translation metrics
    mouse_cycle_error = np.mean(np.abs(mouse_samples_np - mouse_to_human_to_mouse))
    human_cycle_error = np.mean(np.abs(human_samples_np - human_to_mouse_to_human))
    
    # Save metrics
    metrics = {
        'epoch': epoch,
        'mouse_cycle_error': float(mouse_cycle_error),
        'human_cycle_error': float(human_cycle_error)
    }
    
    with open(os.path.join(viz_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Log to wandb if available
    if wandb.run is not None:
        wandb.log({
            'epoch': epoch,
            'mouse_cycle_error': float(mouse_cycle_error),
            'human_cycle_error': float(human_cycle_error),
            'pca_visualization': wandb.Image(os.path.join(viz_dir, 'pca_visualization.png'))
        })
    
    # Return models to training mode
    g_m2h.train()
    g_h2m.train()
    
    print(f"Visualization for epoch {epoch+1} saved to {viz_dir}")
    print(f"Mouse cycle error: {mouse_cycle_error:.4f}")
    print(f"Human cycle error: {human_cycle_error:.4f}")


def final_evaluation(g_m2h, g_h2m, mouse_expr, human_expr, mouse_adata, human_adata, 
                    output_dir, device):
    """
    Perform final evaluation without cell type constraints
    """
    # Set models to evaluation mode
    g_m2h.eval()
    g_h2m.eval()
    
    # Create directory for final evaluation
    eval_dir = os.path.join(output_dir, "final_evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Sample more data for final evaluation
    num_samples = 200
    mouse_indices = np.random.choice(mouse_expr.shape[0], num_samples, replace=False)
    human_indices = np.random.choice(human_expr.shape[0], num_samples, replace=False)
    
    mouse_samples = torch.tensor(mouse_expr[mouse_indices], dtype=torch.float32).to(device)
    human_samples = torch.tensor(human_expr[human_indices], dtype=torch.float32).to(device)
    
    # Generate translations with larger batch size
    batch_size = 50
    mouse_to_human_batches = []
    human_to_mouse_batches = []
    mouse_to_human_to_mouse_batches = []
    human_to_mouse_to_human_batches = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            
            # Mouse to human translation
            m2h_batch = g_m2h(mouse_samples[i:end_idx]).cpu().numpy()
            mouse_to_human_batches.append(m2h_batch)
            
            # Human to mouse translation
            h2m_batch = g_h2m(human_samples[i:end_idx]).cpu().numpy()
            human_to_mouse_batches.append(h2m_batch)
            
            # Cycle translations
            m2h2m_batch = g_h2m(torch.tensor(m2h_batch, dtype=torch.float32).to(device)).cpu().numpy()
            mouse_to_human_to_mouse_batches.append(m2h2m_batch)
            
            h2m2h_batch = g_m2h(torch.tensor(h2m_batch, dtype=torch.float32).to(device)).cpu().numpy()
            human_to_mouse_to_human_batches.append(h2m2h_batch)
    
    # Combine batches
    mouse_to_human = np.vstack(mouse_to_human_batches)
    human_to_mouse = np.vstack(human_to_mouse_batches)
    mouse_to_human_to_mouse = np.vstack(mouse_to_human_to_mouse_batches)
    human_to_mouse_to_human = np.vstack(human_to_mouse_to_human_batches)
    
    # Get original samples in numpy format
    mouse_samples_np = mouse_samples.cpu().numpy()
    human_samples_np = human_samples.cpu().numpy()
    
    # Save generated data for further analysis
    np.save(os.path.join(eval_dir, 'mouse_samples.npy'), mouse_samples_np)
    np.save(os.path.join(eval_dir, 'human_samples.npy'), human_samples_np)
    np.save(os.path.join(eval_dir, 'mouse_to_human.npy'), mouse_to_human)
    np.save(os.path.join(eval_dir, 'human_to_mouse.npy'), human_to_mouse)
    np.save(os.path.join(eval_dir, 'mouse_to_human_to_mouse.npy'), mouse_to_human_to_mouse)
    np.save(os.path.join(eval_dir, 'human_to_mouse_to_human.npy'), human_to_mouse_to_human)
    
    # Create AnnData objects for translated data
    try:
        # Mouse to human translation as AnnData
        m2h_adata = ad.AnnData(X=mouse_to_human)
        
        # Human to mouse translation as AnnData
        h2m_adata = ad.AnnData(X=human_to_mouse)
        
        # Save AnnData objects
        m2h_adata.write_h5ad(os.path.join(eval_dir, 'mouse_to_human.h5ad'))
        h2m_adata.write_h5ad(os.path.join(eval_dir, 'human_to_mouse.h5ad'))
        
        print("Saved translated data as AnnData objects.")
    except Exception as e:
        print(f"Warning: Could not create AnnData objects for translations. Error: {e}")
    
    # Perform t-SNE for better visualization
    # Combine data for t-SNE
    mouse_domain_data = np.vstack([mouse_samples_np, human_to_mouse, mouse_to_human_to_mouse])
    human_domain_data = np.vstack([human_samples_np, mouse_to_human, human_to_mouse_to_human])
    
    # Labels for the datasets
    mouse_domain_labels = np.concatenate([
        np.full(num_samples, 'Real Mouse'),
        np.full(num_samples, 'Human→Mouse'),
        np.full(num_samples, 'Mouse→Human→Mouse')
    ])
    
    human_domain_labels = np.concatenate([
        np.full(num_samples, 'Real Human'),
        np.full(num_samples, 'Mouse→Human'),
        np.full(num_samples, 'Human→Mouse→Human')
    ])
    
    # t-SNE for mouse domain
    print("Performing t-SNE for mouse domain...")
    tsne_mouse = TSNE(n_components=2, random_state=42, perplexity=30)
    mouse_domain_tsne = tsne_mouse.fit_transform(mouse_domain_data)
    
    # t-SNE for human domain
    print("Performing t-SNE for human domain...")
    tsne_human = TSNE(n_components=2, random_state=42, perplexity=30)
    human_domain_tsne = tsne_human.fit_transform(human_domain_data)
    
    # Plot t-SNE visualizations
    plt.figure(figsize=(16, 7))
    
    # Mouse domain t-SNE
    plt.subplot(1, 2, 1)
    scatter_m = sns.scatterplot(
        x=mouse_domain_tsne[:, 0], 
        y=mouse_domain_tsne[:, 1], 
        hue=mouse_domain_labels,
        palette='viridis',
        alpha=0.8
    )
    plt.title('t-SNE: Mouse Domain')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Human domain t-SNE
    plt.subplot(1, 2, 2)
    scatter_h = sns.scatterplot(
        x=human_domain_tsne[:, 0], 
        y=human_domain_tsne[:, 1], 
        hue=human_domain_labels,
        palette='viridis',
        alpha=0.8
    )
    plt.title('t-SNE: Human Domain')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'tsne_visualization.png'), dpi=300)
    plt.close()
    
    # Generate heatmaps of top variable genes
    print("Generating gene expression heatmaps...")
    generate_expression_heatmaps(
        mouse_samples_np, human_samples_np, 
        mouse_to_human, human_to_mouse,
        mouse_adata, human_adata,
        eval_dir
    )
    
    # Calculate standard statistical metrics
    mouse_cycle_mse = np.mean(np.square(mouse_samples_np - mouse_to_human_to_mouse))
    human_cycle_mse = np.mean(np.square(human_samples_np - human_to_mouse_to_human))
    
    mouse_cycle_mae = np.mean(np.abs(mouse_samples_np - mouse_to_human_to_mouse))
    human_cycle_mae = np.mean(np.abs(human_samples_np - human_to_mouse_to_human))
    
    # Calculate correlation between original and cycle-translated data
    mouse_corr = np.mean([np.corrcoef(mouse_samples_np[i], mouse_to_human_to_mouse[i])[0, 1] 
                          for i in range(num_samples)])
    human_corr = np.mean([np.corrcoef(human_samples_np[i], human_to_mouse_to_human[i])[0, 1] 
                          for i in range(num_samples)])
    
    # Metrics for how well the distributions match
    mouse_to_human_distribution = calculate_distribution_distance(mouse_to_human, human_samples_np)
    human_to_mouse_distribution = calculate_distribution_distance(human_to_mouse, mouse_samples_np)
    
    # Calculate gene correlation preservation (pattern preservation)
    gene_corr_m = calculate_gene_correlation_preservation(mouse_samples_np, mouse_to_human_to_mouse)
    gene_corr_h = calculate_gene_correlation_preservation(human_samples_np, human_to_mouse_to_human)
    
    # Save all metrics
    metrics = {
        'mouse_cycle_mse': float(mouse_cycle_mse),
        'human_cycle_mse': float(human_cycle_mse),
        'mouse_cycle_mae': float(mouse_cycle_mae),
        'human_cycle_mae': float(human_cycle_mae),
        'mouse_cycle_correlation': float(mouse_corr),
        'human_cycle_correlation': float(human_corr),
        'mouse_to_human_distribution': float(mouse_to_human_distribution),
        'human_to_mouse_distribution': float(human_to_mouse_distribution),
        'mouse_gene_correlation_preservation': float(gene_corr_m),
        'human_gene_correlation_preservation': float(gene_corr_h)
    }
    
    with open(os.path.join(eval_dir, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print summary
    print("\n===== Final Evaluation Metrics =====")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    
    # Log to wandb if available
    if wandb.run is not None:
        wandb.log(metrics)
        wandb.log({'tsne_visualization': wandb.Image(os.path.join(eval_dir, 'tsne_visualization.png'))})
    
    print(f"Final evaluation complete. Results saved to {eval_dir}")


def generate_expression_heatmaps(mouse_real, human_real, mouse_to_human, human_to_mouse, 
                               mouse_adata, human_adata, output_dir, n_top_genes=30):
    """Generate heatmaps of top variable genes to visualize translation quality"""
    # Get gene names if available
    mouse_genes = mouse_adata.var_names if hasattr(mouse_adata, 'var_names') else None
    human_genes = human_adata.var_names if hasattr(human_adata, 'var_names') else None
    
    # Find top variable genes in each dataset
    def get_top_variable_genes(data, gene_names=None, n=n_top_genes):
        variances = np.var(data, axis=0)
        top_indices = np.argsort(variances)[-n:][::-1]  # Get indices of top n variable genes
        if gene_names is not None:
            return top_indices, [gene_names[i] for i in top_indices]
        return top_indices, [f"Gene {i}" for i in top_indices]
    
    # Get top variable genes
    top_mouse_indices, top_mouse_gene_names = get_top_variable_genes(mouse_real, mouse_genes)
    top_human_indices, top_human_gene_names = get_top_variable_genes(human_real, human_genes)
    
    # Subset data to top variable genes
    mouse_subset = mouse_real[:, top_mouse_indices]
    h2m_subset = human_to_mouse[:, top_mouse_indices]
    
    human_subset = human_real[:, top_human_indices]
    m2h_subset = mouse_to_human[:, top_human_indices]
    
    # Function to plot heatmap
    def plot_comparison_heatmap(data1, data2, gene_names, title1, title2, output_path):
        # Sample a subset of cells for clarity
        n_cells = min(50, data1.shape[0])
        indices = np.random.choice(data1.shape[0], n_cells, replace=False)
        
        # Combine data
        combined_data = np.vstack([data1[indices], data2[indices]])
        
        # Create labels
        labels = np.concatenate([
            np.full(n_cells, title1),
            np.full(n_cells, title2)
        ])
        
        # Create DataFrame for seaborn
        df = pd.DataFrame(combined_data, columns=gene_names)
        df['Dataset'] = labels
        
        # Reshape for heatmap
        df_melted = df.melt(id_vars=['Dataset'], var_name='Gene', value_name='Expression')
        
        # Create pivot table
        pivot_df = df_melted.pivot_table(index='Dataset', columns='Gene', values='Expression')
        
        # Plot heatmap
        plt.figure(figsize=(15, 6))
        sns.heatmap(pivot_df, cmap='viridis', robust=True)
        plt.title(f'Expression Comparison: {title1} vs {title2}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Heatmap saved to {output_path}")
    
    # Plot heatmaps
    plot_comparison_heatmap(
        mouse_subset, h2m_subset, 
        top_mouse_gene_names, 
        'Real Mouse', 'Human→Mouse',
        os.path.join(output_dir, 'mouse_domain_heatmap.png')
    )
    
    plot_comparison_heatmap(
        human_subset, m2h_subset, 
        top_human_gene_names, 
        'Real Human', 'Mouse→Human',
        os.path.join(output_dir, 'human_domain_heatmap.png')
    )
    
    return top_mouse_indices, top_human_indices


def calculate_gene_correlation_preservation(original, translated, top_gene_indices=None):
    """
    Calculate how well gene-gene correlations are preserved in the translation
    Returns a score between 0 (no preservation) and 1 (perfect preservation)
    """
    # Use all genes if no subset is specified
    if top_gene_indices is None:
        # Use up to 500 random genes to avoid memory issues with large datasets
        n_genes = min(500, original.shape[1])
        gene_indices = np.random.choice(original.shape[1], n_genes, replace=False)
    else:
        gene_indices = top_gene_indices
    
    # Subset to selected genes
    orig_subset = original[:, gene_indices]
    trans_subset = translated[:, gene_indices]
    
    # Calculate correlation matrices
    orig_corr = np.corrcoef(orig_subset.T)
    trans_corr = np.corrcoef(trans_subset.T)
    
    # Flatten the upper triangular part of correlation matrices
    n = orig_corr.shape[0]
    orig_flat = []
    trans_flat = []
    
    for i in range(n):
        for j in range(i+1, n):
            orig_flat.append(orig_corr[i, j])
            trans_flat.append(trans_corr[i, j])
    
    # Calculate correlation between the correlation matrices
    preservation_score = np.corrcoef(orig_flat, trans_flat)[0, 1]
    
    return preservation_score


def calculate_distribution_distance(generated, real):
    """
    Calculate a distance metric between generated and real data distributions
    using mean absolute difference of means and std deviations
    """
    gen_means = np.mean(generated, axis=0)
    real_means = np.mean(real, axis=0)
    
    gen_stds = np.std(generated, axis=0)
    real_stds = np.std(real, axis=0)
    
    mean_diff = np.mean(np.abs(gen_means - real_means))
    std_diff = np.mean(np.abs(gen_stds - real_stds))
    
    return mean_diff + std_diff


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a CycleGAN for gene expression translation')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--mouse_data', type=str, help='Path to mouse gene expression data (.h5ad)')
    parser.add_argument('--human_data', type=str, help='Path to human gene expression data (.h5ad)')
    
    args = parser.parse_args()
    
    # Update config with paths from args
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    if args.mouse_data:
        config['mouse_data_path'] = args.mouse_data
    
    if args.human_data:
        config['human_data_path'] = args.human_data
    
    # Save updated config
    if args.config or args.mouse_data or args.human_data:
        config_path = 'temp_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        config_path = None
    
    # Run main function
    main(config_path=config_path)
    
    # Clean up temporary config file
    if config_path and config_path == 'temp_config.json':
        os.remove(config_path)