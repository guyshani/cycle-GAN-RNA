import scanpy as sc
import gseapy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from scipy import sparse
from scipy import stats as scipy_stats
from tqdm import tqdm
import h5py
import anndata as ad
import os
import time
from statsmodels.stats.multitest import fdrcorrection
from sklearn.metrics import precision_recall_curve, average_precision_score
from joblib import Parallel, delayed
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

# Try to import numba for optimized functions
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available. Some optimizations will be disabled.")
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics import precision_recall_curve, average_precision_score
import gseapy as gp
from statsmodels.stats.multitest import fdrcorrection
import time
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# ---- Helper Functions for Memory Efficient Processing ----

def process_anndata_in_chunks(adata, chunk_size=1000):
    """
    Process an AnnData object in chunks to avoid memory issues
    """
    print(f"Processing AnnData with {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Create a new AnnData with only necessary data
    if 'cell_type' not in adata.obs.columns and 'myannotations' in adata.obs.columns:
        adata.obs['cell_type'] = adata.obs['myannotations']
    
    # Get the data
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    
    # Calculate the library size (total counts per cell)
    library_size = np.zeros(n_cells)
    
    # Process in chunks to avoid memory issues
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        
        # Get chunk data
        chunk = adata[start:end]
        X_chunk = chunk.X.toarray() if hasattr(chunk.X, 'toarray') else chunk.X
        
        # Calculate sums
        library_size[start:end] = X_chunk.sum(axis=1)
    
    # Create a new AnnData for normalized data
    adata_norm = ad.AnnData(
        X=np.zeros((n_cells, n_genes), dtype=np.float32),
        obs=adata.obs[['cell_type']].copy(),
        var=pd.DataFrame(index=adata.var_names)
    )
    
    # Normalize and log-transform in chunks
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        
        # Get chunk data
        chunk = adata[start:end]
        X_chunk = chunk.X.toarray() if hasattr(chunk.X, 'toarray') else chunk.X
        
        # Normalize to 10,000 counts
        X_norm = X_chunk / library_size[start:end, np.newaxis] * 10000
        
        # Log transform
        X_log = np.log1p(X_norm)
        
        # Store in new AnnData
        adata_norm.X[start:end] = X_log
        
        # Clean up to free memory
        del X_chunk, X_norm, X_log
        gc.collect()
    
    # The processed data is ready
    print(f"Processed data shape: {adata_norm.shape}")
    return adata_norm


def compute_rankings_for_cell_type(adata, cell_type, chunk_size=100):
    """
    Compute gene rankings for one cell type vs all others with minimal memory footprint
    """
    print(f"Computing rankings for cell type: {cell_type}")
    
    # Create binary mask for cell type
    cell_mask = adata.obs['cell_type'] == cell_type
    
    # Get indices for the cell type and others
    cell_type_indices = np.where(cell_mask)[0]
    other_indices = np.where(~cell_mask)[0]
    
    n_genes = adata.n_vars
    
    # Results storage
    log2fc = np.zeros(n_genes)
    pvals = np.ones(n_genes)
    
    # Process genes in chunks to minimize memory usage
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        
        # Get gene expression for this chunk
        if hasattr(adata.X, 'toarray'):
            # For sparse matrices, only convert what we need
            expr_chunk = adata.X[:, start:end].toarray()
        else:
            expr_chunk = adata.X[:, start:end]
        
        # Calculate statistics for each gene in the chunk
        for i in range(end - start):
            gene_idx = start + i
            
            # Get expression data for the current gene
            gene_expr = expr_chunk[:, i]
            
            # Calculate fold change
            mean_cell_type = np.mean(gene_expr[cell_type_indices])
            mean_others = np.mean(gene_expr[other_indices])
            
            # Avoid division by zero
            epsilon = 1e-10
            log2fc[gene_idx] = np.log2((mean_cell_type + epsilon) / (mean_others + epsilon))
            
            # Calculate p-value
            try:
                _, pvals[gene_idx] = stats.mannwhitneyu(
                    gene_expr[cell_type_indices],
                    gene_expr[other_indices],
                    alternative='two-sided'
                )
            except:
                pvals[gene_idx] = 1.0
        
        # Clean up to free memory
        del expr_chunk
        gc.collect()
    
    # Ensure no zero p-values (to avoid infinite -log10)
    pvals = np.maximum(pvals, np.finfo(float).tiny)
    
    # Create ranking score
    ranking_metric = -np.log10(pvals) * np.sign(log2fc)
    
    # Create ranked gene list
    rankings = pd.Series(
        ranking_metric,
        index=adata.var_names,
        name='ranking'
    ).sort_values(ascending=False)
    
    return rankings


def run_gsea_for_cell_type(cell_type, rankings, gene_sets, min_size=15, max_size=500, permutations=1000):
    """
    Run GSEA for a single cell type
    """
    try:
        print(f"Running GSEA for cell type: {cell_type}")
        pre_res = gp.prerank(
            rnk=rankings,
            gene_sets=gene_sets,
            min_size=min_size,
            max_size=max_size,
            permutation_num=permutations,
            threads=1,  # Use single thread for this cell type
            seed=42,
            no_plot=True
        )
        return cell_type, pre_res.res2d
    except Exception as e:
        print(f"Error running GSEA for {cell_type}: {e}")
        return cell_type, pd.DataFrame()


def run_gsea_parallel(adata, gene_sets, max_workers=4, min_size=15, max_size=500, permutations=1000):
    """
    Run GSEA analysis for all cell types with parallel processing
    """
    cell_types = adata.obs['cell_type'].unique()
    results = {}
    
    # First compute rankings for each cell type
    rankings_dict = {}
    for cell_type in cell_types:
        rankings = compute_rankings_for_cell_type(adata, cell_type)
        rankings_dict[cell_type] = rankings
    
    # Run GSEA in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_cell_type = {
            executor.submit(
                run_gsea_for_cell_type, 
                cell_type, 
                rankings_dict[cell_type],
                gene_sets,
                min_size,
                max_size,
                permutations
            ): cell_type for cell_type in cell_types
        }
        
        for future in future_to_cell_type:
            cell_type, res = future.result()
            if not res.empty:
                results[cell_type] = res
    
    return results


def compare_gsea_results(gen_results, real_results):
    """
    Compare GSEA results between generated and real data
    """
    # Convert to dataframes
    df_gen = pd.concat([df.assign(cell_type=ct) for ct, df in gen_results.items()])
    df_real = pd.concat([df.assign(cell_type=ct) for ct, df in real_results.items()])
    
    # Ensure numeric columns
    for df in [df_gen, df_real]:
        df['NES'] = pd.to_numeric(df['NES'], errors='coerce')
        df['FDR q-val'] = pd.to_numeric(df['FDR q-val'], errors='coerce')
    
    # Merge results
    merged = pd.merge(
        df_gen[['Term', 'cell_type', 'NES', 'FDR q-val']].rename(
            columns={'NES': 'NES_gen', 'FDR q-val': 'FDR_gen'}
        ),
        df_real[['Term', 'cell_type', 'NES', 'FDR q-val']].rename(
            columns={'NES': 'NES_real', 'FDR q-val': 'FDR_real'}
        ),
        on=['Term', 'cell_type'],
        how='inner'
    )
    
    print(f"Merged {len(merged)} pathways across {len(merged['cell_type'].unique())} cell types")
    
    return df_gen, df_real, merged


def plot_gsea_correlation(merged_df, output_path=None):
    """
    Plot correlation between generated and real GSEA results
    """
    # Calculate correlations for each cell type
    cell_types = merged_df['cell_type'].unique()
    correlations = []
    
    for ct in cell_types:
        ct_data = merged_df[merged_df['cell_type'] == ct]
        if len(ct_data) > 1:
            corr, pval = stats.pearsonr(ct_data['NES_gen'], ct_data['NES_real'])
            correlations.append({
                'cell_type': ct,
                'correlation': corr,
                'p_value': pval,
                'n_pathways': len(ct_data)
            })
    
    # Create correlation dataframe
    corr_df = pd.DataFrame(correlations)
    
    # Apply multiple testing correction
    _, corr_df['adj_p_value'] = fdrcorrection(corr_df['p_value'])
    
    # Sort by correlation
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    # Plot correlation for each cell type
    plt.figure(figsize=(12, 8))
    
    # Create a colormap for significant correlations
    colors = ['green' if p < 0.05 else 'grey' for p in corr_df['adj_p_value']]
    
    # Create barplot
    sns.barplot(x='correlation', y='cell_type', data=corr_df, palette=colors)
    
    # Add a vertical line at 0
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add labels with p-values
    for i, row in enumerate(corr_df.itertuples()):
        sig_stars = '***' if row.adj_p_value < 0.001 else ('**' if row.adj_p_value < 0.01 else 
                                                     ('*' if row.adj_p_value < 0.05 else ''))
        plt.text(row.correlation + 0.05, i, f"r={row.correlation:.2f} {sig_stars} (n={row.n_pathways})")
    
    plt.title('Correlation between Generated and Real GSEA Results by Cell Type')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Cell Type')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return corr_df


def plot_pathway_correlations(merged_df, output_dir=None):
    """
    Plot scatter plots of NES correlations for each cell type
    """
    cell_types = merged_df['cell_type'].unique()
    
    # Calculate how many rows and columns we need for the subplot grid
    n_types = len(cell_types)
    n_cols = min(3, n_types)
    n_rows = (n_types + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    # Handle the case with only one subplot
    if n_types == 1:
        axes = np.array([axes])
    
    # Flatten the axes array for easy indexing
    axes = axes.flatten()
    
    # Define colormap for different p-value thresholds
    cmap = {
        'both_sig': 'blue',      # Significant in both
        'real_only': 'green',    # Significant in real only
        'gen_only': 'orange',    # Significant in generated only
        'none': 'lightgrey'      # Not significant in either
    }
    
    # Plot each cell type
    for i, ct in enumerate(cell_types):
        # Get data for this cell type
        ct_data = merged_df[merged_df['cell_type'] == ct].copy()
        
        # Create significance categories
        ct_data['significance'] = 'none'
        ct_data.loc[(ct_data['FDR_gen'] < 0.1) & (ct_data['FDR_real'] < 0.1), 'significance'] = 'both_sig'
        ct_data.loc[(ct_data['FDR_real'] < 0.1) & (ct_data['FDR_gen'] >= 0.1), 'significance'] = 'real_only'
        ct_data.loc[(ct_data['FDR_gen'] < 0.1) & (ct_data['FDR_real'] >= 0.1), 'significance'] = 'gen_only'
        
        # Plot points with different colors based on significance
        for category, color in cmap.items():
            subset = ct_data[ct_data['significance'] == category]
            if len(subset) > 0:
                axes[i].scatter(
                    subset['NES_gen'], 
                    subset['NES_real'], 
                    c=color, 
                    alpha=0.7, 
                    label=f"{category} (n={len(subset)})"
                )
        
        # Calculate correlation
        if len(ct_data) > 1:
            corr, pval = stats.pearsonr(ct_data['NES_gen'], ct_data['NES_real'])
            axes[i].set_title(f"{ct} (r={corr:.2f}, p={pval:.1e}, n={len(ct_data)})")
        else:
            axes[i].set_title(ct)
        
        # Add correlation line for all points
        if len(ct_data) > 1:
            m, b = np.polyfit(ct_data['NES_gen'], ct_data['NES_real'], 1)
            axes[i].plot(
                np.array([-3, 3]), 
                m * np.array([-3, 3]) + b, 
                '-', 
                color='red', 
                alpha=0.7
            )
        
        # Add diagonal reference line
        axes[i].plot([-3, 3], [-3, 3], '--', color='grey', alpha=0.5)
        
        # Set axis labels and limits
        axes[i].set_xlabel('Generated NES')
        axes[i].set_ylabel('Real NES')
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        
        # Add legend to first plot only to avoid clutter
        if i == 0:
            axes[i].legend(loc='best')
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output directory is provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pathway_correlation_by_celltype.png'), 
                    dpi=300, bbox_inches='tight')
    
    return fig


def create_precision_recall_curves(merged_df, output_dir=None):
    """
    Create precision-recall curves for pathway prediction
    """
    # Group data by cell type
    cell_types = merged_df['cell_type'].unique()
    
    # Setup PR curve data
    pr_data = {}
    
    plt.figure(figsize=(12, 8))
    
    # Calculate PR curves for each cell type
    for ct in cell_types:
        # Get data for this cell type
        ct_data = merged_df[merged_df['cell_type'] == ct].copy()
        
        # Create binary labels (1 for significant in real data, 0 for not)
        y_true = (ct_data['FDR_real'] < 0.1).astype(int)
        
        # Skip if no positive examples
        if y_true.sum() == 0:
            continue
        
        # Use -log10(p-value) from generated data as score
        scores = -np.log10(ct_data['FDR_gen'])
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        average_precision = average_precision_score(y_true, scores)
        
        # Store results
        pr_data[ct] = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': average_precision,
            'n_positive': y_true.sum(),
            'n_total': len(y_true)
        }
        
        # Plot the curve
        plt.plot(recall, precision, label=f'{ct} (AP={average_precision:.2f}, {y_true.sum()}/{len(y_true)})')
    
    # Add a random baseline
    plt.plot([0, 1], [0.5, 0.5], 'k--', label='Random baseline')
    
    # Add labels and legend
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Pathway Significance Prediction')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save figure if output directory is provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), 
                    dpi=300, bbox_inches='tight')
    
    return pr_data


# ---- Main Functions ----

def load_gmt_file(gmt_file):
    """
    Load and parse GMT file into dictionary format
    """
    gene_sets = {}
    with open(gmt_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:  # Skip malformed lines
                continue
            pathway_name = parts[0]
            pathway_description = parts[1]
            genes = [gene for gene in parts[2:] if gene]  # Remove empty strings
            gene_sets[pathway_name] = genes
    
    print(f"Loaded {len(gene_sets)} gene sets")
    return gene_sets


def run_translation_gsea_analysis(
    real_adata, 
    gen_adata, 
    gmt_file, 
    output_dir=None, 
    max_workers=4,
    min_size=15, 
    max_size=500, 
    permutations=1000
):
    """
    Run GSEA analysis to compare real and generated data
    
    Parameters:
    -----------
    real_adata : AnnData
        AnnData object with real gene expression data
    gen_adata : AnnData
        AnnData object with generated gene expression data
    gmt_file : str
        Path to GMT file with gene sets
    output_dir : str
        Directory to save output files
    max_workers : int
        Maximum number of parallel workers
    min_size : int
        Minimum gene set size
    max_size : int
        Maximum gene set size
    permutations : int
        Number of permutations for GSEA
        
    Returns:
    --------
    dict
        Dictionary with results
    """
    start_time = time.time()
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Ensure cell_type column exists in both datasets
    for adata, name in [(real_adata, 'real'), (gen_adata, 'generated')]:
        if 'cell_type' not in adata.obs.columns:
            if 'myannotations' in adata.obs.columns:
                adata.obs['cell_type'] = adata.obs['myannotations'].copy()
            else:
                raise ValueError(f"{name} data doesn't have 'cell_type' or 'myannotations' column")
    
    # Preprocess data
    print("Preprocessing real data...")
    real_adata_proc = process_anndata_in_chunks(real_adata)
    
    print("Preprocessing generated data...")
    gen_adata_proc = process_anndata_in_chunks(gen_adata)
    
    # Load gene sets
    print("Loading gene sets...")
    gene_sets = load_gmt_file(gmt_file)
    
    # Run GSEA
    print(f"Running GSEA for real data with {max_workers} workers...")
    real_results = run_gsea_parallel(
        real_adata_proc, 
        gene_sets,
        max_workers=max_workers,
        min_size=min_size,
        max_size=max_size,
        permutations=permutations
    )
    
    print(f"Running GSEA for generated data with {max_workers} workers...")
    gen_results = run_gsea_parallel(
        gen_adata_proc, 
        gene_sets,
        max_workers=max_workers,
        min_size=min_size,
        max_size=max_size,
        permutations=permutations
    )
    
    # Compare results
    print("Comparing GSEA results...")
    df_gen, df_real, merged_df = compare_gsea_results(gen_results, real_results)
    
    # Save merged results
    if output_dir:
        merged_df.to_csv(os.path.join(output_dir, 'merged_gsea_results.csv'))
        df_gen.to_csv(os.path.join(output_dir, 'generated_gsea_results.csv'))
        df_real.to_csv(os.path.join(output_dir, 'real_gsea_results.csv'))
    
    # Generate correlation plots
    print("Generating correlation plots...")
    corr_df = plot_gsea_correlation(merged_df, 
                                  output_path=os.path.join(output_dir, 'gsea_correlation.png') if output_dir else None)
    
    # Generate pathway correlation plots
    fig = plot_pathway_correlations(merged_df, output_dir)
    
    # Generate precision-recall curves
    pr_data = create_precision_recall_curves(merged_df, output_dir)
    
    elapsed_time = time.time() - start_time
    print(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    # Return results
    return {
        'gen_results': gen_results,
        'real_results': real_results,
        'df_gen': df_gen,
        'df_real': df_real,
        'merged_df': merged_df,
        'corr_df': corr_df,
        'pr_data': pr_data
    }

# Example usage
if __name__ == "__main__":
    # Load data
    real_adata = ad.read_h5ad("real_data.h5ad")
    gen_adata = ad.read_h5ad("generated_data.h5ad")
    
    # Run analysis
    results = run_translation_gsea_analysis(
        real_adata,
        gen_adata,
        gmt_file="pathway_database.gmt",
        output_dir="gsea_results",
        max_workers=4,
        permutations=1000
    )
    
    # Examine correlation results
    print(results['corr_df'])





__all__ = [
    'CellTypeGSEA',
    'process_data_for_gsea',
    'fishers_combined_probability',
    'fishers_z_transform',
    'stouffers_z_score',
    'prepare_results_df',
    'calculate_pr_curves',
    'plot_pr_curves',
    'run_comparison_analysis',
    'analyze_gsea_correlation',
    'plot_gsea_correlation',
    'OptimizedCellTypeGSEA',  # New optimized class
    'enable_optimization'     # Function to enable optimizations
]

# Global cache settings
CACHE_DIR = None
ENABLE_CACHE = False

def enable_optimization(cache_dir=None):
    """
    Enable optimization features globally
    
    Parameters:
    -----------
    cache_dir : str, optional
        Directory to use for caching. If None, caching is disabled.
    """
    global CACHE_DIR, ENABLE_CACHE
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        CACHE_DIR = cache_dir
        ENABLE_CACHE = True
        print(f"Optimizations enabled with caching in: {cache_dir}")
    else:
        CACHE_DIR = None
        ENABLE_CACHE = False
        print("Optimizations enabled without caching")


class CellTypeGSEA:
    """
    A class to perform GSEA analysis between different cell types in single-cell data.
    """
    def __init__(
        self, 
        adata: sc.AnnData,
        cell_type_key: str = 'cell_type',
        gmt_file: str = None 
    ):
        """
        Initialize the GSEA analysis object.
        
        Parameters:
        -----------
        adata : AnnData
            Annotated data matrix with cell type annotations
        cell_type_key : str
            Key in adata.obs containing cell type labels
        gmt_file : str
            path to the gene set files (gmt format)
        """
        self.adata = adata
        self.cell_type_key = cell_type_key
        if gmt_file is not None:
            self.gene_sets = self._load_gmt_file(gmt_file)
        self.results = {}

    def _load_gmt_file(self, gmt_file):
        """
        Load and parse GMT file into dictionary format
        """
        gene_sets = {}
        with open(gmt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:  # Skip malformed lines
                    continue
                pathway_name = parts[0]
                pathway_description = parts[1]
                genes = [gene for gene in parts[2:] if gene]  # Remove empty strings
                gene_sets[pathway_name] = genes
        
        print(f"Loaded {len(gene_sets)} gene sets")
        # Print first gene set as example
        first_set = next(iter(gene_sets.items()))
        print(f"\nExample gene set:")
        print(f"Name: {first_set[0]}")
        print(f"Number of genes: {len(first_set[1])}")
        print(f"First few genes: {first_set[1][:5]}")
        
        return gene_sets

    def compute_cell_type_rankings(
        self,
        cell_type: str
    ) -> pd.Series:
        """
        Compute differential expression rankings for one cell type vs all others.
        
        Parameters:
        -----------
        cell_type : str
            Cell type to analyze
            
        Returns:
        --------
        pd.Series
            Ranked gene list with ranking scores
        """
        # Create binary mask for cell type
        cell_mask = self.adata.obs[self.cell_type_key] == cell_type
        
        # Initialize results storage
        n_genes = self.adata.n_vars
        scores = np.zeros(n_genes)
        pvals = np.zeros(n_genes)
        
        # Get expression matrix
        if sparse.issparse(self.adata.X):
            X = self.adata.X.toarray()
        else:
            X = self.adata.X
        
        # Compute rankings for each gene
        for i in range(n_genes):
            gene_expr = X[:, i]
            
            # Perform Mann-Whitney U test
            stat, pval = scipy_stats.mannwhitneyu(
                gene_expr[cell_mask],
                gene_expr[~cell_mask],
                alternative='two-sided'
            )
            
            # Compute effect size (log2 fold change)
            mean_1 = np.mean(gene_expr[cell_mask])
            mean_2 = np.mean(gene_expr[~cell_mask])
            log2fc = np.log2((mean_1 + 1e-10) / (mean_2 + 1e-10))
            
            scores[i] = log2fc
            pvals[i] = pval
        
        # Create ranking metric
        # Add a small epsilon to p-values to avoid log10(0)
        min_pval = np.finfo(float).tiny  # Smallest positive float
        pvals = np.maximum(pvals, min_pval)
        ranking_metric = -np.log10(pvals) * np.sign(scores)
        
        # Create ranked gene list
        gene_names = [f'gene_{i}' for i in range(n_genes)] if self.adata.var_names.empty else self.adata.var_names
        rankings = pd.Series(
            ranking_metric,
            index=gene_names,
            name='ranking'
        ).sort_values(ascending=False)
        
        return rankings

    def run_gsea(
        self,
        min_size: int = 15,
        max_size: int = 500,
        permutations: int = 1000,
        threads: int = 4
    ) -> Dict:
        """
        Run GSEA analysis for all cell types.
        
        Parameters:
        -----------
        min_size : int
            Minimum gene set size
        max_size : int
            Maximum gene set size
        permutations : int
            Number of permutations
        threads : int
            Number of parallel threads
            
        Returns:
        --------
        Dict
            Dictionary containing GSEA results for each cell type
        """
        # Get unique cell types
        cell_types = self.adata.obs[self.cell_type_key].unique()
        
        print("Running GSEA analysis for each cell type...")
        for cell_type in tqdm(cell_types):
            # Get rankings for this cell type
            rankings = self.compute_cell_type_rankings(cell_type)
            
            # Run GSEA
            pre_res = gp.prerank(
                rnk=rankings,
                gene_sets=self.gene_sets,
                min_size=min_size,
                max_size=max_size,
                permutation_num=permutations,
                threads=threads,
                seed=42,
                no_plot=True
            )
            
            # Store results
            self.results[cell_type] = pre_res.res2d
            
        return self.results

    def plot_top_pathways(
        self,
        n_pathways: int = 10,
        fdr_cutoff: float = 0.05,
        figsize: tuple = (15, 10)
    ) -> sns.FacetGrid:
        """
            Plot top enriched pathways for each cell type.
            
            Parameters:
            -----------
            n_pathways : int
                Number of top pathways to show
            fdr_cutoff : float
                FDR cutoff for significance
            figsize : tuple
                Figure size
                
            Returns:
            --------
            sns.FacetGrid
                Seaborn FacetGrid object with the plot
        """
        # Combine all results
        all_results = []
        for cell_type, res in self.results.items():
            df = res.copy()
            df['cell_type'] = cell_type
            all_results.append(df)
        
        combined_results = pd.concat(all_results)

        # Convert NES and FDR q-val to numeric
        combined_results['NES'] = pd.to_numeric(combined_results['NES'], errors='coerce')
        combined_results['FDR q-val'] = pd.to_numeric(combined_results['FDR q-val'], errors='coerce')
        
        # Filter significant pathways using standard boolean indexing
        sig_pathways = combined_results[combined_results['FDR q-val'] < fdr_cutoff]
        
        # Get top pathways for each cell type
        top_pathways_list = []
        for name, group in sig_pathways.groupby('cell_type'):
            top_n = group.nlargest(n_pathways, 'NES')
            top_pathways_list.append(top_n)
    
        top_pathways = pd.concat(top_pathways_list, ignore_index=True)
        
        # Create plot
        fig = plt.figure(figsize=figsize)
        # create FacetGrid
        g = sns.FacetGrid(
            data=top_pathways,
            col='cell_type',
            col_wrap=3,
            height=6,
            aspect=1.5
        )
        
        g.map_dataframe(
            sns.barplot,
            x='NES',
            y='Term',
            hue='FDR q-val',
            palette='RdBu_r'
        )

        # Adjust y-axis label spacing for each subplot
        for ax in g.axes.flat:
            ax.tick_params(axis='y', pad=15)  # Increase padding
            plt.setp(ax.get_yticklabels(), ha='right')  # Align labels

        # Add a colorbar legend
        norm = plt.Normalize(top_pathways['FDR q-val'].min(), fdr_cutoff)
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])
    
        # Add colorbar to the right of the subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('FDR q-value')
    
        
        g.set_axis_labels('Normalized Enrichment Score', 'Pathway')
        g.figure.suptitle('Top Enriched Pathways by Cell Type', y=1.02)
        g.figure.tight_layout()

        # Adjust layout
        plt.subplots_adjust(
            right=0.9,
            wspace=0.4,
            hspace=0.4
        )
    
        # Return the grid
        return g


class OptimizedCellTypeGSEA(CellTypeGSEA):
    """
    An optimized version of CellTypeGSEA with parallel processing.
    Uses the same interface as CellTypeGSEA but with improved performance.
    """
    
    def __init__(
        self, 
        adata: sc.AnnData,
        cell_type_key: str = 'cell_type',
        gmt_file: str = None,
        cache_dir: str = None
    ):
        """
        Initialize the optimized GSEA analysis object.
        
        Parameters:
        -----------
        adata : AnnData
            Annotated data matrix with cell type annotations
        cell_type_key : str
            Key in adata.obs containing cell type labels
        gmt_file : str
            path to the gene set files (gmt format)
        cache_dir : str
            Directory to cache intermediate results (None = no caching)
        """
        super().__init__(adata, cell_type_key, gmt_file)
        
        # Setup caching
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Caching enabled. Results will be cached in {cache_dir}")
    
    def _get_cache_key(self, cell_type, min_size, max_size, permutations):
        """Generate a unique key for caching based on parameters"""
        # Create a string containing all relevant parameters
        data_hash = hashlib.md5(str(self.adata.shape).encode() + 
                             str(self.adata.obs[self.cell_type_key].value_counts().to_dict()).encode()).hexdigest()
        
        gene_sets_hash = "no_gmt"
        if hasattr(self, 'gene_sets'):
            gene_sets_hash = hashlib.md5(str(sorted(self.gene_sets.keys())).encode()).hexdigest()
        
        params = f"{data_hash}_{gene_sets_hash}_{cell_type}_{min_size}_{max_size}_{permutations}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get the file path for a cached result"""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def compute_cell_type_rankings_optimized(self, cell_type: str) -> pd.Series:
        """
        Optimized version that computes differential expression rankings for one cell type vs all others.
        Uses parallelization for much faster performance.
        
        Parameters:
        -----------
        cell_type : str
            Cell type to analyze
            
        Returns:
        --------
        pd.Series
            Ranked gene list with ranking scores
        """
        # Check cache first if enabled
        if self.cache_dir:
            cache_key = self._get_cache_key(cell_type, 0, 0, 0) + '_rankings'
            cache_path = self._get_cache_path(cache_key)
            
            if os.path.exists(cache_path):
                try:
                    print(f"Loading cached rankings for {cell_type}")
                    with open(cache_path, 'rb') as f:
                        rankings = pickle.load(f)
                    return rankings
                except Exception as e:
                    print(f"Error loading cache: {e}")
        
        # Create binary mask for cell type
        cell_mask = self.adata.obs[self.cell_type_key] == cell_type
        
        # Get expression matrix
        if sparse.issparse(self.adata.X):
            X = self.adata.X.toarray()
        else:
            X = self.adata.X
        
        # Split data by cell type
        group1 = X[cell_mask, :]  # Target cell type
        group2 = X[~cell_mask, :] # All other cells
        
        # Calculate means for log2FC
        mean_1 = np.mean(group1, axis=0)
        mean_2 = np.mean(group2, axis=0)
        
        # Calculate log2 fold change
        epsilon = 1e-10  # To avoid division by zero
        log2fc = np.log2((mean_1 + epsilon) / (mean_2 + epsilon))
        
        # Function to calculate Mann-Whitney U for a single gene
        def mwu_for_gene(i):
            try:
                stat, pval = scipy_stats.mannwhitneyu(
                    group1[:, i], 
                    group2[:, i], 
                    alternative='two-sided'
                )
                return pval
            except:
                return 1.0  # Return 1.0 as p-value if test fails
        
        # Calculate p-values in parallel
        n_jobs = min(8, os.cpu_count() or 1)
        print(f"Calculating rankings in parallel with {n_jobs} jobs...")
        pvals = Parallel(n_jobs=n_jobs)(
            delayed(mwu_for_gene)(i) for i in range(X.shape[1])
        )
        pvals = np.array(pvals)
        
        # Create ranking metric
        min_pval = np.finfo(float).tiny
        pvals = np.maximum(pvals, min_pval)
        ranking_metric = -np.log10(pvals) * np.sign(log2fc)
        
        # Create ranked gene list
        gene_names = self.adata.var_names
        rankings = pd.Series(
            ranking_metric,
            index=gene_names,
            name='ranking'
        ).sort_values(ascending=False)
        
        # Cache the result if caching is enabled
        if self.cache_dir:
            cache_key = self._get_cache_key(cell_type, 0, 0, 0) + '_rankings'
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(rankings, f)
        
        return rankings
    
    def run_gsea_optimized(
        self,
        min_size: int = 15,
        max_size: int = 500,
        permutations: int = 1000,
        threads: int = 4,
        parallel_cell_types: bool = True
    ) -> Dict:
        """
        Optimized version that runs GSEA analysis for all cell types with multiprocessing.
        
        Parameters:
        -----------
        min_size : int
            Minimum gene set size
        max_size : int
            Maximum gene set size
        permutations : int
            Number of permutations
        threads : int
            Number of parallel threads for each GSEA run
        parallel_cell_types : bool
            Whether to process cell types in parallel
        
        Returns:
        --------
        Dict
            Dictionary containing GSEA results for each cell type
        """
        # Get unique cell types
        cell_types = self.adata.obs[self.cell_type_key].unique()
        
        # Define function to process one cell type
        def process_cell_type(cell_type):
            # Check cache first if enabled
            if self.cache_dir:
                cache_key = self._get_cache_key(cell_type, min_size, max_size, permutations)
                cache_path = self._get_cache_path(cache_key)
                
                if os.path.exists(cache_path):
                    try:
                        print(f"Loading cached GSEA results for {cell_type}")
                        with open(cache_path, 'rb') as f:
                            res = pickle.load(f)
                        return cell_type, res
                    except Exception as e:
                        print(f"Error loading cache: {e}")
            
            # Get rankings for this cell type
            try:
                # Use the optimized ranking function
                rankings = self.compute_cell_type_rankings_optimized(cell_type)
                
                # Run GSEA
                start_time = time.time()
                pre_res = gp.prerank(
                    rnk=rankings,
                    gene_sets=self.gene_sets,
                    min_size=min_size,
                    max_size=max_size,
                    permutation_num=permutations,
                    threads=max(1, threads // len(cell_types)) if parallel_cell_types else threads,
                    seed=42,
                    no_plot=True
                )
                
                res = pre_res.res2d
                print(f"GSEA for {cell_type} completed in {time.time() - start_time:.1f} seconds")
                
                # Cache the result if caching is enabled
                if self.cache_dir:
                    cache_key = self._get_cache_key(cell_type, min_size, max_size, permutations)
                    cache_path = self._get_cache_path(cache_key)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(res, f)
                
                return cell_type, res
            except Exception as e:
                print(f"Error processing {cell_type}: {e}")
                # Return empty result for this cell type
                return cell_type, pd.DataFrame()
        
        print(f"Running GSEA analysis for {len(cell_types)} cell types...")
        
        # Check if we should use parallelization
        max_workers = min(len(cell_types), os.cpu_count() or 1) if parallel_cell_types else 1
        
        if parallel_cell_types and max_workers > 1:
            # Process cell types in parallel
            print(f"Using {max_workers} processes for parallel cell type analysis")
            results = {}
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for cell_type, res in executor.map(process_cell_type, cell_types):
                    if not res.empty:
                        results[cell_type] = res
                        print(f"Completed GSEA for {cell_type}")
        else:
            # Process cell types sequentially
            results = {}
            for cell_type in tqdm(cell_types):
                cell_type, res = process_cell_type(cell_type)
                if not res.empty:
                    results[cell_type] = res
        
        # Store results
        self.results = results
        
        return results

    def run_gsea(
        self,
        min_size: int = 15,
        max_size: int = 500,
        permutations: int = 1000,
        threads: int = 4
    ) -> Dict:
        """
        Overrides the base class method to use the optimized version.
        
        Parameters:
        -----------
        min_size : int
            Minimum gene set size
        max_size : int
            Maximum gene set size
        permutations : int
            Number of permutations
        threads : int
            Number of parallel threads
            
        Returns:
        --------
        Dict
            Dictionary containing GSEA results for each cell type
        """
        return self.run_gsea_optimized(min_size, max_size, permutations, threads)


def process_data_for_gsea(adata):
    """
    Process AnnData object for GSEA analysis.
    This typically involves:
    1. Normalizing counts
    2. Log-transforming
    3. Computing highly variable genes (optional)
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
        
    Returns:
    --------
    AnnData
        Processed data
    """
    # Make a copy to avoid modifying the original data
    adata = adata.copy()
    
    # get counts
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    
    # 1. Normalize to 10,000 counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # 2. Log-transform the data
    sc.pp.log1p(adata)
    
    # 3. Identify highly variable genes (optional)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # 4. Store normalized data in a layer
    adata.layers['log1p_cpt'] = adata.X.copy()
    
    # Optional: Compute and store average expression per cell type
    cell_types = adata.obs['cell_type'].unique()
    avg_expr = pd.DataFrame(index=adata.var_names)
    
    for ct in cell_types:
        cells = adata.obs['cell_type'] == ct
        if np.sum(cells) > 0:  # Ensure there are cells of this type
            avg_expr[ct] = np.array(adata[cells].X.mean(axis=0)).flatten()
    
    # Store as annotation
    adata.uns['avg_expr_by_celltype'] = avg_expr
    
    return adata


def process_data_for_gsea_optimized(adata, batch_size=5000, inplace=False):
    """
    Optimized version: Process AnnData object for GSEA analysis with batched operations
    for memory efficiency.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    batch_size : int
        Size of batches for processing large datasets
    inplace : bool
        Whether to modify adata in-place
        
    Returns:
    --------
    AnnData
        Processed data
    """
    start_time = time.time()
    
    if not inplace:
        # Make a copy to avoid modifying the original data
        adata = adata.copy()
    
    # Check if data is backed
    backed = hasattr(adata, 'isbacked') and adata.isbacked
    
    # Determine dataset size
    total_cells = adata.n_obs
    is_large = total_cells > batch_size
    
    print(f"Processing dataset with {total_cells} cells and {adata.n_vars} genes...")
    
    # Normalize data in chunks if backed or large
    if backed or is_large:
        n_batches = (total_cells + batch_size - 1) // batch_size
        
        print(f"Processing data in {n_batches} batches of {batch_size} cells...")
        
        # First compute total counts per cell across all batches
        total_counts = np.zeros(total_cells)
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_cells)
            
            # Get batch data
            batch = adata[start:end]
            batch_data = batch.X.toarray() if sparse.issparse(batch.X) else batch.X
            
            # Calculate sums
            total_counts[start:end] = batch_data.sum(axis=1)
        
        # Store total counts for diagnostics
        adata.obs['n_counts'] = total_counts
        
        # Now normalize batch by batch
        norm_data = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_cells)
            
            # Get batch data
            batch = adata[start:end]
            batch_data = batch.X.toarray() if sparse.issparse(batch.X) else batch.X
            
            # Normalize to 10,000 counts and log1p transform
            scaling_factors = 1e4 / total_counts[start:end, np.newaxis]
            batch_norm = batch_data * scaling_factors
            batch_log = np.log1p(batch_norm)
            
            norm_data.append(batch_log)
        
        # Combine normalized data
        adata.X = np.vstack(norm_data)
    else:
        # For smaller datasets, process all at once
        if sparse.issparse(adata.X):
            adata.X = adata.X.toarray()
        
        # 1. Normalize to 10,000 counts
        sc.pp.normalize_total(adata, target_sum=1e4)
        
        # 2. Log-transform the data
        sc.pp.log1p(adata)
    
    # Compute highly variable genes
    print("Computing highly variable genes...")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Store normalized data in a layer
    adata.layers['log1p_cpt'] = adata.X.copy()
    
    # If needed, calculate cell type averages more efficiently
    cell_type_key = 'cell_type'
    
    if cell_type_key in adata.obs.columns:
        print("Computing average expression per cell type...")
        cell_types = adata.obs[cell_type_key].unique()
        avg_expr = pd.DataFrame(index=adata.var_names)
        
        # Process cell types in parallel
        def process_cell_type(ct):
            cells = np.where(adata.obs[cell_type_key] == ct)[0]
            if len(cells) > 0:
                # Process in batches if many cells of this type
                if len(cells) > batch_size:
                    ct_chunks = [cells[i:i+batch_size] for i in range(0, len(cells), batch_size)]
                    ct_avg = np.zeros(adata.n_vars)
                    for chunk in ct_chunks:
                        chunk_data = adata.X[chunk]
                        if sparse.issparse(chunk_data):
                            chunk_data = chunk_data.toarray()
                        ct_avg += chunk_data.sum(axis=0)
                    return ct, ct_avg / len(cells)
                else:
                    # Small enough to process at once
                    cell_data = adata.X[cells]
                    if sparse.issparse(cell_data):
                        cell_data = cell_data.toarray()
                    return ct, np.array(cell_data.mean(axis=0)).flatten()
            return ct, None
        
        # Only parallelize if we have multiple cores and multiple cell types
        n_jobs = min(8, os.cpu_count() or 1) if len(cell_types) > 1 else 1
        if n_jobs > 1:
            results = Parallel(n_jobs=n_jobs)(delayed(process_cell_type)(ct) for ct in cell_types)
            for ct, avg in results:
                if avg is not None:
                    avg_expr[ct] = avg
        else:
            for ct in cell_types:
                ct, avg = process_cell_type(ct)
                if avg is not None:
                    avg_expr[ct] = avg
        
        # Store as annotation
        adata.uns['avg_expr_by_celltype'] = avg_expr
    
    print(f"Data processing completed in {time.time() - start_time:.2f} seconds")
    
    return adata


def fishers_combined_probability(p_values):
    """
    Fisher's method for combining p-values.
    
    Parameters:
    -----------
    p_values : array-like
        Array of p-values to combine
        
    Returns:
    --------
    tuple
        (chi-square statistic, combined p-value)
    """
    # Ensure p_values are greater than 0 to avoid log(0)
    p_values = np.maximum(np.array(p_values).astype(float), np.finfo(float).tiny)
    
    # Calculate Fisher's statistic - optimized with np.log
    chi_square = -2 * np.sum(np.log(p_values))
    
    # Calculate p-value
    df = 2 * len(p_values)
    p_value = scipy_stats.chi2.sf(chi_square, df)
    
    return chi_square, p_value


def fishers_z_transform(correlations, sample_sizes):
    """
    Fisher's Z-transformation for combining correlation coefficients.
    
    Parameters:
    -----------
    correlations : array-like
        Array of correlation coefficients
    sample_sizes : array-like
        Corresponding sample sizes
        
    Returns:
    --------
    tuple
        (combined correlation, 95% confidence interval)
    """
    # Convert correlations to z-scores (vectorized)
    correlations = np.array(correlations).astype(float)
    # Clip correlations to avoid issues with values near -1 or 1
    correlations = np.clip(correlations, -0.9999, 0.9999)
    
    # Vectorized z-transform
    zs = 0.5 * np.log((1 + correlations) / (1 - correlations))
    
    # Weight by sample sizes
    weights = np.array(sample_sizes) - 3
    weights = np.maximum(weights, 1)  # Ensure weights are at least 1
    
    # Calculate weighted average z
    z_weighted = np.sum(zs * weights) / np.sum(weights)
    
    # Standard error of weighted z
    se_z = 1 / np.sqrt(np.sum(weights))
    
    # 95% confidence interval for z
    z_lower = z_weighted - 1.96 * se_z
    z_upper = z_weighted + 1.96 * se_z
    
    # Convert z back to correlation (vectorized)
    r_combined = (np.exp(2 * z_weighted) - 1) / (np.exp(2 * z_weighted) + 1)
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    return r_combined, (r_lower, r_upper)


def stouffers_z_score(z_scores, weights=None):
    """
    Stouffer's method for combining z-scores.
    
    Parameters:
    -----------
    z_scores : array-like
        Array of z-scores to combine
    weights : array-like, optional
        Weights for the z-scores
        
    Returns:
    --------
    tuple
        (combined z-score, combined p-value)
    """
    z_scores = np.array(z_scores)
    
    if weights is None:
        weights = np.ones_like(z_scores)
    else:
        weights = np.array(weights)
    
    # Calculate weighted sum of z-scores
    z_combined = np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2))
    
    # Calculate p-value
    p_combined = scipy_stats.norm.sf(abs(z_combined)) * 2  # Two-tailed
    
    return z_combined, p_combined


def prepare_results_df(results, suffix=''):
    """
    Organize GSEA results into a dataframe.
    
    Parameters:
    -----------
    results : dict
        Dictionary of GSEA results by cell type
    suffix : str, optional
        Suffix to add to column names
        
    Returns:
    --------
    pd.DataFrame
        Combined results dataframe
    """
    all_results = []
    for cell_type, res in results.items():
        res = res.copy()
        res['cell_type'] = cell_type
        all_results.append(res)
    
    if not all_results:
        return pd.DataFrame()
        
    df = pd.concat(all_results)
    
    # Ensure NES is numeric
    df['NES'] = pd.to_numeric(df['NES'], errors='coerce')
    df['FDR q-val'] = pd.to_numeric(df['FDR q-val'], errors='coerce')
    
    return df


def calculate_pr_curves(merged_df, df_gen, df_real):
    """
    Calculate precision-recall curves for each cell type.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe containing both generated and real results
    df_gen : pd.DataFrame
        Dataframe with results from generated data
    df_real : pd.DataFrame
        Dataframe with results from real data
        
    Returns:
    --------
    dict
        Dictionary containing PR curves for each cell type
    """
    pr_results = {}
    
    # Print statistics about significant pathways in real data
    print("\nNumber of significant pathways (p < 0.1) in real data by cell type:")
    print("-" * 60)
    
    for cell_type in merged_df['cell_type'].unique():
        # Get data for this cell type
        cell_data = merged_df[merged_df['cell_type'] == cell_type].merge(
            df_gen[['Term', 'cell_type', 'FDR q-val']].rename(columns={'FDR q-val': 'pvalue_gen'}),
            on=['Term', 'cell_type']
        ).merge(
            df_real[['Term', 'cell_type', 'FDR q-val']].rename(columns={'FDR q-val': 'pvalue_real'}),
            on=['Term', 'cell_type']
        )
        
        # Create binary labels (1 for significant in real data, 0 for not)
        y_true = (cell_data['pvalue_real'] < 0.1).astype(int)
        
        # Print statistics for this cell type
        n_total = len(y_true)
        n_sig = sum(y_true)
        print(f"{cell_type:<20} {n_sig:>4} / {n_total:<4} significant pathways ({n_sig/n_total*100:.1f}%)")
        
        # Skip cell types with no significant pathways in real data
        if n_sig == 0:
            print(f"Skipping {cell_type} - no significant pathways in real data")
            continue
            
        # Convert p-values to numeric, handling any string values
        pvalues = pd.to_numeric(cell_data['pvalue_gen'], errors='coerce')
        
        # Replace NaN and 0 with smallest positive float
        pvalues = pvalues.fillna(np.finfo(float).tiny)
        pvalues = np.maximum(pvalues, np.finfo(float).tiny)
        
        # Calculate -log10 scores
        scores = -np.log10(pvalues.astype(float))
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        
        # Calculate average precision
        ap = average_precision_score(y_true, scores)
        
        # Store results
        pr_results[cell_type] = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': ap,
            'n_significant': n_sig,
            'n_total': n_total
        }
        
        # Calculate actual precision and recall at p < 0.1
        pred = (pvalues < 0.1).astype(int)
        true_pos = np.sum((pred == 1) & (y_true == 1))
        false_pos = np.sum((pred == 1) & (y_true == 0))
        false_neg = np.sum((pred == 0) & (y_true == 1))
        
        precision_at_01 = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall_at_01 = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        
        pr_results[cell_type]['precision_at_01'] = precision_at_01
        pr_results[cell_type]['recall_at_01'] = recall_at_01

    return pr_results


def plot_pr_curves(pr_results, output_path=None):
    """
    Plot precision-recall curves for each cell type.
    
    Parameters:
    -----------
    pr_results : dict
        Dictionary containing PR curves for each cell type
    output_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if len(pr_results) == 0:
        print("\nNo cell types had significant pathways in the real data.")
        return None
        
    # Set the style to white background
    plt.style.use('default')
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Define a custom color palette (you can adjust these colors)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot individual PR curves
    for i, (cell_type, results) in enumerate(pr_results.items()):
        color = colors[i % len(colors)]
        
        # Plot the curve with higher alpha
        ax.plot(results['recall'], results['precision'], 
               label=f"{cell_type} (AP={results['average_precision']:.2f}, n={results['n_significant']})",
               color=color, alpha=0.8, linewidth=2)
        
        # Plot point at p < 0.1 with larger size
        ax.scatter(results['recall_at_01'], results['precision_at_01'], 
                  color=color, marker='o', s=120, zorder=5)

    # Plot random baseline with subtle style
    ax.plot([0, 1], [0.5, 0.5], '--', color='#cccccc', alpha=0.5, 
            label='Random baseline', linewidth=1.5)

    # Customize axis appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Customize ticks
    ax.tick_params(axis='both', colors='#333333')
    
    # Add labels with custom font size
    ax.set_xlabel('Recall', fontsize=12, color='#333333')
    ax.set_ylabel('Precision', fontsize=12, color='#333333')
    ax.set_title('Precision-Recall Curves by Cell Type', 
                fontsize=14, color='#333333', pad=20)

    # Customize axes limits with small padding
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Customize legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, fancybox=True, framealpha=1,
             edgecolor='none', fontsize=10)

    # Adjust layout to prevent legend cropping
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path is not None:
        plt.savefig(output_path, 
                    bbox_inches='tight', dpi=400, 
                    facecolor='white', edgecolor='none')
    
    return fig


def run_comparison_analysis(adataGen, adataReal, gmt_file, output_dir=None, 
                            save_toppathways=False, min_size=15, max_size=500, 
                            permutations=1000, threads=4, use_optimization=True,
                            cache_dir=None, batch_size=5000):
    """
    Run a complete GSEA comparison analysis between generated and real data.
    
    Parameters:
    -----------
    adataGen : AnnData
        Annotated data matrix with generated data
    adataReal : AnnData
        Annotated data matrix with real data
    gmt_file : str
        Path to the gene set files (gmt format)
    output_dir : str, optional
        Directory to save results
    save_toppathways : bool
        Whether to save detailed top pathway plots
    min_size : int
        Minimum gene set size
    max_size : int
        Maximum gene set size
    permutations : int
        Number of permutations
    threads : int
        Number of parallel threads
    use_optimization : bool
        Whether to use optimized functions
    cache_dir : str, optional
        Directory to cache results (None = no caching)
    batch_size : int
        Batch size for processing large datasets
        
    Returns:
    --------
    tuple
        (gsea_analyzer_gen, gsea_analyzer_real, df_gen, df_real, merged_df)
    """
    total_start_time = time.time()
    
    # Ensure both AnnData objects have cell_type column
    if 'cell_type' not in adataGen.obs.columns:
        if 'myannotations' in adataGen.obs.columns:
            adataGen.obs['cell_type'] = adataGen.obs['myannotations'].copy()
        else:
            raise ValueError("Generated data doesn't have 'cell_type' or 'myannotations' column")
            
    if 'cell_type' not in adataReal.obs.columns:
        if 'myannotations' in adataReal.obs.columns:
            adataReal.obs['cell_type'] = adataReal.obs['myannotations'].copy()
        else:
            raise ValueError("Real data doesn't have 'cell_type' or 'myannotations' column")
    
    # Process data for GSEA
    start_time = time.time()
    print("Processing generated data...")
    
    if use_optimization:
        adataGen_processed = process_data_for_gsea_optimized(adataGen, batch_size=batch_size)
    else:
        adataGen_processed = process_data_for_gsea(adataGen)
    
    print("Processing real data...")
    if use_optimization:
        adataReal_processed = process_data_for_gsea_optimized(adataReal, batch_size=batch_size)
    else:
        adataReal_processed = process_data_for_gsea(adataReal)
    
    preprocess_time = time.time() - start_time
    print(f"Data preprocessing completed in {preprocess_time:.2f} seconds")
    
    # Setup cache directories if enabled
    if cache_dir and output_dir:
        cache_base = os.path.join(output_dir, cache_dir)
        gen_cache_dir = os.path.join(cache_base, "generated")
        real_cache_dir = os.path.join(cache_base, "real")
        os.makedirs(gen_cache_dir, exist_ok=True)
        os.makedirs(real_cache_dir, exist_ok=True)
    else:
        gen_cache_dir = None
        real_cache_dir = None
    
    # Print cell type stats
    gen_types = set(adataGen_processed.obs['cell_type'])
    real_types = set(adataReal_processed.obs['cell_type'])
    common_types = gen_types.intersection(real_types)
    
    print("\nCell type stats:")
    print(f"Generated cell types: {len(gen_types)}")
    print(f"Real data cell types: {len(real_types)}")
    print(f"Common cell types: {len(common_types)}")
    
    # Initialize GSEA analyzers
    print("\nInitializing GSEA analyzers...")
    start_time = time.time()
    
    if use_optimization:
        # Use optimized GSEA class
        gsea_analyzer_gen = OptimizedCellTypeGSEA(
            adataGen_processed,
            cell_type_key='cell_type',
            gmt_file=gmt_file,
            cache_dir=gen_cache_dir
        )
        
        gsea_analyzer_real = OptimizedCellTypeGSEA(
            adataReal_processed,
            cell_type_key='cell_type',
            gmt_file=gmt_file,
            cache_dir=real_cache_dir
        )
    else:
        # Use standard GSEA class
        gsea_analyzer_gen = CellTypeGSEA(
            adataGen_processed,
            cell_type_key='cell_type',
            gmt_file=gmt_file
        )
        
        gsea_analyzer_real = CellTypeGSEA(
            adataReal_processed,
            cell_type_key='cell_type',
            gmt_file=gmt_file
        )
    
    # Run GSEA, splitting available threads between the two analyses
    threads_per_analyzer = max(1, threads // 2)
    print(f"\nRunning GSEA with {threads_per_analyzer} threads per dataset...")
    
    # Run in parallel if we have enough threads
    if threads >= 2 and use_optimization:
        # Run the GSEAs in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit both GSEA jobs
            gen_future = executor.submit(
                gsea_analyzer_gen.run_gsea_optimized,
                min_size=min_size,
                max_size=max_size,
                permutations=permutations,
                threads=threads_per_analyzer
            )
            
            real_future = executor.submit(
                gsea_analyzer_real.run_gsea_optimized,
                min_size=min_size,
                max_size=max_size,
                permutations=permutations,
                threads=threads_per_analyzer
            )
            
            # Wait for both to complete
            print("Running GSEA on generated data...")
            resultsGen = gen_future.result()
            
            print("Running GSEA on real data...")
            resultsReal = real_future.result()
    else:
        # Run sequentially
        print("\nRunning GSEA on generated data...")
        if use_optimization:
            resultsGen = gsea_analyzer_gen.run_gsea_optimized(
                min_size=min_size,
                max_size=max_size,
                permutations=permutations,
                threads=threads
            )
        else:
            resultsGen = gsea_analyzer_gen.run_gsea(
                min_size=min_size,
                max_size=max_size,
                permutations=permutations,
                threads=threads
            )
        
        print("\nRunning GSEA on real data...")
        if use_optimization:
            resultsReal = gsea_analyzer_real.run_gsea_optimized(
                min_size=min_size,
                max_size=max_size,
                permutations=permutations,
                threads=threads
            )
        else:
            resultsReal = gsea_analyzer_real.run_gsea(
                min_size=min_size,
                max_size=max_size,
                permutations=permutations,
                threads=threads
            )
    
    gsea_time = time.time() - start_time
    print(f"GSEA analysis completed in {gsea_time:.2f} seconds")
    
    # Only create plots if requested
    if save_toppathways:
        start_time = time.time()
        print("\nPlotting top pathways for generated data...")
        gG = gsea_analyzer_gen.plot_top_pathways()
        
        print("\nPlotting top pathways for real data...")
        gR = gsea_analyzer_real.plot_top_pathways()
        
        # Save plots if output directory is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            gG.figure.savefig(os.path.join(output_dir, 'gsea_gen_results.pdf'), bbox_inches='tight', dpi=300)
            gR.figure.savefig(os.path.join(output_dir, 'gsea_real_results.pdf'), bbox_inches='tight', dpi=300)
            
            # Save results to CSV
            for cell_type, res in resultsGen.items():
                res.to_csv(os.path.join(output_dir, f'gsea_gen_results_{cell_type}.csv'))
            for cell_type, res in resultsReal.items():
                res.to_csv(os.path.join(output_dir, f'gsea_real_results_{cell_type}.csv'))
        
        plotting_time = time.time() - start_time
        print(f"Pathway plotting completed in {plotting_time:.2f} seconds")
    
    # Prepare dataframes for comparison
    start_time = time.time()
    df_gen = prepare_results_df(resultsGen, '_gen')
    df_real = prepare_results_df(resultsReal, '_real')
    
    # Merge the dataframes on cell type and pathway term
    merged_df = pd.merge(
        df_gen[['Term', 'cell_type', 'NES']].rename(columns={'NES': 'NES_gen'}),
        df_real[['Term', 'cell_type', 'NES']].rename(columns={'NES': 'NES_real'}),
        on=['Term', 'cell_type'],
        how='inner'
    )
    
    print(f"\nMerged {len(merged_df)} pathways across {len(merged_df['cell_type'].unique())} cell types")
    
    # Save merged results if output directory is provided
    if output_dir is not None:
        merged_df.to_csv(os.path.join(output_dir, 'merged_gsea_results.csv'))
    
    # Calculate PR curves
    pr_results = calculate_pr_curves(merged_df, df_gen, df_real)
    
    # Plot PR curves
    if output_dir is not None:
        plot_pr_curves(pr_results, os.path.join(output_dir, 'precision_recall_curves.pdf'))
    else:
        plot_pr_curves(pr_results)
    
    analysis_time = time.time() - start_time
    print(f"Results analysis completed in {analysis_time:.2f} seconds")
    
    # Print total execution time
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"  - Preprocessing: {preprocess_time:.2f} seconds ({preprocess_time/total_time*100:.1f}%)")
    print(f"  - GSEA analysis: {gsea_time:.2f} seconds ({gsea_time/total_time*100:.1f}%)")
    
    if save_toppathways:
        print(f"  - Pathway plotting: {plotting_time:.2f} seconds ({plotting_time/total_time*100:.1f}%)")
    
    print(f"  - Results analysis: {analysis_time:.2f} seconds ({analysis_time/total_time*100:.1f}%)")
    
    # Return results
    return gsea_analyzer_gen, gsea_analyzer_real, df_gen, df_real, merged_df


def analyze_gsea_correlation(merged_df, df_gen, df_real, output_dir=None):
    """
    Analyze the correlation between GSEA results from generated and real data.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe containing both generated and real results
    df_gen : pd.DataFrame
        Dataframe with results from generated data
    df_real : pd.DataFrame
        Dataframe with results from real data
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    tuple
        (correlation_results, combined_stats_df)
    """
    # Perform statistical test for each cell type
    statistical_results = []
    for cell_type in merged_df['cell_type'].unique():
        cell_data = merged_df[merged_df['cell_type'] == cell_type]
        # Perform paired t-test
        t_stat, p_val = scipy_stats.ttest_rel(
            cell_data['NES_gen'],
            cell_data['NES_real']
        )
        # Calculate correlation
        corr, corr_p = scipy_stats.pearsonr(
            cell_data['NES_gen'],
            cell_data['NES_real']
        )
        statistical_results.append({
            'cell_type': cell_type,
            'n_pathways': len(cell_data),
            't_statistic': t_stat,
            'p_value': p_val,
            'correlation': corr,
            'correlation_p': corr_p
        })

    # Convert results to dataframe
    results_df = pd.DataFrame(statistical_results)

    # Add multiple testing correction
    _, results_df['p_value_adj'] = fdrcorrection(results_df['p_value'])
    _, results_df['correlation_p_adj'] = fdrcorrection(results_df['correlation_p'])

    # Calculate combined statistics across all cell types
    # 1. Fisher's combined probability for t-test p-values
    chi_square_t, combined_p_t = fishers_combined_probability(results_df['p_value'])

    # 2. Fisher's Z-transformation for correlations
    correlations = results_df['correlation'].values
    sample_sizes = results_df['n_pathways'].values
    combined_r, ci_r = fishers_z_transform(correlations, sample_sizes)

    # 3. Convert t-statistics to z-scores and use Stouffer's method
    # Note: for t-statistics, we can convert to z-scores using the survival function
    z_scores = scipy_stats.norm.ppf(scipy_stats.t.sf(abs(results_df['t_statistic']), 
                                       df=results_df['n_pathways']-1))
    weights = np.sqrt(results_df['n_pathways'])  # weight by sqrt of sample size
    combined_z, combined_p_z = stouffers_z_score(z_scores, weights)

    # Add combined statistics to results summary
    print("\nCombined Statistics Across Cell Types:")
    print(f"Fisher's Combined P-value (t-tests): {combined_p_t:.2e}")
    print(f"Combined Correlation (Fisher's Z): {combined_r:.3f} (95% CI: {ci_r[0]:.3f}, {ci_r[1]:.3f})")
    print(f"Stouffer's Combined Z-score: {combined_z:.3f} (p = {combined_p_z:.2e})")

    # Sort by p-value
    results_df = results_df.sort_values('p_value')

    print("\nStatistical comparison results by cell type:")
    print(results_df.to_string(float_format=lambda x: '{:.2e}'.format(x) if isinstance(x, float) else str(x)))
    
    # Create combined stats DataFrame
    combined_stats = pd.DataFrame({
        'statistic': ['Combined t-test p-value', 'Combined correlation', 'Combined correlation CI lower', 
                     'Combined correlation CI upper', 'Combined Z-score', 'Combined Z-score p-value'],
        'value': [combined_p_t, combined_r, ci_r[0], ci_r[1], combined_z, combined_p_z]
    })
    
    # Save results if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        merged_df.to_csv(os.path.join(output_dir, 'NES_both_datasets.csv'))
        results_df.to_csv(os.path.join(output_dir, 'statistical_results_comparison.csv'))
        combined_stats.to_csv(os.path.join(output_dir, 'combined_statistics.csv'))
    
    return results_df, combined_stats


def plot_gsea_correlation(merged_df, df_gen, df_real, output_dir=None):
    """
    Plot the correlation between GSEA results from generated and real data.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe containing both generated and real results
    df_gen : pd.DataFrame
        Dataframe with results from generated data
    df_real : pd.DataFrame
        Dataframe with results from real data
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    tuple
        (figure for cell type plots, figure for combined plot)
    """
    # First, ensure df_gen and df_real have numeric FDR q-val columns
    for df in [df_gen, df_real]:
        df['FDR q-val'] = pd.to_numeric(df['FDR q-val'], errors='coerce')
        # Replace NaN with maximum value
        df['FDR q-val'].fillna(1.0, inplace=True)
    
    # Create subplot grid
    n_cell_types = len(merged_df['cell_type'].unique())
    n_cols = min(3, n_cell_types)
    n_rows = (n_cell_types + n_cols - 1) // n_cols

    # Increase figure width to accommodate legend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))  # Increased width
    if n_rows * n_cols == 1:  # Handle case with only one subplot
        axes = np.array([axes])
    axes = axes.flatten()

    # Define colors for different significance categories
    color_map = {
        'both': 'blue',
        'real_only': 'green',
        'gen_only': 'orange',
        'none': 'gray'
    }

    # Set axis limits
    AXIS_MIN = -4
    AXIS_MAX = 4

    for idx, cell_type in enumerate(merged_df['cell_type'].unique()):
        cell_data = merged_df[merged_df['cell_type'] == cell_type]
        
        # Add p-values from both datasets
        cell_data = cell_data.merge(
            df_gen[['Term', 'cell_type', 'FDR q-val']].rename(columns={'FDR q-val': 'pvalue_gen'}),
            on=['Term', 'cell_type']
        )
        cell_data = cell_data.merge(
            df_real[['Term', 'cell_type', 'FDR q-val']].rename(columns={'FDR q-val': 'pvalue_real'}),
            on=['Term', 'cell_type']
        )
        
        # Categorize points based on significance
        cell_data['significance'] = 'none'
        cell_data.loc[(cell_data['pvalue_gen'] < 0.1) & (cell_data['pvalue_real'] < 0.1), 'significance'] = 'both'
        cell_data.loc[(cell_data['pvalue_real'] < 0.1) & (cell_data['pvalue_gen'] >= 0.1), 'significance'] = 'real_only'
        cell_data.loc[(cell_data['pvalue_gen'] < 0.1) & (cell_data['pvalue_real'] >= 0.1), 'significance'] = 'gen_only'
        
        # Calculate counts for each category
        sig_counts = cell_data['significance'].value_counts()
        
        # Calculate correlations for different categories
        correlations = {}
        for category in ['both', 'real_only', 'all']:
            if category == 'all':
                cat_data = cell_data
            else:
                cat_data = cell_data[cell_data['significance'] == category]
            
            if len(cat_data) > 1:
                corr, _ = scipy_stats.pearsonr(cat_data['NES_gen'], cat_data['NES_real'])
                correlations[category] = corr
            else:
                correlations[category] = float('nan')
        
        # Plot points for each category
        for category, color in color_map.items():
            cat_data = cell_data[cell_data['significance'] == category]
            if len(cat_data) > 0:  # Only plot if we have data for this category
                alpha = 0.5 if category == 'none' else 1.0
                
                sns.scatterplot(
                    data=cat_data,
                    x='NES_gen',
                    y='NES_real',
                    color=color,
                    alpha=alpha,
                    ax=axes[idx],
                    label=f"{category} (n={len(cat_data)})"
                )
        
        # Add correlation lines for significant categories
        for category, color in {'both': 'blue', 'real_only': 'green'}.items():
            cat_data = cell_data[cell_data['significance'] == category]
            if len(cat_data) > 1:
                sns.regplot(
                    data=cat_data,
                    x='NES_gen',
                    y='NES_real',
                    scatter=False,
                    ax=axes[idx],
                    color=color,
                    line_kws={'linestyle': '-' if category == 'both' else '--'}
                )
        
        # Update title to show correlations
        title_lines = [cell_type]
        if 'all' in correlations:
            title_lines.append(f"All r={correlations['all']:.2f}")
        if 'both' in correlations and not np.isnan(correlations['both']):
            title_lines.append(f"Both sig r={correlations['both']:.2f}")
        if 'real_only' in correlations and not np.isnan(correlations['real_only']):
            title_lines.append(f"Real only r={correlations['real_only']:.2f}")
        
        axes[idx].set_title('\n'.join(title_lines))
        axes[idx].set_xlabel('Generated NES')
        axes[idx].set_ylabel('Real NES')
        
        # Set axis limits
        axes[idx].set_xlim(AXIS_MIN, AXIS_MAX)
        axes[idx].set_ylim(AXIS_MIN, AXIS_MAX)
        
        # Add diagonal line within the limited range
        axes[idx].plot([AXIS_MIN, AXIS_MAX], [AXIS_MIN, AXIS_MAX], '--', color='gray', alpha=0.5)
        
        # Add legend with adjusted position
        if len(cell_data) > 0:  # Only add legend if we have data
            axes[idx].legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    # Remove empty subplots
    for idx in range(n_cell_types, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout with extra space for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # Leave more space on the right for legends

    # Save cell type plots if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'gsea_comparison_byCellType.pdf'), 
                    bbox_inches='tight', dpi=400)

    # Create a new figure for the combined plot
    plt.figure(figsize=(12, 10))

    # Process all data together
    combined_data = merged_df.copy()

    # Add p-values from both datasets
    combined_data = combined_data.merge(
        df_gen[['Term', 'cell_type', 'FDR q-val']].rename(columns={'FDR q-val': 'pvalue_gen'}),
        on=['Term', 'cell_type']
    )
    combined_data = combined_data.merge(
        df_real[['Term', 'cell_type', 'FDR q-val']].rename(columns={'FDR q-val': 'pvalue_real'}),
        on=['Term', 'cell_type']
    )

    # Categorize points based on significance
    combined_data['significance'] = 'none'
    combined_data.loc[(combined_data['pvalue_gen'] < 0.1) & (combined_data['pvalue_real'] < 0.1), 'significance'] = 'both'
    combined_data.loc[(combined_data['pvalue_real'] < 0.1) & (combined_data['pvalue_gen'] >= 0.1), 'significance'] = 'real_only'
    combined_data.loc[(combined_data['pvalue_gen'] < 0.1) & (combined_data['pvalue_real'] >= 0.1), 'significance'] = 'gen_only'

    # Calculate correlations for different categories
    correlations = {}
    for category in ['both', 'real_only', 'all']:
        if category == 'all':
            cat_data = combined_data
        else:
            cat_data = combined_data[combined_data['significance'] == category]
        
        if len(cat_data) > 1:
            corr, _ = scipy_stats.pearsonr(cat_data['NES_gen'], cat_data['NES_real'])
            correlations[category] = corr
        else:
            correlations[category] = float('nan')

    # Plot points for each category
    for category, color in color_map.items():
        cat_data = combined_data[combined_data['significance'] == category]
        if len(cat_data) > 0:  # Only plot if we have data for this category
            alpha = 0.5 if category == 'none' else 1.0
            
            plt.scatter(
                cat_data['NES_gen'],
                cat_data['NES_real'],
                color=color,
                alpha=alpha,
                label=f"{category} (n={len(cat_data)})"
            )

    # Add correlation lines for significant categories
    for category, color in {'both': 'blue', 'real_only': 'green', 'all': 'purple'}.items():
        cat_data = combined_data[combined_data['significance'] == category] if category != 'all' else combined_data
        if len(cat_data) > 1:
            z = np.polyfit(cat_data['NES_gen'], cat_data['NES_real'], 1)
            p = np.poly1d(z)
            x_range = np.array([-4, 4])
            linestyle = '-' if category == 'both' else ('--' if category == 'real_only' else '-.')
            plt.plot(x_range, p(x_range), color=color, 
                    linestyle=linestyle,
                    label=f"{category} fit (r={correlations.get(category, 0):.2f})")

    # Set axis limits
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    # Add diagonal line
    plt.plot([-4, 4], [-4, 4], '--', color='gray', alpha=0.5)

    # Labels and title
    plt.xlabel('Generated NES', fontsize=12)
    plt.ylabel('Real NES', fontsize=12)
    title_lines = ['Combined Cell Types']
    if 'all' in correlations:
        title_lines.append(f"All data r={correlations['all']:.2f}")
    if 'both' in correlations and not np.isnan(correlations['both']):
        title_lines.append(f"Both sig r={correlations['both']:.2f}")
    if 'real_only' in correlations and not np.isnan(correlations['real_only']):
        title_lines.append(f"Real only r={correlations['real_only']:.2f}")
    plt.title('\n'.join(title_lines), fontsize=14)

    # Add legend
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save combined plot if output_dir is provided
    combined_fig = plt.gcf()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'gsea_comparison_combined.pdf'), 
                    bbox_inches='tight', dpi=400)

    # Print some summary statistics
    print("\nSummary Statistics:")
    for category in ['all', 'both', 'real_only', 'gen_only', 'none']:
        if category == 'all':
            count = len(combined_data)
        else:
            count = len(combined_data[combined_data['significance'] == category])
        percentage = (count / len(combined_data)) * 100 if len(combined_data) > 0 else 0
        print(f"{category}: {count} pathways ({percentage:.1f}%)")
        
        if category in correlations and not np.isnan(correlations[category]):
            print(f"Correlation for {category}: {correlations[category]:.3f}")
        print()
    
    # Return both figures
    return fig, combined_fig


def process_data_for_gsea_memory_efficient(adata, batch_size=1000, use_temp_files=True, temp_dir=None):
    """
    Memory-efficient version: Process AnnData object for GSEA analysis with careful memory management.
    """
    import gc
    import tempfile
    import os
    
    print(f"Processing dataset with memory-efficient approach...")
    print(f"Dataset: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Create a copy with only the necessary data
    if use_temp_files and temp_dir:
        # Create temporary directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, 'temp_anndata.h5ad')
        print(f"Using temporary file: {temp_file}")
        
        # Create a proper DataFrame for var
        var_df = pd.DataFrame(index=adata.var_names)
        
        # Save a basic version of the AnnData object
        basic_adata = ad.AnnData(
            X=adata.X.copy() if not sparse.issparse(adata.X) else adata.X,
            obs=adata.obs[['cell_type']].copy(),
            var=var_df  # Using DataFrame instead of Index
        )
        basic_adata.write_h5ad(temp_file, compression='gzip')
        
        # Clear memory
        del basic_adata
        gc.collect()
        
        # Reload as backed mode
        adata_proc = ad.read_h5ad(temp_file, backed='r')
    else:
        # Create a minimal copy to save memory
        var_df = pd.DataFrame(index=adata.var_names)
        
        adata_proc = ad.AnnData(
            X=adata.X.copy() if not sparse.issparse(adata.X) else adata.X,
            obs=adata.obs[['cell_type']].copy(),
            var=var_df  # Using DataFrame instead of Index
        )
    
    # Calculate total counts cell by cell to save memory
    print("Calculating total counts per cell...")
    total_cells = adata_proc.n_obs
    total_counts = np.zeros(total_cells)
    
    for i in range(0, total_cells, batch_size):
        # Process in small batches
        end = min(i + batch_size, total_cells)
        
        # Get batch data
        if sparse.issparse(adata_proc.X):
            batch_data = adata_proc[i:end].X.toarray()
        else:
            batch_data = adata_proc[i:end].X
        
        # Calculate sums and store - FIX FOR BROADCASTING ISSUE
        sums = np.sum(batch_data, axis=1)
        # If sums is multi-dimensional (e.g., shape (1000,1) instead of (1000,)), flatten it
        if sums.ndim > 1:
            sums = sums.flatten()
            
        total_counts[i:end] = sums
        
        # Clear memory after each batch
        del batch_data
        gc.collect()
    
    # Store total counts
    adata_proc.obs['n_counts'] = total_counts
    
    # Create output matrix
    if use_temp_files and temp_dir:
        # Save normalized data to temp files to avoid memory issues
        norm_temp_file = os.path.join(temp_dir, 'normalized_data.npy')
        
        # Open a memory-mapped array for writing
        shape = (adata_proc.n_obs, adata_proc.n_vars)
        norm_data = np.memmap(norm_temp_file, dtype='float32', mode='w+', shape=shape)
    else:
        # Create in memory if not using temp files
        norm_data = np.zeros((adata_proc.n_obs, adata_proc.n_vars), dtype='float32')
    
    # Normalize batch by batch
    print("Normalizing and log-transforming data in batches...")
    for i in range(0, total_cells, batch_size):
        # Process in small batches
        end = min(i + batch_size, total_cells)
        
        # Get batch data
        if sparse.issparse(adata_proc.X):
            batch_data = adata_proc[i:end].X.toarray()
        else:
            batch_data = adata_proc[i:end].X
        
        # Normalize to 10,000 counts
        batch_totals = total_counts[i:end, np.newaxis]
        batch_norm = batch_data * (1e4 / batch_totals)
        
        # Log transform
        batch_log = np.log1p(batch_norm)
        
        # Store in output matrix
        norm_data[i:end] = batch_log
        
        # Clear memory after each batch
        del batch_data, batch_norm, batch_log
        gc.collect()
    
    # Create a new AnnData with normalized data
    if use_temp_files and temp_dir:
        # Create a proper DataFrame for var
        var_df = pd.DataFrame(index=adata_proc.var_names)
        
        # Create a new AnnData with the memory-mapped data
        adata_norm = ad.AnnData(
            X=norm_data,
            obs=adata_proc.obs.copy(),
            var=var_df  # Using DataFrame instead of Index
        )
        
        # Save to file
        norm_h5ad_file = os.path.join(temp_dir, 'normalized_anndata.h5ad')
        adata_norm.write_h5ad(norm_h5ad_file, compression='gzip')
        
        # Close the memmap
        if isinstance(norm_data, np.memmap):
            norm_data._mmap.close()
        del norm_data
        
        # Reload the normalized data
        adata_final = ad.read_h5ad(norm_h5ad_file)
    else:
        # Create a proper DataFrame for var
        var_df = pd.DataFrame(index=adata_proc.var_names)
        
        # Create new AnnData in memory
        adata_final = ad.AnnData(
            X=norm_data,
            obs=adata_proc.obs.copy(),
            var=var_df  # Using DataFrame instead of Index
        )
    
    # Compute average expression by cell type with minimal memory
    print("Computing average expression by cell type...")
    cell_types = adata_final.obs['cell_type'].unique()
    avg_expr = pd.DataFrame(index=adata_final.var_names)
    
    for ct in cell_types:
        # Get indices for this cell type
        cell_indices = np.where(adata_final.obs['cell_type'] == ct)[0]
        ct_mean = np.zeros(adata_final.n_vars)
        
        # Calculate mean in batches
        for i in range(0, len(cell_indices), batch_size):
            idx_batch = cell_indices[i:i+batch_size]
            if len(idx_batch) > 0:
                # Get batch data and calculate sum - FIX FOR POTENTIAL BROADCASTING ISSUES
                batch_sum = np.sum(adata_final.X[idx_batch], axis=0)
                if batch_sum.ndim > 1:
                    batch_sum = batch_sum.flatten()
                ct_mean += batch_sum
        
        # Divide by total number of cells
        ct_mean = ct_mean / len(cell_indices) if len(cell_indices) > 0 else ct_mean
        avg_expr[ct] = ct_mean
    
    # Store the average expression
    adata_final.uns['avg_expr_by_celltype'] = avg_expr
    
    # Clean up temporary files if used
    if use_temp_files and temp_dir:
        try:
            os.remove(temp_file)
            os.remove(norm_temp_file)
            os.remove(norm_h5ad_file)
            print("Temporary files cleaned up")
        except:
            print("Warning: Could not clean up some temporary files")
    
    # Final cleanup
    gc.collect()
    
    return adata_final


class MemoryEfficientCellTypeGSEA(CellTypeGSEA):
    """
    Memory-efficient version of CellTypeGSEA that limits parallel processing
    and manages memory carefully.
    """
    
    def compute_cell_type_rankings(self, cell_type: str) -> pd.Series:
        """
        Memory-efficient implementation that computes rankings one gene at a time
        to avoid memory spikes.
        """
        import gc
        
        # Create binary mask for cell type
        cell_mask = self.adata.obs[self.cell_type_key] == cell_type
        
        # Initialize results storage
        n_genes = self.adata.n_vars
        scores = np.zeros(n_genes)
        pvals = np.zeros(n_genes)
        
        # Process in smaller batches for large datasets
        batch_size = 100  # Process genes in batches of 100
        
        # Get expression matrix - avoid full conversion for sparse matrices
        if sparse.issparse(self.adata.X):
            # This assumes we're processing one gene at a time below
            # so we don't need to convert the entire matrix
            X = self.adata.X
            is_sparse = True
        else:
            X = self.adata.X
            is_sparse = False
        
        # Pre-calculate masks to avoid recalculating
        cell_type_indices = np.where(cell_mask)[0]
        other_indices = np.where(~cell_mask)[0]
        
        for batch_start in range(0, n_genes, batch_size):
            batch_end = min(batch_start + batch_size, n_genes)
            
            for i in range(batch_start, batch_end):
                # Extract gene expression - handle sparse matrices efficiently
                if is_sparse:
                    # Extract a single gene from sparse matrix
                    gene_expr = X[:, i].toarray().flatten()
                else:
                    gene_expr = X[:, i]
                
                # Use pre-calculated indices
                gene_expr_cell_type = gene_expr[cell_type_indices]
                gene_expr_others = gene_expr[other_indices]
                
                # Perform Mann-Whitney U test
                try:
                    stat, pval = scipy_stats.mannwhitneyu(
                        gene_expr_cell_type,
                        gene_expr_others,
                        alternative='two-sided'
                    )
                except:
                    pval = 1.0  # Default p-value if test fails
                
                # Compute effect size (log2 fold change)
                mean_1 = np.mean(gene_expr_cell_type)
                mean_2 = np.mean(gene_expr_others)
                log2fc = np.log2((mean_1 + 1e-10) / (mean_2 + 1e-10))
                
                scores[i] = log2fc
                pvals[i] = pval
                
                # Clear gene expression data
                del gene_expr, gene_expr_cell_type, gene_expr_others
            
            # Run garbage collection periodically
            if batch_end % 500 == 0:
                gc.collect()
        
        # Create ranking metric
        # Add a small epsilon to p-values to avoid log10(0)
        min_pval = np.finfo(float).tiny  # Smallest positive float
        pvals = np.maximum(pvals, min_pval)
        ranking_metric = -np.log10(pvals) * np.sign(scores)
        
        # Create ranked gene list
        gene_names = [f'gene_{i}' for i in range(n_genes)] if self.adata.var_names.empty else self.adata.var_names
        rankings = pd.Series(
            ranking_metric,
            index=gene_names,
            name='ranking'
        ).sort_values(ascending=False)
        
        return rankings
    
    def run_gsea(
        self,
        min_size: int = 15,
        max_size: int = 500,
        permutations: int = 1000,
        threads: int = 1  # Default to single-threaded to save memory
    ) -> Dict:
        """
        Run GSEA analysis for all cell types sequentially to limit memory usage.
        """
        # Get unique cell types
        cell_types = self.adata.obs[self.cell_type_key].unique()
        
        print("Running memory-efficient GSEA analysis for each cell type...")
        
        # Limit threads to avoid memory issues
        threads = min(threads, 2)  # Cap at 2 threads to prevent memory problems
        
        for cell_type in tqdm(cell_types):
            # Get rankings for this cell type
            rankings = self.compute_cell_type_rankings(cell_type)
            
            # Run GSEA with limited resources
            pre_res = gp.prerank(
                rnk=rankings,
                gene_sets=self.gene_sets,
                min_size=min_size,
                max_size=max_size,
                permutation_num=permutations,
                threads=threads,
                seed=42,
                no_plot=True
            )
            
            # Store results
            self.results[cell_type] = pre_res.res2d
            
            # Force garbage collection after each cell type
            import gc
            gc.collect()
            
        return self.results


def run_memory_efficient_analysis(adataGen, adataReal, gmt_file, output_dir=None, 
                                 save_toppathways=False, min_size=15, max_size=500, 
                                 permutations=1000, threads=1):
    """
    Run a memory-efficient GSEA comparison analysis between generated and real data.
    """
    import os
    import time
    import gc
    
    # Create a temporary directory for memory-mapped files
    if output_dir:
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
    else:
        import tempfile
        temp_dir = tempfile.mkdtemp()
    
    print(f"Using temporary directory: {temp_dir}")
    
    total_start_time = time.time()
    
    # Ensure both AnnData objects have cell_type column
    if 'cell_type' not in adataGen.obs.columns:
        if 'myannotations' in adataGen.obs.columns:
            adataGen.obs['cell_type'] = adataGen.obs['myannotations'].copy()
        else:
            raise ValueError("Generated data doesn't have 'cell_type' or 'myannotations' column")
            
    if 'cell_type' not in adataReal.obs.columns:
        if 'myannotations' in adataReal.obs.columns:
            adataReal.obs['cell_type'] = adataReal.obs['myannotations'].copy()
        else:
            raise ValueError("Real data doesn't have 'cell_type' or 'myannotations' column")
    
    # Process data for GSEA with memory efficiency
    start_time = time.time()
    print("Processing generated data...")
    
    # Use memory-efficient processing
    adataGen_processed = process_data_for_gsea_memory_efficient(
        adataGen, 
        batch_size=1000,  # Smaller batch size
        use_temp_files=True,
        temp_dir=os.path.join(temp_dir, "gen")
    )
    
    # Force garbage collection
    gc.collect()
    
    print("Processing real data...")
    adataReal_processed = process_data_for_gsea_memory_efficient(
        adataReal,
        batch_size=1000,  # Smaller batch size
        use_temp_files=True,
        temp_dir=os.path.join(temp_dir, "real")
    )
    
    # Force garbage collection
    gc.collect()
    
    preprocess_time = time.time() - start_time
    print(f"Data preprocessing completed in {preprocess_time:.2f} seconds")
    
    # Print cell type stats
    gen_types = set(adataGen_processed.obs['cell_type'])
    real_types = set(adataReal_processed.obs['cell_type'])
    common_types = gen_types.intersection(real_types)
    
    print("\nCell type stats:")
    print(f"Generated cell types: {len(gen_types)}")
    print(f"Real data cell types: {len(real_types)}")
    print(f"Common cell types: {len(common_types)}")
    
    # Initialize memory-efficient GSEA analyzers
    print("\nInitializing memory-efficient GSEA analyzers...")
    start_time = time.time()
    
    gsea_analyzer_gen = MemoryEfficientCellTypeGSEA(
        adataGen_processed,
        cell_type_key='cell_type',
        gmt_file=gmt_file
    )
    
    gsea_analyzer_real = MemoryEfficientCellTypeGSEA(
        adataReal_processed,
        cell_type_key='cell_type',
        gmt_file=gmt_file
    )
    
    # Run GSEA sequentially to limit memory usage
    print("\nRunning memory-efficient GSEA sequentially with limited threads...")
    
    # Limit threads to control memory usage
    threads_to_use = max(1, min(threads, 2))  # Limit to 1 or 2 threads
    
    print("\nRunning GSEA on generated data...")
    resultsGen = gsea_analyzer_gen.run_gsea(
        min_size=min_size,
        max_size=max_size,
        permutations=permutations,
        threads=threads_to_use
    )
    
    # Force garbage collection
    gc.collect()
    
    print("\nRunning GSEA on real data...")
    resultsReal = gsea_analyzer_real.run_gsea(
        min_size=min_size,
        max_size=max_size,
        permutations=permutations,
        threads=threads_to_use
    )
    
    gsea_time = time.time() - start_time
    print(f"GSEA analysis completed in {gsea_time:.2f} seconds")
    
    # Only create plots if requested
    if save_toppathways:
        start_time = time.time()
        print("\nPlotting top pathways for generated data...")
        gG = gsea_analyzer_gen.plot_top_pathways()
        
        print("\nPlotting top pathways for real data...")
        gR = gsea_analyzer_real.plot_top_pathways()
        
        # Save plots if output directory is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            gG.figure.savefig(os.path.join(output_dir, 'gsea_gen_results.pdf'), bbox_inches='tight', dpi=300)
            gR.figure.savefig(os.path.join(output_dir, 'gsea_real_results.pdf'), bbox_inches='tight', dpi=300)
            
            # Save results to CSV
            for cell_type, res in resultsGen.items():
                res.to_csv(os.path.join(output_dir, f'gsea_gen_results_{cell_type}.csv'))
            for cell_type, res in resultsReal.items():
                res.to_csv(os.path.join(output_dir, f'gsea_real_results_{cell_type}.csv'))
        
        plotting_time = time.time() - start_time
        print(f"Pathway plotting completed in {plotting_time:.2f} seconds")
    
    # Prepare dataframes for comparison
    start_time = time.time()
    df_gen = prepare_results_df(resultsGen, '_gen')
    df_real = prepare_results_df(resultsReal, '_real')
    
    # Merge the dataframes on cell type and pathway term
    merged_df = pd.merge(
        df_gen[['Term', 'cell_type', 'NES']].rename(columns={'NES': 'NES_gen'}),
        df_real[['Term', 'cell_type', 'NES']].rename(columns={'NES': 'NES_real'}),
        on=['Term', 'cell_type'],
        how='inner'
    )
    
    print(f"\nMerged {len(merged_df)} pathways across {len(merged_df['cell_type'].unique())} cell types")
    
    # Save merged results if output directory is provided
    if output_dir is not None:
        merged_df.to_csv(os.path.join(output_dir, 'merged_gsea_results.csv'))
    
    # Calculate PR curves
    pr_results = calculate_pr_curves(merged_df, df_gen, df_real)
    
    # Plot PR curves
    if output_dir is not None:
        plot_pr_curves(pr_results, os.path.join(output_dir, 'precision_recall_curves.pdf'))
    else:
        plot_pr_curves(pr_results)
    
    analysis_time = time.time() - start_time
    print(f"Results analysis completed in {analysis_time:.2f} seconds")
    
    # Print total execution time
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"  - Preprocessing: {preprocess_time:.2f} seconds ({preprocess_time/total_time*100:.1f}%)")
    print(f"  - GSEA analysis: {gsea_time:.2f} seconds ({gsea_time/total_time*100:.1f}%)")
    
    if save_toppathways:
        print(f"  - Pathway plotting: {plotting_time:.2f} seconds ({plotting_time/total_time*100:.1f}%)")
    
    print(f"  - Results analysis: {analysis_time:.2f} seconds ({analysis_time/total_time*100:.1f}%)")
    
    # Clean up temporary directory
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Temporary directory cleaned up: {temp_dir}")
    except:
        print(f"Warning: Could not clean up temporary directory: {temp_dir}")
    
    # Return results
    return gsea_analyzer_gen, gsea_analyzer_real, df_gen, df_real, merged_df

