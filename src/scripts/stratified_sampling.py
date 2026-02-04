"""
Stratified sampling for h5ad files.

Samples a specified percentage of cells from each cell type (stratified sampling)
to maintain the original distribution of cell types.
"""

import argparse
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path


def stratified_sample(adata, groupby, fraction=0.3, random_seed=None):
    """
    Perform stratified sampling on an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    groupby : str
        The column in adata.obs to stratify by.
    fraction : float
        Fraction of cells to sample from each group (default: 0.3).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        The sampled AnnData object.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sampled_indices = []

    for group_val in adata.obs[groupby].unique():
        group_mask = adata.obs[groupby] == group_val
        group_indices = np.where(group_mask)[0]

        # Calculate sample size (at least 1 if group has cells)
        n_sample = max(1, int(len(group_indices) * fraction))

        # Randomly sample without replacement
        sampled = np.random.choice(group_indices, min(n_sample, len(group_indices)), replace=False)
        sampled_indices.extend(sampled)

    return adata[sampled_indices].copy()


def main():
    parser = argparse.ArgumentParser(description='Perform stratified sampling on h5ad files.')
    parser.add_argument('input', type=str, help='Input h5ad file path')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output h5ad file path')
    parser.add_argument('--fraction', '-f', type=float, default=0.3, help='Fraction to sample (default: 0.3)')
    parser.add_argument('--groupby', '-g', type=str, default='celltype_l3', help='Column to stratify by (default: celltype_l3)')
    parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--verify', action='store_true', help='Print distribution comparison before and after sampling')

    args = parser.parse_args()

    # Read data
    print(f"Reading data from: {args.input}")
    adata = sc.read_h5ad(args.input)
    print(f"Original shape: {adata.shape}")
    print(f"Number of cell types: {adata.obs[args.groupby].nunique()}")

    # Print original distribution if verifying
    if args.verify:
        print("\n=== Original Distribution ===")
        original_dist = adata.obs[args.groupby].value_counts(normalize=True)
        print(original_dist)

    # Perform stratified sampling
    print(f"\nPerforming stratified sampling ({args.fraction * 100:.0f}% per cell type)...")
    adata_sampled = stratified_sample(adata, args.groupby, args.fraction, args.seed)

    print(f"Sampled shape: {adata_sampled.shape}")
    print(f"Total cells sampled: {adata_sampled.n_obs}")

    # Print sampled distribution if verifying
    if args.verify:
        print("\n=== Sampled Distribution ===")
        sampled_dist = adata_sampled.obs[args.groupby].value_counts(normalize=True)
        print(sampled_dist)

        print("\n=== Distribution Comparison ===")
        comparison = pd.DataFrame({
            'Original': original_dist,
            'Sampled': sampled_dist,
            'Difference': sampled_dist - original_dist
        })
        print(comparison)

    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        stem = input_path.stem
        args.output = str(input_path.parent / f"{stem}_sampled_{int(args.fraction * 100)}pct.h5ad")

    # Save sampled data
    print(f"\nSaving sampled data to: {args.output}")
    adata_sampled.write(args.output)
    print("Done!")


if __name__ == '__main__':
    main()
