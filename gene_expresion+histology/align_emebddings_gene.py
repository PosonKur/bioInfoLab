
"""
build_multimodal_table.py
~~~~~~~~~~~~~~~~~~~~~~~~~

create multimodal training tables that align **histology patch embeddings**
with **spot-level gene expression** 

modes
---------
1.  spot    – one row per Visium spot / WSI patch

Optional PCA (separately for each modality) produces compact 50-dim blocks.

example usage
-------------
# 1) spot-level, full dims
python build_multimodal_table.py \\
       --mode spot \\
       --wsi-csv WSI_patch_embeddings_centered-224_adenocarcinoma.csv \\
       --tissue-csv spatial/tissue_positions_list.csv \\
       --clusters-csv clusters_leiden_0.3.csv \\
       --visium-dir /path/to/VisiumRun  \\
       --count-file Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5

"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA



def read_wsi_embeddings(path: str) -> pd.DataFrame:
    """
    Load the WSI patch embeddings.

    Expected columns
    ----------------
    '0'…'1535'     embedding dims (strings)
    'X', 'Y'       tile centre coordinates
    plus any extra metadata we ignore downstream.
    """
    df = pd.read_csv(path)
    if not {"X", "Y"}.issubset(df.columns):
        raise ValueError(f"WSI file '{path}' must contain columns 'X' and 'Y'.")
    return df


def read_tissue_positions(path: str) -> pd.DataFrame:
    """
    10x `tissue_positions_list.csv`  (no header).

    Per 10x spec:
      col 0 = barcode
      col 1,2,3 = in_tissue, array_row, array_col
      col 4 = Y, col 5 = X

    We read all columns then drop the unused ones.
    """
    col_names = ["barcode","in_tissue","array_row","array_col","Y","X"]
    df = pd.read_csv(path, header=None, names=col_names, dtype={"barcode": str})
    return df[["barcode","X","Y"]]


def read_clusters(path: str) -> pd.DataFrame:
    """
    CSV with at least columns [barcode, cluster].
    """
    df = pd.read_csv(path, dtype={"barcode": str})
    if not {"barcode","cluster"}.issubset(df.columns):
        raise ValueError(f"Cluster file '{path}' must contain 'barcode' and 'cluster'.")
    return df[["barcode","cluster"]]


def make_patch_map(
    tissue_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    wsi_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge barcode→cluster→(X,Y) with WSI embeddings by spatial join on X,Y.

    Returns one row per patch:
      [barcode, cluster, Patch_X, Patch_Y, '0'…'1535', …]
    """
    # 1) barcode → (cluster, X, Y)
    spot_map = clusters_df.merge(tissue_df, on="barcode", how="inner")
    spot_map.rename(columns={"X":"Patch_X","Y":"Patch_Y"}, inplace=True)

    # 2) join embeddings on those coords
    patch_df = spot_map.merge(
        wsi_df,
        left_on=["Patch_X","Patch_Y"],
        right_on=["X","Y"],
        how="inner",
        suffixes=("","_wsi")
    )

    # drop the duplicate X,Y brought in from wsi_df
    return patch_df.drop(columns=["X","Y"])


def extract_embedding_matrix(patch_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Pull out the block of columns named '0'..'1535' as a Float32 DataFrame.
    Returns also a list of the other cols.
    """
    embedding_cols = [c for c in patch_df.columns if c.isdigit()]
    other_cols     = [c for c in patch_df.columns if c not in embedding_cols]
    embed_df       = patch_df[embedding_cols].astype("float32")
    return embed_df, other_cols


def fit_transform_pca(mat: pd.DataFrame, n_pc: int = 50, seed: int = 0) -> pd.DataFrame:
    """
    Apply sklearn.PCA to mat.values, return PC1…PCn DataFrame.
    """
    pca = PCA(n_components=n_pc, random_state=seed, svd_solver="auto")
    pcs = pca.fit_transform(mat.values)
    cols = [f"PC{i+1}" for i in range(n_pc)]
    return pd.DataFrame(pcs, index=mat.index, columns=cols)


def gene_matrix_from_adata(
    adata: sc.AnnData,
    barcodes: pd.Index
) -> pd.DataFrame:
    """
    Extract a dense (spots × genes) matrix *after* adata has already been
    normalized & log1p-transformed globally.

    Returns
    -------
    DataFrame indexed by barcode, columns = adata.var_names
    """
    sub = adata[barcodes, :].copy()
    X = sub.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return pd.DataFrame(X, index=barcodes, columns=adata.var_names).astype("float32")


def make_pseudobulk(expr_df: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """
    Compute mean expression per cluster.
    """
    return expr_df.join(clusters).groupby("cluster").mean()


def save_table(df: pd.DataFrame, path: str):
    """
    Ensure parent directory exists (if any) and write CSV.
    """
    parent = os.path.dirname(path)
    if parent:  # only try to make a directory if there's a non-empty parent path
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✔ Saved: {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")


# --------------------------------------------------------------------- #
# --------------------------- Main pipeline --------------------------- #
# --------------------------------------------------------------------- #

def build_table(args):
    # 1) load data
    print("reading Visium AnnData …")
    adata = sc.read_visium(
        args.visium_dir,
        count_file=args.count_file,
        load_images=False
    )

    # global gene-count normalization & log1p 
    print("• Normalizing & log1p-transformation of gene counts …")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 2) Load your three CSVs
    print("• Reading CSVs …")
    wsi_df      = read_wsi_embeddings(args.wsi_csv)
    tissue_df   = read_tissue_positions(args.tissue_csv)
    clusters_df = read_clusters(args.clusters_csv)

    # 3) Spatially match patches and spots
    patch_df = make_patch_map(tissue_df, clusters_df, wsi_df)
    print(f"• Patches matched to spots: {len(patch_df):,}")

    # 4) Split off the histology embedding block
    embed_df, other_cols = extract_embedding_matrix(patch_df)

    # 5) Extract the (already-normalized) gene matrix
    expr_df = gene_matrix_from_adata(adata, patch_df["barcode"])
    print(f"• Gene matrix shape           : {expr_df.shape}")

    # 6) If cluster mode, build pseudobulk and remap to patches
    if args.mode == "cluster":
        pseudo = make_pseudobulk(expr_df, patch_df["cluster"])
        expr_block = patch_df["cluster"].map(pseudo.to_dict("index"))
        expr_block = pd.DataFrame(expr_block.tolist(), index=patch_df.index)
        print(f"• Using pseudobulk expression : {expr_block.shape}")
    else:
        expr_block = expr_df
        print(f"• Using spot-level expression : {expr_block.shape}")

    # 7) Optional PCA on either modality
    if args.pca_embeddings:
        print("pCA → embeddings …")
        embed_df = fit_transform_pca(embed_df, n_pc=args.n_components)
    if args.pca_genes:
        print("pCA → expression …")
        expr_block = fit_transform_pca(expr_block, n_pc=args.n_components)

    # 8) Assemble final table
    final_df = (
    patch_df[["Patch_X","Patch_Y","barcode","cluster"]]
      .rename(columns={"cluster":"label"})
      .join(embed_df)    
      .join(expr_block, on="barcode", rsuffix="_gene")
        )

    # 9) Save with informative filename
    suffix = f"{args.mode}" \
           + ( "_pcaEmb"  if args.pca_embeddings else "" ) \
           + ( "_pcaGene" if args.pca_genes else "" )
    out_name = f"{args.prefix}_{suffix}.csv"
    save_table(final_df.reset_index(drop=True), out_name)




def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="combine WSI patch embeddings with gene expression into a multimodal training table"
    )

    # default paths
    default_wsi = "/Volumes/T7/Informatik/BioInfoLab/notebooks/embeddings/WSI_patch_embeddings/WSI_patch_embeddings_centered-224_adenocarcinoma.csv"
    default_tissue = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/new_data/spatial/tissue_positions_list.csv"
    default_clusters = "/Volumes/T7/Informatik/BioInfoLab/notebooks/scripts/preprocessing_clustering/clusters_leiden_0.3.csv"
    default_visium = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/new_data/spatial/"
    default_count = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/new_data/Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5"

    p.add_argument(
        "--wsi-csv", default=default_wsi,
        help=f"CSV of patch embeddings incl. X,Y cols (default: {default_wsi})"
    )
    p.add_argument(
        "--tissue-csv", default=default_tissue,
        help=f"spatial/tissue_positions_list.csv (default: {default_tissue})"
    )
    p.add_argument(
        "--clusters-csv", default=default_clusters,
        help=f"clusters_leiden_*.csv with barcode,cluster (default: {default_clusters})"
    )
    p.add_argument(
        "--visium-dir", default=default_visium,
        help=f"root dir of Visium run (contains /spatial, /analysis, …) (default: {default_visium})"
    )
    p.add_argument(
        "--count-file", default=default_count,
        help=f"name of 10x HDF5, e.g. *_filtered_feature_bc_matrix.h5 (default: {default_count})"
    )

   # mode & options
    p.add_argument("--mode", choices=["spot", "cluster"], default="spot",
                   help="Merge gene data at spot level or use cluster pseudobulk.")
    # PCA options default to True; use --no-pca-embeddings / --no-pca-genes to disable
    p.add_argument("--no-pca-embeddings", dest="pca_embeddings", action="store_false", default=True,
                   help="Disable reduction of embeddings block to 50 PCs.")
    p.add_argument("--no-pca-genes", dest="pca_genes", action="store_false", default=True,
                   help="Disable reduction of gene-expression block to 50 PCs.")
    p.add_argument("--n-components", type=int, default=50,
                   help="Number of PCs (default 50).")


    p.add_argument("--prefix", default="multimodal",
                   help="Prefix for output filename.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    build_table(args)
