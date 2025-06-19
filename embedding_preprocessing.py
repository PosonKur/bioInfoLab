#!/usr/bin/env python
"""
preprocess_prostate_visium.py

Create a patch-level table (Parquet) and an optional AnnData file that combine
UNI image embeddings with spatial transcriptomics labels.

Assumptions
-----------
* Only one Visium slide is present (change SLIDE_ID derivation if you have >1).
* `adata.obsm["spatial"]` stores spot **centre** coordinates in Visium "spot
  space" (same units used by Space Ranger's `tissue_positions_list.csv`).
* The embedding CSV has full-resolution pixel coordinates of the patch *top-left*
  corner in columns `X` and `Y`.
* Patches are laid out on a non-overlapping 224 × 224-px grid.

Outputs
-------
patch_level.parquet   : patch-level table (image-embeddings + labels)
patch_level.h5ad      : AnnData with gene expression aggregated per patch
"""

# --------------------------------------------------------------------------- #
# 🛠  SETTINGS –––– EDIT THESE THREE LINES ONLY
# --------------------------------------------------------------------------- #
H5AD_PATH        = "/Volumes/T7/Informatik/BioInfoLab/notebooks/annotated_prostate.h5ad"
SCALEFACTORS_JSON = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/spatial/scalefactors_json.json"
EMBEDDINGS_CSV    = "/Volumes/T7/Informatik/BioInfoLab/notebooks/embeddings/WSI_patch_embeddings_Prostate_Anicar_Cancer.csv"

PATCH_SIZE_PX     = 224              # pixel size of one patch/tile
PARQUET_OUT       = "patch_level.parquet"
ANN_OUT           = "patch_level.h5ad"
# --------------------------------------------------------------------------- #

import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------- #
# 1.  READ INPUTS
# ---------------------------------------------------------------------------- #
print("Loading files …")

adata = ad.read_h5ad(H5AD_PATH)
with open(SCALEFACTORS_JSON) as f:
    scalefactors = json.load(f)
df_embed = pd.read_csv(EMBEDDINGS_CSV)

# Make sure the embedding df has expected columns
expected_cols = {"X", "Y"}
missing       = expected_cols - set(df_embed.columns)
if missing:
    raise ValueError(f"Embedding CSV is missing columns: {missing}")

# Clean up embedding dataframe – unify the patch-id column name
patch_cols = [c for c in df_embed.columns if re.match(r"Patch_ID", c, re.I)]
if not patch_cols:
    raise ValueError("Could not infer a Patch_ID column in embeddings CSV.")
if len(patch_cols) > 1:
    # Keep the first, drop the duplicates
    keep = patch_cols[0]
    df_embed = df_embed.drop(columns=[c for c in patch_cols[1:]])
else:
    keep = patch_cols[0]
df_embed = df_embed.rename(columns={keep: "patch_id"})

# Embedding coordinate types
df_embed["X"] = df_embed["X"].astype(int)
df_embed["Y"] = df_embed["Y"].astype(int)

# Figure out slide / library identifier
if "Slide_ID" in df_embed.columns:
    slide_id = df_embed["Slide_ID"].iat[0]
else:  # single-slide fallback
    slide_id = next(iter(adata.uns["spatial"].keys()))
    df_embed["Slide_ID"] = slide_id
print(f"Using slide identifier: {slide_id}")

# ---------------------------------------------------------------------------- #
# 2.  ALIGN CO-ORDINATE SYSTEMS
# ---------------------------------------------------------------------------- #
print("Converting spot centres -> patch top-left …")
hires_scale = scalefactors["tissue_hires_scalef"]          # ≃0.07
spot_centres = adata.obsm["spatial"].copy()               # (n_spots, 2)

# Convert Visium spot units to pixel units of the *full-resolution* image
spot_px = spot_centres / hires_scale

# Get patch top-left by snapping to the 224-px grid
patch_tl = (np.floor_divide(spot_px, PATCH_SIZE_PX) * PATCH_SIZE_PX).astype(int)

adata.obs["X"] = patch_tl[:, 0]
adata.obs["Y"] = patch_tl[:, 1]
adata.obs["Slide_ID"] = slide_id    # add slide id for safe merge

# ---------------------------------------------------------------------------- #
# 3.  MERGE EMBEDDINGS ⟷ SPOTS
# ---------------------------------------------------------------------------- #
print("Merging embedding rows with spot meta …")
spot_df = (
    adata.obs.reset_index(drop=True)
         .loc[:, ["Slide_ID", "X", "Y", "Pathology"]]
)

merged = (df_embed.merge(
            spot_df,
            on=["Slide_ID", "X", "Y"],
            how="left",
            validate="one_to_many",        # 1 patch row  ⋖  many spots
         ))

# ---------------------------------------------------------------------------- #
# 4.  MAJORITY-VOTE LABEL PER PATCH
# ---------------------------------------------------------------------------- #
print("Computing majority vote …")

def majority_vote(series: pd.Series):
    """Return (mode, vote_fraction, n_non_nan)."""
    s = series.dropna()
    if s.empty:
        return pd.Series([np.nan, np.nan, 0], index=["label", "frac", "n"])
    vc = s.value_counts()
    return pd.Series([vc.idxmax(), vc.max() / len(s), len(s)],
                     index=["label", "frac", "n"])

label_df = (
    merged.groupby("patch_id", sort=False)["Pathology"]
          .apply(majority_vote)
          .reset_index()
          .rename(columns={
              "label": "majority_pathology",
              "frac":  "vote_frac",
              "n":     "n_spots"
          })
)

# ---------------------------------------------------------------------------- #
# 5.  FINAL PATCH-LEVEL TABLE
# ---------------------------------------------------------------------------- #
print("Building final patch-level table …")

# Embedding rows are already unique per patch -> simple left join
patch_df = (
    df_embed.merge(label_df, on="patch_id", how="left", validate="one_to_one")
)

# Write Parquet
print(f"Writing {PARQUET_OUT}")
patch_df.to_parquet(PARQUET_OUT, index=False)

# ---------------------------------------------------------------------------- #
# 6.  (OPTIONAL)  PATCH-LEVEL AnnData
# ---------------------------------------------------------------------------- #
print("Aggregating gene expression per patch (this may take a minute) …")

# map spot row index → patch_id
patch_idx = merged[["patch_id"]].copy()
patch_idx.index = merged.index  # align with adata.obs
adata.obs["patch_id"] = patch_idx["patch_id"]

# sparse or dense?
is_sparse = sparse.issparse(adata.X)

patch_order   = patch_df["patch_id"].tolist()
gene_names    = adata.var_names.copy()

agg_X = []
rows  = []

for pid in patch_order:
    spot_rows = np.where(adata.obs["patch_id"].values == pid)[0]
    if len(spot_rows) == 0:
        # No spots mapped (shouldn't happen) – fill zeros
        agg_X.append(sparse.csr_matrix((1, adata.n_vars)) if is_sparse
                     else np.zeros((1, adata.n_vars)))
        rows.append(pid)
        continue

    Xi = adata.X[spot_rows]
    Xi = Xi.mean(axis=0)            # mean across spots

    agg_X.append(Xi)
    rows.append(pid)

# Stack into matrix
X_patch = (sparse.vstack if is_sparse else np.vstack)(agg_X)

patch_obs = patch_df.set_index("patch_id")
patch_adata = ad.AnnData(
    X=X_patch,
    obs=patch_obs,
    var=adata.var.copy(),          # keep gene metadata
    obsm={"uni_embedding": patch_df.iloc[:, :1536].to_numpy(dtype="float32")}
)

print(f"Writing {ANN_OUT}")
patch_adata.write_h5ad(ANN_OUT, compression="gzip")

print("Done.  Outputs:")
print(f" • {Path(PARQUET_OUT).resolve()}")
print(f" • {Path(ANN_OUT).resolve()}")
