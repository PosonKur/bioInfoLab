"""
install scanpy, anndata

run:
conda install -c conda-forge pyarrow fastparquet

"""





#!/usr/bin/env python
"""
preprocess_prostate_visium.py  —  AUTO-OFFSET VERSION (fixed sparse format)
"""

# --------------------------------------------------------------------------- #
#  EDIT ONLY THESE THREE LINES
# --------------------------------------------------------------------------- #
H5AD_PATH         = "/Volumes/T7/Informatik/BioInfoLab/notebooks/annotated_prostate.h5ad"
SCALEFACTORS_JSON = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/spatial/scalefactors_json.json"
EMBEDDINGS_CSV    = "/Volumes/T7/Informatik/BioInfoLab/notebooks/embeddings/WSI_patch_embeddings_Prostate_Anicar_Cancer.csv"
# --------------------------------------------------------------------------- #

PATCH_SIZE_PX = 224
PARQUET_OUT   = "patch_level.parquet"
ANN_OUT       = "patch_level.h5ad"

import json, re, numpy as np, pandas as pd, anndata as ad
from collections import Counter
from pathlib import Path
from scipy import sparse

print("Loading inputs …")
adata = ad.read_h5ad(H5AD_PATH)
with open(SCALEFACTORS_JSON) as f:
    hires_scale = json.load(f)["tissue_hires_scalef"]
dfE = pd.read_csv(EMBEDDINGS_CSV)

# ---- tidy embedding table -------------------------------------------------- #
patch_cols = [c for c in dfE.columns if re.match(r"Patch_ID", c, re.I)]
if not patch_cols:
    raise ValueError("No Patch_ID column found in embeddings CSV.")
dfE = dfE.rename(columns={patch_cols[0]: "patch_id"})
dfE["X"] = dfE["X"].astype(int)
dfE["Y"] = dfE["Y"].astype(int)

if "Slide_ID" in dfE.columns:
    slide_id = dfE["Slide_ID"].iat[0]
else:
    slide_id = next(iter(adata.uns["spatial"].keys()))
    dfE["Slide_ID"] = slide_id
print(f"Using slide identifier: {slide_id}")

# ---- 1.  spot → pixel ------------------------------------------------------ #
spot_centres = adata.obsm["spatial"].copy()
spot_px = spot_centres * hires_scale           # ← multiply, *not* divide
print(f"Applied hires_scale × {hires_scale:.5f} to spot coordinates")

# ---- 2.  derive grid indices for embeddings -------------------------------- #
dfE["patch_ix"] = dfE["X"] // PATCH_SIZE_PX
dfE["patch_iy"] = dfE["Y"] // PATCH_SIZE_PX

# ---- 3.  initial grid indices for spots  ----------------------------------- #
ix_s = (spot_px[:, 0] // PATCH_SIZE_PX).astype(int)
iy_s = (spot_px[:, 1] // PATCH_SIZE_PX).astype(int)

# ---- 4.  automatic global offset (mode-to-mode) ---------------------------- #
ix_e_mode = Counter(dfE["patch_ix"]).most_common(1)[0][0]
iy_e_mode = Counter(dfE["patch_iy"]).most_common(1)[0][0]
ix_s_mode = Counter(ix_s).most_common(1)[0][0]
iy_s_mode = Counter(iy_s).most_common(1)[0][0]

dx = ix_e_mode - ix_s_mode       # embedding grid – spot grid
dy = iy_e_mode - iy_s_mode

print(f"Global patch-grid offset:  Δx = {dx}  Δy = {dy}  (patch units)")
print(f"                         ≈ {dx*PATCH_SIZE_PX} px, {dy*PATCH_SIZE_PX} px")

ix_s += dx
iy_s += dy

adata.obs["patch_ix"] = ix_s
adata.obs["patch_iy"] = iy_s
adata.obs["Slide_ID"] = slide_id

# ---- 5.  merge embeddings ↔ spots ------------------------------------------ #
spot_df = (
    adata.obs.reset_index(drop=True)
         .loc[:, ["Slide_ID", "patch_ix", "patch_iy", "Pathology"]]
)

merged = dfE.merge(
    spot_df,
    on=["Slide_ID", "patch_ix", "patch_iy"],
    how="left",
    validate="one_to_many",
)
matched = merged["Pathology"].notna().sum()
print(f"Matched spots with labels: {matched:,}")

# ---- 6.  majority vote per patch ------------------------------------------ #
mode  = merged.groupby("patch_id")["Pathology"].agg(
    lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else np.nan
)
frac  = (
    merged.groupby("patch_id")["Pathology"]
          .value_counts(normalize=True)
          .groupby(level=0)
          .max()
)
count = merged.groupby("patch_id")["Pathology"].apply(
    lambda s: s.dropna().shape[0]
)

label_df = pd.DataFrame({
    "patch_id":           mode.index,
    "majority_pathology": mode.values,
    "vote_frac":          frac.values,
    "n_spots":            count.values,
})

# ---- 7.  final patch table ------------------------------------------------- #
patch_df = dfE.merge(label_df, on="patch_id", how="left", validate="one_to_one")
print("Non-null labels in patch_df:",
      patch_df["majority_pathology"].notna().sum())

print(f"Writing {PARQUET_OUT}")
patch_df.to_parquet(
    PARQUET_OUT,
    engine="pyarrow",       # or "fastparquet"
    compression="snappy",   # universally supported
    index=False
)

# ---- 8.  optional patch-level AnnData (fixed spot→patch_id mapping) ------- #
print("Aggregating gene expression per patch …")

# Build a quick lookup of (patch_ix, patch_iy) → patch_id
mapping = (
    dfE[["patch_id","patch_ix","patch_iy"]]
      .drop_duplicates(["patch_ix","patch_iy"])
      .set_index(["patch_ix","patch_iy"])["patch_id"]
      .to_dict()
)

# Map each spot in adata.obs to its patch_id
adata.obs["patch_id"] = [
    mapping.get((ix,iy), np.nan)
    for ix, iy in zip(adata.obs["patch_ix"], adata.obs["patch_iy"])
]

# Now aggregate spot-level expression into patch-level
is_sparse = sparse.issparse(adata.X)
patch_order = patch_df["patch_id"].tolist()

agg_X = []
for pid in patch_order:
    spot_rows = np.where(adata.obs["patch_id"].values == pid)[0]
    if len(spot_rows) == 0:
        agg_X.append(
            sparse.csr_matrix((1, adata.n_vars)) if is_sparse
            else np.zeros((1, adata.n_vars))
        )
    else:
        Xi = adata.X[spot_rows].mean(axis=0)
        agg_X.append(Xi)

# Stack; ensure CSR format before writing
X_patch = (sparse.vstack if is_sparse else np.vstack)(agg_X)
if is_sparse:
    X_patch = X_patch.tocsr()

patch_adata = ad.AnnData(
    X=X_patch,
    obs=patch_df.set_index("patch_id"),
    var=adata.var.copy(),
    obsm={"uni_embedding": patch_df.iloc[:, :1536].to_numpy(dtype="float32")},
)

print(f"Writing {ANN_OUT}")
patch_adata.write_h5ad(ANN_OUT, compression="gzip")
print("Done ✓")

