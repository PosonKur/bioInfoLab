#!/usr/bin/env python
"""
marker_enrichment_evaluation_with_plots.py

Extends the original script to also plot Leiden resolution vs.
marker-gene enrichment quality scores (only the three enrichment metrics),
and save per-resolution scores to individual CSVs.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os

# —— User parameters ——
DATA_DIR    = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/new_data"
COUNT_FILE  = "Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5"
LABEL_CSV   = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/new_data/annotation_prostate_cancer.csv"
MARKER_CSV  = "/Volumes/T7/Informatik/BioInfoLab/notebooks/scripts/clustering/data/marker_genes_prostate.csv"
OUTPUT_CSV  = "marker_enrichment_scores.csv"
FIG_DIR     = "figures"

# make sure output directories exist
os.makedirs(FIG_DIR, exist_ok=True)

# Mapping from CellMarker "Cell name" → your 7 pathology categories
CELLNAME_TO_CATEGORY = {
    # Invasive carcinoma
    "Cancer stem cell":   "Invasive carcinoma",
    "Cancer cell":        "Invasive carcinoma",
    "LNCaP cell":         "Invasive carcinoma",
    # Normal gland
    "Epithelial cell":                       "Normal gland",
    "Luminal cell":                          "Normal gland",
    "Basal cell":                            "Normal gland",
    "Luminal epithelial cell":               "Normal gland",
    "Stem cell":                             "Normal gland",
    "Progenitor cell":                       "Normal gland",
    "Prostate stem cell":                    "Normal gland",
    "Late transit-amplifying prostate epithelial cell":  "Normal gland",
    "Early transit-amplifying prostate epithelial cell": "Normal gland",
    # Fibrous tissue
    "Fibroblast":              "Fibrous tissue",
    "Cancer-associated fibroblast": "Fibrous tissue",
    "Mesenchymal cell":        "Fibrous tissue",
    # Fibro-muscular tissue
    "Smooth muscle cell":      "Fibro-muscular tissue",
    # Blood vessel
    "Endothelial cell":        "Blood vessel",
    # Nerve
    "Neuron":                  "Nerve",
    "Dentate granule cell":    "Nerve",
    # Immune Cells
    "T cell":                  "Immune Cells",
    "B cell":                  "Immune Cells",
    "Natural killer cell":     "Immune Cells",
    "Macrophage":              "Immune Cells",
    "Myeloid cell":            "Immune Cells",
    "Plasma cell":             "Immune Cells",
    "Neutrophil":              "Immune Cells",
}

# 1) Load Visium data
adata = sc.read_visium(
    path=DATA_DIR,
    count_file=COUNT_FILE,
    load_images=True
)
adata.var_names_make_unique()

# 2) Align pathology labels
lib = list(adata.uns["spatial"].keys())[0]
print("Using library_id =", lib)
labels = (
    pd.read_csv(LABEL_CSV)
      .set_index("Barcode")
      .reindex(adata.obs_names)
)
labels["Pathology"] = labels["Pathology"].fillna("Unknown")
adata.obs["Pathology"] = labels["Pathology"].astype(str)

# 3) Preprocess for clustering & DE
sc.pp.normalize_total(adata, target_sum=1e4)  # count normalization
sc.pp.log1p(adata)                            # log-transform
adata.raw = adata                             
sc.pp.pca(adata, n_comps=50)                   # PCA to 50 PCs
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")

# 4) Load & aggregate markers
df_markers = pd.read_csv(MARKER_CSV)
df_markers = df_markers[df_markers["Cell name"].isin(CELLNAME_TO_CATEGORY)]
df_markers["Category"] = df_markers["Cell name"].map(CELLNAME_TO_CATEGORY)
agg = (
    df_markers
      .groupby(["Category", "Cell marker"], as_index=False)["Supports"]
      .sum()
)
top5 = (
    agg.sort_values(["Category", "Supports"], ascending=[True, False])
       .groupby("Category")
       .head(5)
)
marker_dict = { cat: list(subdf["Cell marker"]) for cat, subdf in top5.groupby("Category") }

# 5) Filter missing genes
filtered = {}
for cat, genes in marker_dict.items():
    present = [g for g in genes if g in adata.raw.var_names]
    missing = set(genes) - set(present)
    if missing:
        print(f"Warning: {cat} → missing markers (dropped): {missing}")
    if present:
        filtered[cat] = present
    else:
        print(f"Warning: {cat} has no valid markers and will be skipped.")
marker_dict = filtered

# 6) Run clustering, enrichment & per‐resolution CSV export
results = []
for res in np.arange(0.1, 1.01, 0.1):
    res = round(float(res), 2)
    key = f"leiden_{res}"
    print(f"\n→ Clustering at resolution {res}")

    sc.tl.leiden(adata, resolution=res, key_added=key)

    # collect this resolution’s results
    res_results = []
    for clust in adata.obs[key].cat.categories:
        mask = adata.obs[key] == clust
        for category, genes in marker_dict.items():
            pvals, lfc = [], []
            for gene in genes:
                expr = adata.raw[:, gene].X
                expr = expr.toarray().flatten() if hasattr(expr, "toarray") else expr.flatten()
                grp1, grp2 = expr[mask.values], expr[~mask.values]
                _, p = mannwhitneyu(grp1, grp2, alternative="greater")
                pvals.append(p)
                lfc.append(np.mean(grp1) - np.mean(grp2))

            _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
            n_sig      = int(np.sum(p_adj < 0.05))
            mean_lfc   = float(np.mean(lfc))
            comp_score = float(np.mean(-np.log10(p_adj + 1e-300)))

            record = {
                "resolution":    res,
                "cluster":       clust,
                "category":      category,
                "n_significant": n_sig,
                "mean_lfc":      mean_lfc,
                "comp_score":    comp_score,
            }
            results.append(record)
            res_results.append(record)

    # save this resolution’s slice
    df_res_res = pd.DataFrame(res_results)
    res_tag   = int(res * 100)
    fn        = f"marker_enrichment_scores_resolution_{res_tag}.csv"
    df_res_res.to_csv(fn, index=False)
    print(f"  → Saved per-resolution scores to {fn}")

# 7) Save aggregated results
df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_CSV, index=False)
print("\nResults written to", OUTPUT_CSV)

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_CSV, index=False)
print("Columns in df_res:", df_res.columns.tolist())
print(df_res.head())

# If 'resolution' is missing, this will raise an assert and stop you early
assert "resolution" in df_res.columns, (
    "ERROR: df_res has no 'resolution' column. "
    "Check your loop where you populate `results`!"
)

# Now it’s safe to group
summary = (
    df_res
      .groupby("resolution")
      .agg({
          "n_significant": "mean",
          "mean_lfc":      "mean",
          "comp_score":    "mean",
      })
      .reset_index()
)

# Plot avg # significant markers vs resolution
plt.figure()
plt.plot(summary["resolution"], summary["n_significant"], marker='o')
plt.xlabel('Leiden resolution')
plt.ylabel('Avg. # significant markers')
plt.title('Resolution vs. Avg. # Significant Markers')
plt.savefig(os.path.join(FIG_DIR, "resolution_vs_n_significant_markers.png"))

# Plot avg log2 fold-change vs resolution
plt.figure()
plt.plot(summary["resolution"], summary["mean_lfc"], marker='o')
plt.xlabel('Leiden resolution')
plt.ylabel('Avg. log2 fold-change')
plt.title('Resolution vs. Avg. Log2 Fold-Change')
plt.savefig(os.path.join(FIG_DIR, "resolution_vs_mean_lfc.png"))

# Plot avg composite score vs resolution
plt.figure()
plt.plot(summary["resolution"], summary["comp_score"], marker='o')
plt.xlabel('Leiden resolution')
plt.ylabel('Avg. composite -log10(adj p)')
plt.title('Resolution vs. Avg. Composite Score')
plt.savefig(os.path.join(FIG_DIR, "resolution_vs_comp_score.png"))

plt.show()
