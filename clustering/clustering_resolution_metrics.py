


import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# paths
DATA_DIR    = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/new_data"
COUNT_FILE  = "Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5"
LABEL_CSV   = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/new_data/annotation_prostate_cancer.csv"
OUTPUT_DIR  = "."

# bootstrap params
# fraction of spots for subsampling
SUBSAMPLE_FRAC = 0.5 
# number of bootstrap iterations
N_BOOTSTRAPS   = 20    

# DE test thresholds
# adjusted p-value cutoff
PVALUE_THRESHOLD = 0.05
# log2 fold-change cutoff
LOGFC_THRESHOLD  = 1.0   


os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) load and preprocess data
adata = sc.read_visium(
    path=DATA_DIR,
    count_file=COUNT_FILE,
    load_images=True
)
adata.var_names_make_unique()

# align pathology labels
lib = list(adata.uns["spatial"].keys())[0]
print("using library_id =", lib)
labels = (
    pd.read_csv(LABEL_CSV)
      .set_index("Barcode")
      .reindex(adata.obs_names)
)
labels["Pathology"] = labels["Pathology"].fillna("Unknown")
adata.obs["Pathology"] = labels["Pathology"].astype(str)

# preprocessing for clustering and DE
sc.pp.normalize_total(adata, target_sum=1e4)  
sc.pp.log1p(adata)                            
adata.raw = adata                              
sc.pp.pca(adata, n_comps=50)                  
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")  

# 2) Loop over Leiden resolutions
results = []
resolutions = np.round(np.arange(0.1, 1.01, 0.1), 2)
for res in resolutions:
    print(f"\nProcessing Leiden resolution = {res}")
    key = f"leiden_{res}"

    # 2.1 Compute clustering labels
    sc.tl.leiden(adata, resolution=res, key_added=key)
    labels_arr = adata.obs[key].values

    # 2.2 Internal validation on PCA space
    sil = silhouette_score(adata.obsm["X_pca"], labels_arr)
    ch  = calinski_harabasz_score(adata.obsm["X_pca"], labels_arr)
    db  = davies_bouldin_score(adata.obsm["X_pca"], labels_arr)

    # 2.3 Bootstrap subsampling + Jaccard stability
    n_spots = adata.n_obs
    subsz   = int(n_spots * SUBSAMPLE_FRAC)
    jaccs   = []
    for i in range(N_BOOTSTRAPS):
        # Sample a random subset of spots
        idx = np.random.choice(n_spots, subsz, replace=False)
        sub = adata[idx].copy()  # copy only subsample

        # Recompute neighbor graph & clustering on subsample
        sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca")
        sc.tl.leiden(sub, resolution=res, key_added="leiden_sub")

        # Compare original labels (restricted to subset) vs new
        orig_sub = adata.obs[key].values[idx]
        new_lbls = sub.obs["leiden_sub"].values
        uniq = np.unique(orig_sub)
        js = []
        for clu in uniq:
            orig_idx = set(np.where(orig_sub == clu)[0])
            # find best matching new cluster Jaccard
            best = 0
            for nc in np.unique(new_lbls):
                new_idx = set(np.where(new_lbls == nc)[0])
                inter = len(orig_idx & new_idx)
                union = len(orig_idx | new_idx)
                if union:
                    j = inter/union
                    if j > best:
                        best = j
            js.append(best)
        jaccs.append(np.mean(js))

        # Free memory from subsample
        del sub
        gc.collect()

    stability = float(np.mean(jaccs))

    # 2.4 Differential expression counts using Scanpy
    sc.tl.rank_genes_groups(
        adata,
        groupby=key,
        method="wilcoxon",
        use_raw=True,
        n_genes=adata.raw.n_vars  # test all genes
    )
    de = adata.uns["rank_genes_groups"]
    clusters = de["names"].dtype.names
    de_counts = []
    for clu in clusters:
        p_adj = np.array(de["pvals_adj"][clu])
        lfcs  = np.array(de["logfoldchanges"][clu])
        # Count genes passing thresholds
        mask = (p_adj < PVALUE_THRESHOLD) & (lfcs >= LOGFC_THRESHOLD)
        de_counts.append(int(np.sum(mask)))
    avg_de = float(np.mean(de_counts))

    # Clean up DE results to free memory
    del adata.uns["rank_genes_groups"]
    gc.collect()

    # Remove clustering labels from adata.obs
    adata.obs.drop(columns=[key], inplace=True)
    gc.collect()

    # Store metrics
    results.append({
        "resolution": res,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "stability": stability,
        "de_counts": avg_de
    })

# 3) Save metrics as CSV
df = pd.DataFrame(results)
out_csv = os.path.join(OUTPUT_DIR, "clustering_quality_metrics.csv")
df.to_csv(out_csv, index=False)
print("Saved metrics to", out_csv)

# 4) Plot resolution vs each metric
for metric in ["silhouette", "calinski_harabasz", "davies_bouldin", "stability", "de_counts"]:
    plt.figure()
    plt.plot(df["resolution"], df[metric], marker='o')
    plt.xlabel("Leiden resolution")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"Resolution vs {metric.replace('_', ' ').title()}")
    out_fig = os.path.join(OUTPUT_DIR, f"{metric}_vs_resolution.png")
    plt.savefig(out_fig, dpi=150)
    print("Saved plot:", out_fig)
    plt.close()

print("All done.")



