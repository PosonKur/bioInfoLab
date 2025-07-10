
import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# paths
DATA_DIR         = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/new_data"
COUNT_FILE       = "Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5"
LABEL_CSV        = os.path.join(DATA_DIR, "annotation_prostate_cancer.csv")
MARKER_CSV       = "/Volumes/T7/Informatik/BioInfoLab/notebooks/scripts/clustering/data/marker_genes_prostate.csv"
OUTPUT_DIR       = "."
FIG_DIR          = "figures"

# fraction for bootstrap subsampling
SUBSAMPLE_FRAC   = 0.5     
# number of bootstrap iterations
N_BOOTSTRAPS     = 20      
PVALUE_THRESHOLD = 0.05
LOGFC_THRESHOLD  = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 1. load data
adata = sc.read_visium(
    path=DATA_DIR,
    count_file=COUNT_FILE,
    load_images=True
)
adata.var_names_make_unique()

# align pathology labels
lib = list(adata.uns["spatial"].keys())[0]
print("Using library_id =", lib)
labels = (
    pd.read_csv(LABEL_CSV)
      .set_index("Barcode")
      .reindex(adata.obs_names)
)
labels["Pathology"] = labels["Pathology"].fillna("Unknown")
adata.obs["Pathology"] = labels["Pathology"].astype(str)

# normalization, log, PCA
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")

# 1. load and filter marker gens
# these genes are from http://bio-bigdata.hrbmu.edu.cn/CellMarker/CellMarkerSearch.jsp?quickSearchInfo=prostate&index_key=2#framekuang 
# under prostate 
CELLNAME_TO_CATEGORY = {
    "Cancer stem cell":   "Invasive carcinoma",
    "Cancer cell":        "Invasive carcinoma",
    "LNCaP cell":         "Invasive carcinoma",
    "Epithelial cell":    "Normal gland",
    "Luminal cell":       "Normal gland",
    "Basal cell":         "Normal gland",
    "Luminal epithelial cell": "Normal gland",
    "Stem cell":          "Normal gland",
    "Progenitor cell":    "Normal gland",
    "Prostate stem cell": "Normal gland",
    "Late transit-amplifying prostate epithelial cell":  "Normal gland",
    "Early transit-amplifying prostate epithelial cell": "Normal gland",
    "Fibroblast":         "Fibrous tissue",
    "Cancer-associated fibroblast": "Fibrous tissue",
    "Mesenchymal cell":   "Fibrous tissue",
    "Smooth muscle cell": "Fibro-muscular tissue",
    "Endothelial cell":   "Blood vessel",
    "Neuron":             "Nerve",
    "Dentate granule cell":"Nerve",
    "T cell":             "Immune Cells",
    "B cell":             "Immune Cells",
    "Natural killer cell":"Immune Cells",
    "Macrophage":         "Immune Cells",
    "Myeloid cell":       "Immune Cells",
    "Plasma cell":        "Immune Cells",
    "Neutrophil":         "Immune Cells",
}

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
marker_dict = {
    cat: list(subdf["Cell marker"])
    for cat, subdf in top5.groupby("Category")
}

filtered = {}
for cat, genes in marker_dict.items():
    present = [g for g in genes if g in adata.raw.var_names]
    missing = set(genes) - set(present)
    if missing:
        print(f"Warning: {cat} missing markers dropped: {missing}")
    if present:
        filtered[cat] = present
    else:
        print(f"Warning: {cat} has no valid markers; skipping.")
marker_dict = filtered

# 3. parameter grids 
methods = {
    "leiden":       np.round(np.arange(0.1, 1.01, 0.1), 2),
    "louvain":      np.round(np.arange(0.1, 1.01, 0.1), 2),
    "kmeans":       list(range(2, 12)),  # 2–11
    "hierarchical": list(range(2, 12))
}

results = []

for method, params in methods.items():
    for p in params:
        key = f"{method}_{p}"
        print(f"\n→ {method.upper()} clustering, param = {p}")

        # 3.1 assign labels + compute x-axis value
        if method == "leiden":
            sc.tl.leiden(adata, resolution=p, key_added=key)
            labels = adata.obs[key].values
            xval   = p
        elif method == "louvain":
            sc.tl.louvain(adata, resolution=p, key_added=key)
            labels = adata.obs[key].values
            xval   = p
        else:
            X = adata.obsm["X_pca"]
            if method == "kmeans":
                labels = KMeans(n_clusters=int(p), random_state=0).fit_predict(X)
            else:
                labels = AgglomerativeClustering(n_clusters=int(p)).fit_predict(X)
            adata.obs[key] = pd.Categorical([str(int(l)) for l in labels])
            # map k=2→0.1, k=3→0.2, …, k=11→1.0
            xval = (p - 1) / 10.0

        # 3.2 internal validation
        sil = silhouette_score(adata.obsm["X_pca"], labels)
        ch  = calinski_harabasz_score(adata.obsm["X_pca"], labels)
        db  = davies_bouldin_score(adata.obsm["X_pca"], labels)

        # 3.3 bootstrap stability (jaccard)
        n_obs = adata.n_obs
        sub_n = int(n_obs * SUBSAMPLE_FRAC)
        jaccs = []
        for i in range(N_BOOTSTRAPS):
            idx = np.random.choice(n_obs, sub_n, replace=False)
            sub = adata[idx].copy()
            if method in ("leiden", "louvain"):
                sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca")
                if method == "leiden":
                    sc.tl.leiden(sub, resolution=p, key_added="sub")
                else:
                    sc.tl.louvain(sub, resolution=p, key_added="sub")
                newl = sub.obs["sub"].values
            else:
                Xsub = sub.obsm["X_pca"]
                if method == "kmeans":
                    newl = KMeans(n_clusters=int(p), random_state=0).fit_predict(Xsub)
                else:
                    newl = AgglomerativeClustering(n_clusters=int(p)).fit_predict(Xsub)
            orig = labels[idx]
            scores = []
            for clu in np.unique(orig):
                orig_idx = set(np.where(orig == clu)[0])
                best = 0
                for nc in np.unique(newl):
                    new_idx = set(np.where(newl == nc)[0])
                    inter = len(orig_idx & new_idx)
                    union = len(orig_idx | new_idx)
                    if union:
                        best = max(best, inter/union)
                scores.append(best)
            jaccs.append(np.mean(scores))
            del sub; gc.collect()
        stability = float(np.mean(jaccs))

        # 3.4 de‐gene counts
        sc.tl.rank_genes_groups(
            adata,
            groupby=key,
            method="wilcoxon",
            use_raw=True,
            n_genes=adata.raw.n_vars
        )
        de = adata.uns["rank_genes_groups"]
        de_counts = []
        for clu in de["names"].dtype.names:
            p_adj = np.array(de["pvals_adj"][clu])
            lfcs  = np.array(de["logfoldchanges"][clu])
            mask  = (p_adj < PVALUE_THRESHOLD) & (lfcs >= LOGFC_THRESHOLD)
            de_counts.append(int(mask.sum()))
        avg_de = float(np.mean(de_counts))
        del adata.uns["rank_genes_groups"]; gc.collect()

        # 3.5 marker‐enrichment
        res_results = []
        cats = adata.obs[key].cat.categories if key in adata.obs else np.unique(labels.astype(str))
        for clust in cats:
            mask = (adata.obs[key] == clust).values
            for category, genes in marker_dict.items():
                pvals, lfc = [], []
                for g in genes:
                    expr = adata.raw[:, g].X
                    vec  = expr.toarray().ravel() if hasattr(expr, "toarray") else expr.ravel()
                    grp1, grp2 = vec[mask], vec[~mask]
                    _, pval   = mannwhitneyu(grp1, grp2, alternative="greater")
                    pvals.append(pval)
                    lfc.append(np.mean(grp1) - np.mean(grp2))
                _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
                n_sig    = int((p_adj < 0.05).sum())
                mean_lfc = float(np.mean(lfc))
                comp     = float(np.mean(-np.log10(p_adj + 1e-300)))
                res_results.append({
                    "method":        method,
                    "param":         p,
                    "cluster":       clust,
                    "category":      category,
                    "n_significant": n_sig,
                    "mean_lfc":      mean_lfc,
                    "comp_score":    comp
                })
        df_res = pd.DataFrame(res_results)
        tag    = str(p).replace('.', '_')
        fn     = os.path.join(OUTPUT_DIR, f"{method}_marker_enrichment_{tag}.csv")
        df_res.to_csv(fn, index=False)
        print(f"  → Saved marker‐enrichment to {fn}")

        # 3.6 record aggregated metrics
        results.append({
            "method":            method,
            "param":             p,
            "xval":              xval,
            "silhouette":        sil,
            "calinski_harabasz": ch,
            "davies_bouldin":    db,
            "stability":         stability,
            "de_counts":         avg_de,
            "avg_n_significant": df_res["n_significant"].mean(),
            "avg_mean_lfc":      df_res["mean_lfc"].mean(),
            "avg_comp_score":    df_res["comp_score"].mean()
        })

        # memory cleanup so this bs actually runs
        if key in adata.obs:
            adata.obs.drop(columns=[key], inplace=True)
        gc.collect()

# 4. save combined metrics 
df_all = pd.DataFrame(results)
df_all.to_csv(os.path.join(OUTPUT_DIR, "all_methods_metrics.csv"), index=False)
print("saved combined metrics to all_methods_metrics.csv")

# prepare ticks: 0.1–1.0 mapped to “res/k” pairs
resolutions = np.round(np.arange(0.1, 1.01, 0.1), 2)
ks          = list(range(2, 12))
tick_labels = [f"{r:.1f}/{k}" for r, k in zip(resolutions, ks)]

metrics = [
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "stability",
    "de_counts",
    "avg_n_significant",
    "avg_mean_lfc",
    "avg_comp_score"
]
for metric in metrics:
    plt.figure()
    for method in methods:
        sub = df_all[df_all.method == method]
        plt.plot(sub["xval"], sub[metric], marker='o', label=method)
    plt.xticks(resolutions, tick_labels, rotation=45)
    plt.xlabel("Leiden/Louvain resolution  /  K-Means-Hierarchical clusters")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"{metric.replace('_', ' ').title()} by method")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, f"{metric}_by_method.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"saved plot: {fname}")

print("All done.")


