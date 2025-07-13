import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from scipy.spatial.distance import pdist
import igraph as ig
import os

# ==================== Pfade ====================
DATA_PATH = "/home/schever/Documents/BioInfo/Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5"
LABEL_PATH = "/home/schever/Documents/BioInfo/Pathology.csv"
TISSUE_POSITIONS_PATH = "/home/schever/Documents/BioInfo/spatial/tissue_positions_list.csv"

# =============== Daten laden ================
adata = sc.read_10x_h5(DATA_PATH)
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

# ========== Spatial-Koordinaten laden ==========
tissue_pos = pd.read_csv(TISSUE_POSITIONS_PATH, header=None)
tissue_pos.columns = [
    'barcode', 'in_tissue', 'array_row', 'array_col',
    'pxl_row_in_fullres', 'pxl_col_in_fullres'
]
tissue_pos = tissue_pos.set_index('barcode').loc[adata.obs_names]
adata.obsm['spatial'] = tissue_pos[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values

# ========== Ground truth laden ===========
labels_df = pd.read_csv(LABEL_PATH)
labels_df = labels_df.set_index('Barcode')
labels_df = labels_df.loc[adata.obs_names]
adata.obs['true_labels'] = labels_df['Pathology']

# ========== Eigene Funktion für Modularity ===========
def compute_modularity(adata, cluster_labels):
    G = adata.obsp['connectivities']
    sources, targets = G.nonzero()
    edges = list(zip(sources, targets))
    g = ig.Graph(edges=edges, directed=False)
    g.add_vertices(adata.n_obs - g.vcount())
    groups = pd.Categorical(cluster_labels).codes
    return g.modularity(groups)

# ================== Einstellungen ===================
resolutions = np.round(np.arange(0.1, 1.01, 0.1), 1)
ami_scores = []
ari_scores = []
n_clusters_per_resolution = []
modularity_scores = []
spatial_silhouette_scores = []
mean_intra_spatial_dist = []

if not os.path.exists('figures'):
    os.makedirs('figures')

for res in resolutions:
    key = f'leiden_{res:.1f}'
    sc.tl.leiden(adata, resolution=res, key_added=key)
    cluster_labels = adata.obs[key].astype(str)
    true_labels = adata.obs['true_labels'].astype(str)

    # Debug-Ausgabe (optional)
    print(f"{key}: ", adata.uns.get(key, {}).keys())

    ami = adjusted_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    ami_scores.append(ami)
    ari_scores.append(ari)

    n_clusters = cluster_labels.nunique()
    n_clusters_per_resolution.append(n_clusters)

    # Modularity Score
    try:
        modularity = compute_modularity(adata, cluster_labels)
    except Exception as e:
        print(f"Modularity konnte für {key} nicht berechnet werden: {e}")
        modularity = np.nan
    modularity_scores.append(modularity)

    # Spatial Silhouette Score
    if n_clusters > 1 and n_clusters < len(cluster_labels):
        sil_score = silhouette_score(adata.obsm['spatial'], cluster_labels)
    else:
        sil_score = np.nan
    spatial_silhouette_scores.append(sil_score)

    # Mean intra-cluster spatial distance
    intra_dists = []
    for cluster in cluster_labels.unique():
        coords = adata.obsm['spatial'][cluster_labels == cluster]
        if len(coords) > 1:
            dists = pdist(coords)
            intra_dists.append(np.mean(dists))
    mean_intra = np.mean(intra_dists) if intra_dists else np.nan
    mean_intra_spatial_dist.append(mean_intra)

    # CSV-Export wie gehabt
    df = pd.DataFrame({
        'barcode': adata.obs_names,
        'cluster': cluster_labels
    })
    out_file = f'clusters_{key}.csv'
    df.to_csv(out_file, index=False)
    print(f"saved clustering for resolution={res:.1f} to {out_file}")

# =================== Einzelplots ===================

def save_lineplot(x, y, ylabel, title, fname):
    plt.figure(figsize=(7,5))
    plt.plot(x, y, marker='o')
    plt.xlabel('Leiden Resolution', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

save_lineplot(resolutions, ami_scores, 'AMI', 'Adjusted Mutual Information', 'figures/AMI_vs_resolution.png')
save_lineplot(resolutions, ari_scores, 'ARI', 'Adjusted Rand Index', 'figures/ARI_vs_resolution.png')
save_lineplot(resolutions, modularity_scores, 'Modularity Q', 'Modularity Q (eigene Berechnung)', 'figures/Modularity_vs_resolution.png')
save_lineplot(resolutions, spatial_silhouette_scores, 'Spatial Silhouette', 'Spatial Silhouette Score (spatial)', 'figures/SpatialSilhouette_vs_resolution.png')
save_lineplot(resolutions, mean_intra_spatial_dist, 'Mean Intra-cluster Dist. (spatial)', 'Mean Intra-cluster Dist. (spatial)', 'figures/MeanIntraClusterSpatialDist_vs_resolution.png')

# Plot: Clusteranzahl vs. Resolution
save_lineplot(resolutions, n_clusters_per_resolution, 'Anzahl Cluster', 'Anzahl Cluster vs. Resolution', 'figures/Clusteranzahl_vs_resolution.png')

# ============= Gemeinsamer Plot: ARI und AMI =============
plt.figure(figsize=(8,6))
plt.plot(resolutions, ari_scores, marker='o', label='ARI')
plt.plot(resolutions, ami_scores, marker='o', label='AMI')
plt.xlabel('Leiden Resolution', fontsize=16)
plt.ylabel('Score', fontsize=16)
plt.title('ARI & AMI vs. Resolution', fontsize=18)
plt.legend(fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig('figures/ARI_AMI_vs_resolution.png', dpi=200)
plt.close()

# =========== Gemeinsamer Plot: Modularity, Silhouette, Intra-cluster ===========

# Min-max-Normierung für mean_intra_spatial_dist
intra = np.array(mean_intra_spatial_dist)
valid = ~np.isnan(intra)
if np.any(valid):
    intra_norm = (intra - np.nanmin(intra[valid])) / (np.nanmax(intra[valid]) - np.nanmin(intra[valid]))
else:
    intra_norm = intra

plt.figure(figsize=(8,6))
plt.plot(resolutions, modularity_scores, marker='o', label='Modularity Q')
plt.plot(resolutions, spatial_silhouette_scores, marker='o', label='Spatial Silhouette')
plt.plot(resolutions, intra_norm, marker='o', label='Mean Intra-cluster Dist. (spatial, norm.)')
plt.xlabel('Leiden Resolution', fontsize=16)
plt.ylabel('Score / (norm.)', fontsize=16)
plt.title('Unsupervised Scores vs. Resolution', fontsize=18)
plt.legend(fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
plt.savefig('figures/UnsupervisedScores_vs_resolution.png', dpi=200)
plt.close()

print("done")

