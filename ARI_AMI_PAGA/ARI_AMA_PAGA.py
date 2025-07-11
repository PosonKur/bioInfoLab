import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

DATA_PATH = "/home/schever/Documents/BioInfo/ARI_AMI_PAGA/Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5"
LABEL_PATH = "/home/schever/Documents/BioInfo/ARI_AMI_PAGA/Pathology.csv"

# === Daten laden und vorverarbeiten ===
adata = sc.read_10x_h5(DATA_PATH)
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

# === Ground truth einlesen ===
labels_df = pd.read_csv(LABEL_PATH)
# Passe ggf. die Spaltennamen hier an:
# Angenommen: 'barcode' und 'label'
labels_df = labels_df.set_index('Barcode')
# Filter auf Barcodes, die auch in adata vorkommen:
labels_df = labels_df.loc[adata.obs_names]
adata.obs['true_labels'] = labels_df['Pathology']

resolutions = np.round(np.arange(0.1, 1.01, 0.1), 1)
ami_scores = []
ari_scores = []
paga_connectivities = []

for res in resolutions:
    key = f'leiden_{res:.1f}'
    sc.tl.leiden(adata, resolution=res, key_added=key)
    
    cluster_labels = adata.obs[key].astype(str)
    true_labels = adata.obs['true_labels'].astype(str)

    ami = adjusted_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    ami_scores.append(ami)
    ari_scores.append(ari)

    # Sicherstellen: Mindestens 2 nicht-leere Cluster
    cluster_counts = cluster_labels.value_counts()
    n_nonempty_clusters = (cluster_counts > 0).sum()
    if n_nonempty_clusters < 2:
        print(f"Resolution {res:.2f}: Weniger als 2 nicht-leere Cluster, PAGA übersprungen.")
        paga_connectivities.append(np.nan)
    else:
        try:
            sc.tl.paga(adata, groups=key)
            paga_score = adata.uns['paga']['connectivities'].mean()
            paga_connectivities.append(paga_score)
        except Exception as e:
            print(f"PAGA Fehler bei Resolution {res:.2f}: {e}")
            paga_connectivities.append(np.nan)
    
    # CSV-Export wie gehabt
    df = pd.DataFrame({
        'barcode': adata.obs_names,
        'cluster': cluster_labels
    })
    out_file = f'clusters_{key}.csv'
    df.to_csv(out_file, index=False)
    print(f"saved clustering for resolution={res:.1f} to {out_file}")

# === Plots ===
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(resolutions, ami_scores, marker='o')
plt.xlabel('Leiden Resolution')
plt.ylabel('AMI')
plt.title('AMI vs. Resolution')

plt.subplot(1,3,2)
plt.plot(resolutions, ari_scores, marker='o')
plt.xlabel('Leiden Resolution')
plt.ylabel('ARI')
plt.title('ARI vs. Resolution')

plt.subplot(1,3,3)
plt.plot(resolutions, paga_connectivities, marker='o')
plt.xlabel('Leiden Resolution')
plt.ylabel('PAGA Mean Connectivity')
plt.title('PAGA Connectivity vs. Resolution')

plt.tight_layout()
plt.savefig('figures/ARI_AMI_PAGA_vs_resolution.png', dpi=200)
plt.show()

print("done")

