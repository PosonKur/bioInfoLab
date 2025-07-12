import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---- Parameter anpassen ----
COUNT_FILE = "filtered_feature_bc_matrix_adenocarcinoma.h5"
#LABEL_CSV  = "/home/schever/Documents/BioInfo/cluster_plotting_tissue/annotation_prostate_cancer.csv"
FIG_DIR    = "figures_adenocarcinoma"

os.makedirs(FIG_DIR, exist_ok=True)

# ---- 1) Read the Visium dataset (mit Bilddaten!) ----
adata = sc.read_visium(
    path='./',
    count_file=COUNT_FILE,
    load_images=True
)
adata.var_names_make_unique()

# ---- 2) Library-ID bestimmen ----
lib = list(adata.uns["spatial"].keys())[0]
print("Using library_id =", lib)

# ---- 3) Preprocessing ----
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
print("Preprocessing done.")

# ---- 4) Resolutions ----
resolutions = np.round(np.arange(0.1, 1.01, 0.1), 1)

# ---- 5) Loop: Leiden clustering und Plot ----
for res in resolutions:
    key = f'leiden_{res:.1f}'
    print(f"Running leiden clustering for resolution={res:.1f}")
    sc.tl.leiden(adata, resolution=res, key_added=key)
    
    # Plot auf Histobild mit Clusterfarben
    sc.pl.spatial(
        adata,
        library_id=lib,
        img_key="hires",
        color=key,
        spot_size=200,
        show=False
    )
    # Speichern
    plot_fname = os.path.join(FIG_DIR, f"spatial_leiden_{res:.1f}.png")
    plt.savefig(plot_fname, dpi=300)
    plt.close()
    print(f"saved plot: {plot_fname}")

    # (Optional) CSV export
    df = pd.DataFrame({
        'barcode': adata.obs_names,
        'cluster': adata.obs[key].astype(str)
    })
    out_file = os.path.join(FIG_DIR, f'clusters_leiden_{res:.1f}.csv')
    df.to_csv(out_file, index=False)
    print(f"saved clustering for resolution={res:.1f} to {out_file}")

print("All plots and CSVs done!")

