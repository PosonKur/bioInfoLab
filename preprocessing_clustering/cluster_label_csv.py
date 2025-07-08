import scanpy as sc
import numpy as np
import pandas as pd


DATA_PATH = "Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix_adenocarcinoma.h5"


adata = sc.read_10x_h5(DATA_PATH)
adata.var_names_make_unique()


sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=10)


sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# pca and neighborhood graph
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

# resolutions
resolutions = np.round(np.arange(0.1, 1.01, 0.1), 1)

# run leiden for each resolution and create csv file
for res in resolutions:
    key = f'leiden_{res:.1f}'
    sc.tl.leiden(adata, resolution=res, key_added=key)

   
    df = pd.DataFrame({
        'barcode': adata.obs_names,
        'cluster': adata.obs[key].astype(str)
    })

    # csv out
    out_file = f'clusters_{key}.csv'
    df.to_csv(out_file, index=False)
    print(f"saved clustering for resolution={res:.1f} to {out_file}")

print("done")
