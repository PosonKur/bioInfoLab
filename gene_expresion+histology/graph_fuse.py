

# build graph and combine hist + gene expression similar to spaGCN s into a 50-dim latent 


import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

import matplotlib.pyplot as plt 



def parse_args():
    p = argparse.ArgumentParser(
        description="Fuse histology & gene PCs with a Graph Auto-Encoderand save a 50-dim embedding CSV"
    )
    p.add_argument("--input",  required=True, help="multimodal CSV file")
    p.add_argument("--output", required=True, help="CSV to write fused embeddings")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--k",      type=int, default=8, help="k-NN for graph")
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--hidden", type=int,   default=64, help="hidden dim")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# graph construction

def build_edge_index(coords3d: np.ndarray, k: int):
    """return symmetric edge_index (2×E) using k-NN in 3-D"""
    nbrs  = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(coords3d)
    idx   = nbrs.kneighbors(return_distance=False)  
    # shape (N,k+1) inc. self
    sources = np.repeat(np.arange(coords3d.shape[0]), k)
    # drop self-edge (first col)
    targets = idx[:, 1:].reshape(-1)   
    # make edges symmetric
    edge_index = np.vstack([np.hstack([sources, targets]),
                            np.hstack([targets, sources])])
    return torch.tensor(edge_index, dtype=torch.long)


# GNN autoencoder

class GraphAE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int):
        super().__init__()
        self.enc1 = SAGEConv(in_dim, hidden)
        self.enc2 = SAGEConv(hidden, latent)
        # feature reconstruction
        self.dec  = nn.Linear(latent, in_dim)        

    def forward(self, x, edge_index):
        h = self.enc1(x, edge_index).relu()
        z = self.enc2(h, edge_index)    
         # fused 50-dim code
        x_hat = self.dec(z)
        return z, x_hat




def main():
    args   = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    # 1) load csv
    df = pd.read_csv(args.input)
    meta_cols = ["Patch_X", "Patch_Y", "barcode", "label"]

    hist_cols = [f"PC{i}"      for i in range(1, 51)]
    gene_cols = [f"PC{i}_gene" for i in range(1, 51)]
    feat_cols = hist_cols + gene_cols

    # sanity check
    assert all(c in df.columns for c in feat_cols), "Missing PC columns!"

    # (N,100)
    X  = df[feat_cols].values.astype("float32")      

    # 2) Standard-scale node features
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    # 3) Build 3-D coordinates (x,y,z=PC1) 
    coords2d = df[["Patch_X", "Patch_Y"]].values
    z_hist   = df["PC1"].values.reshape(-1, 1)
    coords3d = np.hstack([coords2d, z_hist])         

    edge_index = build_edge_index(coords3d, k=args.k)

    # 4) create PyG data object
    data = Data(
        x           = torch.tensor(X_std, dtype=torch.float32),
        edge_index  = edge_index.to(device)
    ).to(device)

    # 5) model/loss/optimizer
    model = GraphAE(in_dim=X_std.shape[1], hidden=args.hidden, latent=50).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse   = nn.MSELoss()

    # 6) training loop with EARLY STOP
    losses = []
    best_val = float("inf")
    patience = 10
    wait     = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        optim.zero_grad()

        z, x_hat = model(data.x, data.edge_index)
        loss = mse(x_hat, data.x)
        loss.backward()
        optim.step()

        loss_val = loss.item()
        losses.append(loss_val)
        print(f"epoch {epoch:03d}  |  recon MSE {loss_val:.4f}")

        # early stopping on recon loss plateau
        if loss_val < best_val - 1e-4:
            best_val = loss_val
            wait = 0
            torch.save(model.state_dict(), "best_graphae.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # 7) plot and save the loss curve
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE")
    plt.title("Graph Auto-Encoder Training Loss")
    plt.tight_layout()
    plt.savefig("loss_curve_graph.png")
    print("saved loss curve to loss_curve_graph.png")
    plt.close()

    # 8) encode full graph 
    model.load_state_dict(torch.load("best_graphae.pt"))
    model.eval()
    with torch.no_grad():
        Z, _ = model(data.x, data.edge_index)
        Z = Z.cpu().numpy()

    # 9) write output csv
    z_cols = [f"Z{i+1}" for i in range(1, 51)]
    out_df = pd.concat(
        [df[meta_cols].reset_index(drop=True),
         pd.DataFrame(Z, columns=z_cols)],
        axis=1
    )
    out_df.to_csv(args.output, index=False)
    print(f"saved fused embeddings to {args.output} "
          f"({out_df.shape[0]} rows × {out_df.shape[1]} cols)")


if __name__ == "__main__":
    main()
