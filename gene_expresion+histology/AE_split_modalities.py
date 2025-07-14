

# ---------------------------------------------------------------
# autoencoder that fuses histology PCs and gene expression PCs into a 50-dim latent representation, with separate histology vs gene loss
# ---------------------------------------------------------------

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------- CLI ----------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="train a multimodal AE on histology+gene PCs and output 50-dim fused embeddings"
    )
    p.add_argument("--input",  required=True, help="CSV with PC1…PC50 and PC1_gene…PC50_gene")
    p.add_argument("--output", required=True, help="CSV to write fused embeddings (Z1…Z50)")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# autoencoder

class AE(nn.Module):
    """symmetric AE: input → 128 → 64 → 50 → 64 → 128 → input"""
    def __init__(self, in_dim, bottleneck=50):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z




def main():
    args = get_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    # 1) load csv
    df = pd.read_csv(args.input)
    # keep identifiers
    meta = df[["Patch_X", "Patch_Y", "barcode", "label"]].copy()

    # explicitly pick histology vs gene PCA columns
    hist_cols = [f"PC{i}"      for i in range(1,51) if f"PC{i}"      in df.columns]
    gene_cols = [f"PC{i}_gene" for i in range(1,51) if f"PC{i}_gene" in df.columns]
    feature_cols = hist_cols + gene_cols
    hist_dim = len(hist_cols)

    X = df[feature_cols].values.astype("float32")

    # 2) scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) train val split
    X_train, X_val = train_test_split(
        X_scaled, test_size=0.2, random_state=0, shuffle=True
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=args.batch, shuffle=True
    )
    val_loader   = DataLoader(
        TensorDataset(torch.from_numpy(X_val)),
        batch_size=args.batch, shuffle=False
    )

    # 4) build model
    # in_dim should be hist_dim + gene_dim = 100
    in_dim = X.shape[1]  
    model = AE(in_dim=in_dim, bottleneck=50).to(device)
    mse_loss = nn.MSELoss()
    optim    = torch.optim.Adam(model.parameters(),
                                lr=args.lr, weight_decay=1e-4)

    # 5) train loop with sepearte losses
    train_tot, train_hist, train_gene = [], [], []
    val_tot,   val_hist,   val_gene   = [], [], []
    best_val = float("inf")
    patience = 5
    wait     = 0

    for epoch in range(1, args.epochs+1):
        # — train —
        model.train()
        running_tot = running_hist = running_gene = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            recon, _ = model(batch_x)

            # split recon & true into hist vs gene
            recon_h = recon[:, :hist_dim]
            recon_g = recon[:, hist_dim:]
            true_h  = batch_x[:, :hist_dim]
            true_g  = batch_x[:, hist_dim:]

            loss_h = mse_loss(recon_h, true_h)
            loss_g = mse_loss(recon_g, true_g)
            loss   = loss_h + loss_g

            optim.zero_grad()
            loss.backward()
            optim.step()

            n = batch_x.size(0)
            running_tot  += loss.item() * n
            running_hist += loss_h.item() * n
            running_gene += loss_g.item() * n

        n_train = len(train_loader.dataset)
        tr_tot = running_tot / n_train
        tr_h   = running_hist / n_train
        tr_g   = running_gene / n_train
        train_tot.append(tr_tot)
        train_hist.append(tr_h)
        train_gene.append(tr_g)

        # — validate —
        model.eval()
        running_tot = running_hist = running_gene = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon, _ = model(batch_x)

                recon_h = recon[:, :hist_dim]
                recon_g = recon[:, hist_dim:]
                true_h  = batch_x[:, :hist_dim]
                true_g  = batch_x[:, hist_dim:]

                loss_h = mse_loss(recon_h, true_h)
                loss_g = mse_loss(recon_g, true_g)
                loss   = loss_h + loss_g

                n = batch_x.size(0)
                running_tot  += loss.item() * n
                running_hist += loss_h.item() * n
                running_gene += loss_g.item() * n

        n_val = len(val_loader.dataset)
        vl_tot = running_tot / n_val
        vl_h   = running_hist / n_val
        vl_g   = running_gene / n_val
        val_tot.append(vl_tot)
        val_hist.append(vl_h)
        val_gene.append(vl_g)

        print(f"Epoch {epoch:03d} | "
              f"train_total {tr_tot:.4f}  hist {tr_h:.4f}  gene {tr_g:.4f}  | "
              f"val_total   {vl_tot:.4f}  hist {vl_h:.4f}  gene {vl_g:.4f}")

        # ERALY STOPPING IMP
        if vl_tot < best_val - 1e-4:
            best_val = vl_tot
            wait = 0
            torch.save(model.state_dict(), "best_ae.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # 6) plot loss
    plt.figure()
    plt.plot(train_tot, label="train_total")
    plt.plot(val_tot,   label="val_total")
    plt.plot(train_hist, '--', label="train_hist")
    plt.plot(val_hist,   '--', label="val_hist")
    plt.plot(train_gene, ':', label="train_gene")
    plt.plot(val_gene,   ':', label="val_gene")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("AE Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi = 400)
    print("saved loss curve as loss_curve.png")
    plt.close()

    # 7) encode all patches
    model.load_state_dict(torch.load("best_ae.pt"))
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(device)
        _, Z = model(X_tensor)
        Z = Z.cpu().numpy()

    # 8) save combined embeddings
    z_cols = [f"Z{i+1}" for i in range(Z.shape[1])]
    out_df = pd.concat([meta.reset_index(drop=True),
                        pd.DataFrame(Z, columns=z_cols)],
                       axis=1)
    out_df.to_csv(args.output, index=False)
    print(f"saved combined embeddings to {args.output}")

if __name__ == "__main__":
    main()
