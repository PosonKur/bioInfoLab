

# auto-encoder that fuses histology PCs and gene-expression PCs into a 50-dim latent representation



import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt 



def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="train a multimodal autoencoder on PCA-reduced histology + gene data and output fused 50-dim embeddings."
    )
    p.add_argument("--input",  required=True, help="Input CSV with PCA columns")
    p.add_argument("--output", required=True, help="Output CSV with Z1…Z50")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# autoencoder

class AE(nn.Module):
    """A tiny symmetric auto-encoder: 100 → 128 → 64 → 50 → 64 → 128 → 100."""
    def __init__(self, in_dim: int = 100, bottleneck: int = 50):
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
         # reconstruction, latent
        return out, z 



def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    return columns that represent PCA features:
      - histology PCs  :  PC\d+            (e.g. 'PC1')
      - gene PCs       :  PC\d+_gene       (e.g. 'PC1_gene')
    """
    return [c for c in df.columns if c.startswith("PC")]

def train_epoch(model, loader, loss_fn, optim, device):
    model.train()
    running = 0.0
    for xb, in loader:
        xb = xb.to(device)
        recon, _ = model(xb)
        loss = loss_fn(recon, xb)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            running += loss.item() * xb.size(0)
    return running / len(loader.dataset)



def main():
    args = get_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    # 1) load CSV 
    df = pd.read_csv(args.input)
    meta_cols = ["Patch_X", "Patch_Y", "barcode", "label"]
    feature_cols = pick_feature_columns(df)

    X = df[feature_cols].values.astype("float32")
    # keep identifiers / label
    meta = df[meta_cols].copy()    

    # 2) standard-scale each feature 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) train/val split 
    X_train, X_val = train_test_split(
        X_scaled, test_size=0.2, random_state=0, shuffle=True
    )
    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    # 4) model / loss / optim 
    # should be 100 (50+50)
    in_dim = X.shape[1]            
    model = AE(in_dim=in_dim, bottleneck=50).to(device)
    loss_fn = nn.MSELoss()
    optim   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 5) training loop with loss recording 
    train_losses = []
    val_losses   = []
    best_val = float("inf")
    patience = 5
    wait     = 0

    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, loss_fn, optim, device)
        vl_loss = eval_epoch(model, val_loader, loss_fn, device)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"epoch {epoch:03d} | train {tr_loss:.4f} | val {vl_loss:.4f}")

        if vl_loss < best_val - 1e-4:
            best_val = vl_loss
            wait = 0
            torch.save(model.state_dict(), "best_ae.pt")
        else:
            wait += 1
            if wait >= patience:
                print("early stopping.")
                break

    # 6) plot and save loss curves 
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training & Validation Loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi = 400)
    print("saved loss curve to loss_curve.png")
    plt.close()

    # 7) load best model and encode ALL rows
    model.load_state_dict(torch.load("best_ae.pt"))
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(device)
        _, Z = model(X_tensor)
        Z = Z.cpu().numpy()

    # 8) output CSV 
    # Z1…Z50
    z_cols = [f"Z{i+1}" for i in range(Z.shape[1])]  
    out_df = pd.concat([meta.reset_index(drop=True),
                        pd.DataFrame(Z, columns=z_cols)],
                       axis=1)

    out_df.to_csv(args.output, index=False)
    print(f"saved fused embeddings to {args.output}  "
          f"({out_df.shape[0]} rows × {out_df.shape[1]} cols)")

if __name__ == "__main__":
    main()








