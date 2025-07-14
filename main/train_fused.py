

"""
train and evaluate DNN, CNN, SVM on fused_patch_embeddings.csv (these contain gene expression and histology data)
usage:
    python train_fused.py --model dnn
    python train_fused.py --model cnn
    python train_fused.py --model svm
    python train_fused.py --model all
"""
import os
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

# For SVM
from sklearn.svm import SVC
import joblib

import matplotlib.pyplot as plt

from plotter import visualize_tissue_image_with_samples_color_labels

# reproducibility
torch.manual_seed(42)
np.random.seed(42)


# CSV_PATH   = "/Volumes/T7/Informatik/BioInfoLab/notebooks/scripts/gene_expresion+histology/fused_patch_embeddings.csv"

# CSV_PATH   = "/Volumes/T7/Informatik/BioInfoLab/notebooks/scripts/gene_expresion+histology/split_fused_patch_embeddings_split.csv"

CSV_PATH   = "/Volumes/T7/Informatik/BioInfoLab/notebooks/scripts/gene_expresion+histology/fused_graph_embeddings.csv"



IMAGE_PATH = "/Volumes/T7/Informatik/BioInfoLab/notebooks/data/new_data/spatial/tissue_hires_image.png"
RESULT_DIR = "results/fused_emb_models"
EPOCHS     = 100
BATCH_SIZE = 64



parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", "-m",
    choices=["dnn", "cnn", "svm", "all"],
    default="all",
    help="Which model(s) to train & evaluate"
)
args = parser.parse_args()
run_dnn = args.model in ("dnn", "all")
run_cnn = args.model in ("cnn", "all")
run_svm = args.model in ("svm", "all")

# 2) load data
df = pd.read_csv(CSV_PATH)
feat_cols = [c for c in df.columns if c.startswith("Z")]
X = df[feat_cols].astype(np.float32).values
y_le = LabelEncoder()
y = y_le.fit_transform(df["label"].values).astype(np.int64)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tr_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
    batch_size=BATCH_SIZE, shuffle=True
)
te_loader = DataLoader(
    TensorDataset(torch.tensor(X_te), torch.tensor(y_te)),
    batch_size=BATCH_SIZE
)

INPUT_DIM = X.shape[1]
N_CLASSES = len(y_le.classes_)

# 3) define models
class SimpleDNN(nn.Module):
    def __init__(self, in_dim, n_cls, p=0.5):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, n_cls)
        )
    def forward(self, x): return self.seq(x)

class SimpleCNN(nn.Module):
    def __init__(self, n_cls, in_dim, p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(p),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(p)
        )
        conv_out = 32 * (in_dim // 4)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, n_cls)
        )
    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

# 4) train and  helper 
def train_and_eval(model, train_ld, test_ld, is_cnn=False, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    loss_history = []

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for xb, yb in train_ld:
            if is_cnn and xb.ndim == 2:
                xb = xb.unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        loss_history.append(np.mean(epoch_losses))

    model.eval()
    outs, labs = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            if is_cnn and xb.ndim == 2:
                xb = xb.unsqueeze(1)
            outs.append(model(xb)); labs.append(yb)
    logits = torch.cat(outs, 0)
    true   = torch.cat(labs, 0).cpu().numpy()
    pred   = logits.argmax(1).cpu().numpy()

    acc = accuracy_score(true, pred)
    ari = adjusted_rand_score(true, pred)
    f1  = f1_score(true, pred, average="weighted")
    return acc, ari, f1, model, loss_history

# 5) svm train
def train_svm(X_tr, y_tr, X_te, y_te, out_dir):
    print("Training SVM on full embeddings...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale',
              random_state=42, probability=True)
    svm.fit(X_tr, y_tr)
    path = os.path.join(out_dir, "svm_full.pkl")
    joblib.dump(svm, path)
    print(f"saved SVM model to: {path}")

    pred = svm.predict(X_te)
    return "SVM", accuracy_score(y_te, pred), adjusted_rand_score(y_te, pred), f1_score(y_te, pred, average="weighted")

# 6) run everything 
os.makedirs(RESULT_DIR, exist_ok=True)
metrics = []
loss_histories = {}

# — DNN —
if run_dnn:
    print(">>> Training DNN...")
    acc, ari, f1, dnn, losses = train_and_eval(
        SimpleDNN(INPUT_DIM, N_CLASSES),
        tr_loader, te_loader, is_cnn=False
    )
    print(f"DNN → Acc={acc:.3f}, ARI={ari:.3f}, F1={f1:.3f}")
    metrics.append(("DNN", acc, ari, f1))
    loss_histories["DNN"] = losses

# — CNN —
if run_cnn:
    print(">>> Training CNN...")
    tr_ld_c = DataLoader(
        TensorDataset(torch.tensor(X_tr).unsqueeze(1), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    te_ld_c = DataLoader(
        TensorDataset(torch.tensor(X_te).unsqueeze(1), torch.tensor(y_te)),
        batch_size=BATCH_SIZE
    )
    acc, ari, f1, cnn, losses = train_and_eval(
        SimpleCNN(N_CLASSES, INPUT_DIM),
        tr_ld_c, te_ld_c, is_cnn=True
    )
    print(f"CNN → Acc={acc:.3f}, ARI={ari:.3f}, F1={f1:.3f}")
    metrics.append(("CNN", acc, ari, f1))
    loss_histories["CNN"] = losses

# — SVM —
if run_svm:
    svm_res = train_svm(X_tr, y_tr, X_te, y_te, RESULT_DIR)
    metrics.append(svm_res)

# plot loss curves for DNN/CNN only
if loss_histories:
    plt.figure()
    for name, losses in loss_histories.items():
        plt.plot(range(1, len(losses)+1), losses, label=name)
    plt.xlabel("Epoch"); plt.ylabel("Training Loss")
    plt.title("Loss Curves")
    plt.legend(); plt.grid(True)
    p = os.path.join(RESULT_DIR, "loss_curves.png")
    plt.savefig(p, dpi=400); plt.close()
    print(f"Saved loss curves to: {p}")

# 7) overlays for DNN/CNN/SVM 
print("Overlaying predictions on tissue image…")
all_X  = torch.tensor(X).float()
all_Xc = all_X.unsqueeze(1)

if run_dnn:
    with torch.no_grad():
        preds = dnn(all_X).argmax(1).cpu().numpy()
    labels = y_le.inverse_transform(preds)
    df2 = df[["Patch_X","Patch_Y"]].copy(); df2["label"] = labels
    visualize_tissue_image_with_samples_color_labels(
        IMAGE_PATH, df2, 27482, 25219,
        output_path=os.path.join(RESULT_DIR, "dnn_pred.png")
    )

if run_cnn:
    with torch.no_grad():
        preds = cnn(all_Xc).argmax(1).cpu().numpy()
    labels = y_le.inverse_transform(preds)
    df2 = df[["Patch_X","Patch_Y"]].copy(); df2["label"] = labels
    visualize_tissue_image_with_samples_color_labels(
        IMAGE_PATH, df2, 27482, 25219,
        output_path=os.path.join(RESULT_DIR, "cnn_pred.png")
    )

if run_svm:
    svm = joblib.load(os.path.join(RESULT_DIR, "svm_full.pkl"))
    preds = svm.predict(X)
    labels = y_le.inverse_transform(preds)
    df2 = df[["Patch_X","Patch_Y"]].copy(); df2["label"] = labels
    visualize_tissue_image_with_samples_color_labels(
        IMAGE_PATH, df2, 27482, 25219,
        output_path=os.path.join(RESULT_DIR, "svm_pred.png")
    )

# 8) save metrics
pd.DataFrame(
    [{"model": m, "accuracy": a, "ARI": r, "F1_weighted": f}
     for m, a, r, f in metrics]
).to_csv(os.path.join(RESULT_DIR, "scores.csv"), index=False)

print("results in:", RESULT_DIR)




