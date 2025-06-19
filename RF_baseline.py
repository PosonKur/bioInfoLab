#!/usr/bin/env python
"""
train_rf_all_patches.py
================================

Train a random-forest classifier on the patch-level Parquet table created by
`preprocess_prostate_visium.py`.

• Uses *all* labelled patches   – no `vote_frac` threshold.
• Keeps the vote fraction as a **sample-weight** so the forest can down-weight
  ambiguous patches without discarding them entirely.
• Drops any class with fewer than 2 patches (so stratified split can succeed).
• Saves: fitted model (`joblib`), confusion-matrix plot, and per-patch
  predictions on the held-out test set.

--------------------------------------------------------------------------------
EDIT THIS PATH ONLY
--------------------------------------------------------------------------------
"""
from pathlib import Path
PARQUET_PATH = Path("/Volumes/T7/Informatik/BioInfoLab/notebooks/scripts/patch_level.parquet")
# --------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib, json, datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

RANDOM_SEED = 42

# === 1. Load ------------------------------------------------------------------
print(f"Loading {PARQUET_PATH} …")
df = pd.read_parquet(PARQUET_PATH)

# Keep only non‐null labels
df = df[df["majority_pathology"].notna()].reset_index(drop=True)
print(f"Total labelled patches before filtering singletons: {len(df):,}")

# === 1b. Drop any class with fewer than 2 samples -----------------------------
label_counts = df["majority_pathology"].value_counts()
singletons   = label_counts[label_counts < 2].index.tolist()
if singletons:
    print(f"Dropping classes with <2 samples: {singletons}")
    df = df[~df["majority_pathology"].isin(singletons)].reset_index(drop=True)
    print(f"Total labelled patches after dropping singletons: {len(df):,}")

# === 2. Prepare feature matrix & labels --------------------------------------
feature_cols = [str(i) for i in range(1536)]
X = df[feature_cols].astype(np.float32).values

le = LabelEncoder()
y = le.fit_transform(df["majority_pathology"])

sample_w = df["vote_frac"].fillna(1.0).values

# === 3. Stratified split ------------------------------------------------------
X_tr, X_te, y_tr, y_te, w_tr, w_te, idx_tr, idx_te = train_test_split(
    X, y, sample_w, df.index,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_SEED
)

# === 4. Train Random-Forest ---------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    oob_score=True,
)
rf.fit(X_tr, y_tr, sample_weight=w_tr)

# === 5. Evaluation ------------------------------------------------------------
y_pred = rf.predict(X_te)

print(f"\nAccuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Macro-F1:  {f1_score(y_te, y_pred, average='macro'):.4f}\n")
print(classification_report(y_te, y_pred,
                            target_names=le.classes_, digits=3))

cm = confusion_matrix(y_te, y_pred, normalize="true")
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title("Random-Forest confusion matrix (normalized)")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=300)
plt.show()

# === 6. Persist artefacts -----------------------------------------------------
OUT_DIR = Path("rf_prostate_all_patches")
OUT_DIR.mkdir(exist_ok=True)

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = OUT_DIR / f"rf_model_{stamp}.joblib"
joblib.dump(rf, model_path)

pred_df = pd.DataFrame({
    "patch_id":  df.loc[idx_te, "patch_id"].values,
    "true_label": le.inverse_transform(y_te),
    "pred_label": le.inverse_transform(y_pred),
    "vote_frac": df.loc[idx_te, "vote_frac"].values,
})
pred_csv = OUT_DIR / f"test_set_predictions_{stamp}.csv"
pred_df.to_csv(pred_csv, index=False)

with open(OUT_DIR / f"label_encoder_{stamp}.json", "w") as fp:
    json.dump(dict(zip(le.classes_, le.transform(le.classes_))), fp)

print("\nSaved artefacts:")
print(f" • Model:       {model_path.resolve()}")
print(f" • Predictions: {pred_csv.resolve()}")
print(f" • Confusion-matrix plot: {Path('rf_confusion_matrix.png').resolve()}")
