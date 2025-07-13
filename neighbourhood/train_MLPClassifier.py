import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ----- Konfiguration -----
single_train_csv = "single_patch_train.csv"
single_test_csv = "single_patch_test.csv"
concat_train_csv = "concat_patch_train.csv"
concat_test_csv = "concat_patch_test.csv"

# ---- Daten einlesen ----
print("Lese Trainings- und Testdaten für Single Patch ein ...")
df_single_train = pd.read_csv(single_train_csv)
df_single_test = pd.read_csv(single_test_csv)
print(f"  Single Patch: train={len(df_single_train)}, test={len(df_single_test)}")

print("Lese Trainings- und Testdaten für 9er-Gruppe ein ...")
df_concat_train = pd.read_csv(concat_train_csv)
df_concat_test = pd.read_csv(concat_test_csv)
print(f"  9er-Gruppe: train={len(df_concat_train)}, test={len(df_concat_test)}")

# ----- Features und Ziel extrahieren -----
# Single Patch
X_single_train = df_single_train.drop(columns=['Patch_X', 'Patch_Y', 'label'])
y_single_train = df_single_train['label']
X_single_test = df_single_test.drop(columns=['Patch_X', 'Patch_Y', 'label'])
y_single_test = df_single_test['label']

# 9er-Gruppe
X_concat_train = df_concat_train.drop(columns=['Patch_X', 'Patch_Y', 'label'])
y_concat_train = df_concat_train['label']
X_concat_test = df_concat_test.drop(columns=['Patch_X', 'Patch_Y', 'label'])
y_concat_test = df_concat_test['label']

def plot_confusion_matrix(matrix, title, class_names, fname, normalize=False):
    plt.figure(figsize=(8, 7))
    cmap = plt.cm.Blues
    im = plt.imshow(matrix, cmap=cmap, vmin=0, vmax=(1.0 if normalize else None))
    plt.title(title, fontsize=18)
    plt.xlabel("Predicted label", fontsize=16)
    plt.ylabel("True label", fontsize=16)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha='right', fontsize=14)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=14)
    fmt = ".2f" if normalize else "d"
    thresh = matrix.max() / 2. if not normalize else 0.5
    # Zahlen mit Kontrast (weiß auf dunkel, schwarz auf hell)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > thresh else "black"
            plt.text(j, i, format(value, fmt),
                     ha="center", va="center",
                     color=color, fontsize=18, fontweight='bold')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(fname)
    plt.close()
    print(f"  Confusion matrix plot saved as {fname}")

def plot_and_save_confusion(y_true, y_pred, class_names, fname_base):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    plot_confusion_matrix(
        cm,
        "Confusion Matrix (absolute)",
        class_names,
        f"{fname_base}_confusion_matrix_absolute.png",
        normalize=False
    )
    plot_confusion_matrix(
        cm_norm,
        "Confusion Matrix (row-normalized)",
        class_names,
        f"{fname_base}_confusion_matrix_normalized.png",
        normalize=True
    )

def plot_and_save_loss_curve(loss_curve, fname):
    plt.figure()
    plt.plot(loss_curve, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"  Loss curve saved as {fname}")

def evaluate_and_report(model, X_train, y_train, X_test, y_test, name):
    # Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # Log-Loss auf Testdaten
    y_test_prob = model.predict_proba(X_test)
    test_logloss = log_loss(y_test, y_test_prob)

    # Iterationen
    n_iter = model.n_iter_

    print(f"\n=== Ergebnisse für {name} ===")
    print(f"  Benötigte Iterationen      : {n_iter}")
    print(f"  Trainings-Accuracy        : {train_acc:.4f}")
    print(f"  Test-Accuracy             : {test_acc:.4f}")
    print(f"  Test-Log-Loss             : {test_logloss:.4f}")

    # Loss-Kurve
    plot_and_save_loss_curve(model.loss_curve_, f"{name}_loss_curve.png")

    # Confusion Matrix (zwei einzelne Plots, wie gewünscht)
    class_names = np.unique(np.concatenate([y_train, y_test]))
    plot_and_save_confusion(y_test, model.predict(X_test), class_names, name)

# ----- Modelltraining: Single Patch -----
print("\nStarte Training für Single Patch Embedding ...")
mlp_single = MLPClassifier(
    hidden_layer_sizes=(512, 128),
    activation='relu',
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=42,
    verbose=True,
)
mlp_single.fit(X_single_train, y_single_train)

# Modell evaluieren
evaluate_and_report(mlp_single, X_single_train, y_single_train, X_single_test, y_single_test, "single_patch")

# Optional: Modell speichern
joblib.dump(mlp_single, "mlp_single_patch.joblib")

# ---- SPEICHERE Testdaten mit Vorhersagen (y_pred) für Single Patch ----
y_pred_single = mlp_single.predict(X_single_test)
df_single_test['y_pred'] = y_pred_single
df_single_test.to_csv("single_patch_preds.csv", index=False)
print("Testdaten mit y_pred für Single Patch gespeichert als 'single_patch_preds.csv'.")

# ----- Modelltraining: 9er-Gruppe -----
print("\nStarte Training für konkatenierten 9er Patch ...")
mlp_concat = MLPClassifier(
    hidden_layer_sizes=(2048, 256),
    activation='relu',
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=42,
    verbose=True,
)
mlp_concat.fit(X_concat_train, y_concat_train)

# Modell evaluieren
evaluate_and_report(mlp_concat, X_concat_train, y_concat_train, X_concat_test, y_concat_test, "concat_patch")

# Optional: Modell speichern
joblib.dump(mlp_concat, "mlp_concat_patch.joblib")

# ---- SPEICHERE Testdaten mit Vorhersagen (y_pred) für 9er-Gruppe ----
y_pred_concat = mlp_concat.predict(X_concat_test)
df_concat_test['y_pred'] = y_pred_concat
df_concat_test.to_csv("concat_patch_preds.csv", index=False)
print("Testdaten mit y_pred für 9er-Gruppe gespeichert als 'concat_patch_preds.csv'.")

