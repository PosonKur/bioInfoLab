import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

print("Lese CSV ein...")
start = time.time()
input_csv = "../WSI_patch_embeddings_standard-224_adenocarcinoma_leiden_0.3_training-data/WSI_patch_embeddings_standard-224_adenocarcinoma_leiden_0.3_training-data.csv"   # Pfad zu deiner Inputdatei!
df = pd.read_csv(input_csv)
print(f"CSV eingelesen in {time.time()-start:.1f} Sekunden. Anzahl Zeilen: {len(df)}")

embedding_cols = [str(i) for i in range(1536)]
required_cols = ['Patch_X', 'Patch_Y', *embedding_cols, 'label']

print("Filtere nur valide Patches (ohne NaN in Embedding oder Label)...")
start = time.time()
df_valid = df.dropna(subset=required_cols)
print(f"Validierung fertig in {time.time()-start:.1f} Sekunden. Gültige Patches: {len(df_valid)}")
df_valid['Patch_X'] = df_valid['Patch_X'].astype(int)
df_valid['Patch_Y'] = df_valid['Patch_Y'].astype(int)

print("Baue Lookup-Tabelle für Patch-Positionen...")
start = time.time()
patch_lookup = {(row.Patch_X, row.Patch_Y): row for _, row in df_valid.iterrows()}
print(f"Lookup-Tabelle gebaut in {time.time()-start:.1f} Sekunden. Patch-Keys: {len(patch_lookup)}")

PATCH_SIZE = 224
neighbor_offsets = [
    (0, 0),  # Mitte
    (0, -PATCH_SIZE),        # Norden
    (PATCH_SIZE, -PATCH_SIZE), # Nord-Ost
    (PATCH_SIZE, 0),        # Osten
    (PATCH_SIZE, PATCH_SIZE), # Süd-Ost
    (0, PATCH_SIZE),        # Süden
    (-PATCH_SIZE, PATCH_SIZE), # Süd-West
    (-PATCH_SIZE, 0),       # Westen
    (-PATCH_SIZE, -PATCH_SIZE) # Nord-West
]

print("Suche vollständige 9er-Gruppen...")
start = time.time()
groups = []
single_patch_rows = []
checked = 0
found = 0
total = len(patch_lookup)
report_every = max(1, total // 20)  # 5% steps

for (x, y), center_row in patch_lookup.items():
    group_rows = []
    for dx, dy in neighbor_offsets:
        nx, ny = x + dx, y + dy
        neighbor = patch_lookup.get((nx, ny))
        if neighbor is not None:
            group_rows.append(neighbor)
        else:
            break
    if len(group_rows) == 9:
        groups.append(group_rows)
        single_patch_rows.append(center_row)
        found += 1
    checked += 1
    if checked % report_every == 0 or checked == total:
        print(f"  Fortschritt: {checked}/{total} Patches geprüft, {found} vollständige Gruppen gefunden")

print(f"9er-Gruppen-Suche fertig in {time.time()-start:.1f} Sekunden. Insgesamt {found} vollständige Gruppen.")

print("Baue Einzel-Patch-DataFrame...")
start = time.time()
df_single = pd.DataFrame(single_patch_rows)
print(f"Einzel-Patch-DataFrame gebaut in {time.time()-start:.1f} Sekunden. Zeilen: {len(df_single)}")

print("Konstruiere konkatenierten 9er-Gruppen-DataFrame (kann dauern)...")
start = time.time()
concat_data = []
for i, group in enumerate(groups):
    row_dict = {}
    center = group[0]
    row_dict['Patch_X'] = center['Patch_X']
    row_dict['Patch_Y'] = center['Patch_Y']
    row_dict['label'] = center['label']
    for idx, patch in enumerate(group):
        for emb_idx in range(1536):
            colname = f"{emb_idx}_patch{idx}"
            row_dict[colname] = patch[str(emb_idx)]
    concat_data.append(row_dict)
    if (i+1) % max(1, len(groups)//10) == 0 or (i+1) == len(groups):
        print(f"  Fortschritt: {i+1}/{len(groups)} Gruppen verarbeitet")
df_concat = pd.DataFrame(concat_data)
print(f"Konkatenierten DataFrame gebaut in {time.time()-start:.1f} Sekunden. Zeilen: {len(df_concat)}")

print("Bereite Train/Test-Split vor...")
start = time.time()
df_single = df_single.reset_index(drop=True)
df_concat = df_concat.reset_index(drop=True)
labels = df_single['label'].values

X_single_train, X_single_test, idx_train, idx_test = train_test_split(
    df_single, df_single.index, test_size=0.2, random_state=42, stratify=labels
)
X_concat_train = df_concat.loc[idx_train].reset_index(drop=True)
X_concat_test = df_concat.loc[idx_test].reset_index(drop=True)
print(f"Train/Test-Split fertig in {time.time()-start:.1f} Sekunden.")
print(f"  Einzel-Patch: train={len(X_single_train)}, test={len(X_single_test)}")
print(f"  Konkateniert: train={len(X_concat_train)}, test={len(X_concat_test)}")

print("Speichere Ergebnisse als CSV...")
start = time.time()
X_single_train.to_csv("single_patch_train.csv", index=False)
X_single_test.to_csv("single_patch_test.csv", index=False)
X_concat_train.to_csv("concat_patch_train.csv", index=False)
X_concat_test.to_csv("concat_patch_test.csv", index=False)
print(f"CSV-Dateien gespeichert in {time.time()-start:.1f} Sekunden.")

print("Fertig!")

