import pandas as pd
import numpy as np

# Datei laden
file_path = "single_patch_train.csv"
df = pd.read_csv(file_path)

# Spaltennamen anzeigen
print("Spalten im DataFrame:")
print(df.columns.tolist())

# Dimension (Anzahl Zeilen & Spalten)
print("\nDimensionen des DataFrames:")
print(df.shape)  # (Anzahl Zeilen, Anzahl Spalten)

# Zeige die ersten 5 Zeilen als Beispiel
print("\nBeispiel-Inhalt (erste 5 Zeilen):")
print(df.head())

labels = np.unique(df['label'])
print(labels)
