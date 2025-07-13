import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

# ===== User Inputs =====
IMAGE_PATH = "tissue_hires_image.png"
SCALEFACTOR_JSON = "scalefactors_json.json"
SPOT_SIZE = 50

CASES = [
    {
        "name": "single_patch",
        "test_csv": "single_patch_test.csv",
        "pred_csv": "single_patch_preds.csv",
        "outprefix": "viz_classification_single",
        "title": "cell type cluster classification: single patch"
    },
    {
        "name": "concat_patch",
        "test_csv": "concat_patch_test.csv",
        "pred_csv": "concat_patch_preds.csv",
        "outprefix": "viz_classification_concat",
        "title": "cell type cluster classification: multiple patches"
    }
]

# Color palette for up to 10 labels; label==4 always pink/magenta.
CUSTOM_COLORS = [
    '#1f77b4',  # 0: blue
    '#ff7f0e',  # 1: orange
    '#2ca02c',  # 2: green
    '#d62728',  # 3: red
    '#e377c2',  # 4: pink/magenta (distinct for cluster 4)
    '#8c564b',  # 5: brown
    '#9467bd',  # 6: purple
    '#7f7f7f',  # 7: gray
    '#bcbd22',  # 8: yellow-green
    '#17becf'   # 9: cyan
]

def assign_colors(unique_labels):
    label2color = {}
    for i, lab in enumerate(sorted(unique_labels)):
        lab_int = int(lab)
        if lab_int == 4:
            label2color[lab] = '#e377c2'
        elif lab_int < len(CUSTOM_COLORS):
            label2color[lab] = CUSTOM_COLORS[lab_int]
        else:
            color_map = plt.cm.get_cmap('tab20', len(unique_labels))
            label2color[lab] = color_map(i)
    return label2color

# == Load image ==
img = np.array(Image.open(IMAGE_PATH))
h, w = img.shape[:2]
print('h = ', h)
print('w = ', w)

# == Load scalefactor ==
with open(SCALEFACTOR_JSON, 'r') as f:
    scalefactors = json.load(f)
sf = scalefactors['tissue_hires_scalef']
print('scalefactor =', sf)

PATCH_SIZE = 224

# --- First: always read concat labels and build the color mapping ---
concat_case = [case for case in CASES if case['name'] == 'concat_patch'][0]
df_concat_test = pd.read_csv(concat_case['test_csv'])
df_concat_pred = pd.read_csv(concat_case['pred_csv'])
if 'y_pred' not in df_concat_test.columns and 'y_pred' in df_concat_pred.columns:
    df_concat_test['y_pred'] = df_concat_pred['y_pred']
df_concat_test['y_pred'] = df_concat_test['y_pred'].astype(int)
df_concat_test['label'] = df_concat_test['label'].astype(int)
concat_unique_labels = np.unique(np.concatenate([df_concat_test['y_pred'].values, df_concat_test['label'].values]))
label2color = assign_colors(concat_unique_labels)
print("Color map for all plots:")
for lab in sorted(label2color.keys()):
    print(f"  Label {lab}: {label2color[lab]}")

for case in CASES:
    print(f"\n==== Processing case: {case['name']} ====")
    df_test = pd.read_csv(case['test_csv'])
    df_pred = pd.read_csv(case['pred_csv'])
    print("Coordinates min:\n", df_test[['Patch_X', 'Patch_Y']].min())
    print("Coordinates max:\n", df_test[['Patch_X', 'Patch_Y']].max())

    if 'y_pred' not in df_test.columns and 'y_pred' in df_pred.columns:
        df_test['y_pred'] = df_pred['y_pred']

    df_test['y_pred'] = df_test['y_pred'].astype(int)
    df_test['label'] = df_test['label'].astype(int)

    for col in ['Patch_X', 'Patch_Y', 'label', 'y_pred']:
        if col not in df_test.columns:
            raise ValueError(f"Column '{col}' missing in test CSV ({case['test_csv']})!")

    df_test['Patch_X_img'] = df_test['Patch_X'] * sf
    df_test['Patch_Y_img'] = df_test['Patch_Y'] * sf
    df_test['Patch_X_center'] = df_test['Patch_X_img'] + PATCH_SIZE * 0.5 * sf
    df_test['Patch_Y_center'] = df_test['Patch_Y_img'] + PATCH_SIZE * 0.5 * sf

    # == 1. Predicted labels color-coded ==
    plt.figure(figsize=(w/300, h/300), dpi=300)
    plt.imshow(img, extent=[0, w, h, 0])
    plt.xlim(0, w)
    plt.ylim(h, 0)
    for idx, row in df_test.iterrows():
        x = row['Patch_X_center']
        y = row['Patch_Y_center']
        plt.scatter(x, y, color=label2color[row['y_pred']], s=SPOT_SIZE, edgecolor='black', alpha=0.8)
    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=label2color[lab], label=str(lab), markersize=10)
               for lab in sorted(label2color)]
    plt.legend(handles=handles, title="Predicted label", bbox_to_anchor=(1.02,1), loc='upper left')
    plt.title(case['title'])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{case['outprefix']}_pred_labels_overlay.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {case['outprefix']}_pred_labels_overlay.png")

    # == 2. Correct vs. wrong ==
    plt.figure(figsize=(w/300, h/300), dpi=300)
    plt.imshow(img, extent=[0, w, h, 0])
    plt.xlim(0, w)
    plt.ylim(h, 0)
    for idx, row in df_test.iterrows():
        x = row['Patch_X_center']
        y = row['Patch_Y_center']
        color = (0, 1, 0) if row['label'] == row['y_pred'] else (1, 0, 0)
        plt.scatter(x, y, color=color, s=SPOT_SIZE, edgecolor='black', alpha=0.8)
    plt.title("Correct (green) / Wrong (red) classified patches")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{case['outprefix']}_correct_vs_wrong.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {case['outprefix']}_correct_vs_wrong.png")

# == 3. Combined Agreement Plot: green, blue, orange, pink ==
print("\n==== Processing combined agreement plot ====")

# -- Load both results --
df_single_test = pd.read_csv(CASES[0]['test_csv'])
df_single_pred = pd.read_csv(CASES[0]['pred_csv'])
if 'y_pred' not in df_single_test.columns and 'y_pred' in df_single_pred.columns:
    df_single_test['y_pred'] = df_single_pred['y_pred']

df_concat_test = pd.read_csv(CASES[1]['test_csv'])
df_concat_pred = pd.read_csv(CASES[1]['pred_csv'])
if 'y_pred' not in df_concat_test.columns and 'y_pred' in df_concat_pred.columns:
    df_concat_test['y_pred'] = df_concat_pred['y_pred']

# Type conversion and merging for matching
for col in ['y_pred', 'label']:
    df_single_test[col] = df_single_test[col].astype(int)
    df_concat_test[col] = df_concat_test[col].astype(int)

merge_cols = ['Patch_X', 'Patch_Y']

# Merge on Patch_X, Patch_Y and label (to ensure matching)
df_merge = pd.merge(
    df_single_test,
    df_concat_test,
    on=merge_cols + ['label'],
    suffixes=('_single', '_concat')
)

# Calculate coordinates for plotting
df_merge['Patch_X_img'] = df_merge['Patch_X'] * sf
df_merge['Patch_Y_img'] = df_merge['Patch_Y'] * sf
df_merge['Patch_X_center'] = df_merge['Patch_X_img'] + PATCH_SIZE * 0.5 * sf
df_merge['Patch_Y_center'] = df_merge['Patch_Y_img'] + PATCH_SIZE * 0.5 * sf

# Define color codes
color_green  = "#1aff1a"    # both correct (bright green)
color_blue   = "#1f77ff"    # both wrong (vivid blue)
color_orange = "#ff9900"    # only single correct, multiple wrong
color_pink   = "#ff33cc"    # only multiple correct, single wrong

def compare_and_get_color(row):
    single_correct = row['label'] == row['y_pred_single']
    concat_correct = row['label'] == row['y_pred_concat']
    if single_correct and concat_correct:
        return color_green
    elif not single_correct and not concat_correct:
        return color_blue
    elif single_correct and not concat_correct:
        return color_orange
    else:  # not single_correct and concat_correct
        return color_pink

df_merge['compare_color'] = df_merge.apply(compare_and_get_color, axis=1)

plt.figure(figsize=(w/300, h/300), dpi=300)
plt.imshow(img, extent=[0, w, h, 0])
plt.xlim(0, w)
plt.ylim(h, 0)
for idx, row in df_merge.iterrows():
    plt.scatter(row['Patch_X_center'], row['Patch_Y_center'], color=row['compare_color'],
                s=SPOT_SIZE, edgecolor='black', alpha=0.8)

import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color=color_green, label="both correct"),
    mpatches.Patch(color=color_blue, label="both wrong"),
    mpatches.Patch(color=color_orange, label="single correct, multiple wrong"),
    mpatches.Patch(color=color_pink, label="multiple correct, single wrong")
]
plt.legend(handles=legend_handles, title="Classification agreement", bbox_to_anchor=(1.02,1), loc='upper left')

plt.title("Comparison: classification agreement between single and multiple patches")
plt.axis('off')
plt.tight_layout()
plt.savefig("viz_classification_agreement.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_classification_agreement.png")

# == 4. Overlay: identical vs. differing predictions ==
print("\n==== Processing identical vs. differing prediction plot ====")
color_identic = "#00cc44"   # green for identical prediction (distinct, not too bright)
color_diff    = "#e60026"   # red for differing prediction (strong red)

def identical_prediction_color(row):
    return color_identic if row['y_pred_single'] == row['y_pred_concat'] else color_diff

df_merge['identical_pred_color'] = df_merge.apply(identical_prediction_color, axis=1)

plt.figure(figsize=(w/300, h/300), dpi=300)
plt.imshow(img, extent=[0, w, h, 0])
plt.xlim(0, w)
plt.ylim(h, 0)
for idx, row in df_merge.iterrows():
    plt.scatter(row['Patch_X_center'], row['Patch_Y_center'], color=row['identical_pred_color'],
                s=SPOT_SIZE, edgecolor='black', alpha=0.8)

legend_handles2 = [
    mpatches.Patch(color=color_identic, label="identical prediction"),
    mpatches.Patch(color=color_diff, label="differing prediction")
]
plt.legend(handles=legend_handles2, title="Prediction agreement", bbox_to_anchor=(1.02,1), loc='upper left')

plt.title("Agreement of predicted cell type: single vs. multiple patches")
plt.axis('off')
plt.tight_layout()
plt.savefig("viz_classification_pred_agreement.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: viz_classification_pred_agreement.png")

# == 5. Fluktuationsanalyse für label 4 → pred 1 (mit Debug-Ausgabe) ==
print("\n==== Fluctuation check: label 4 classified as 1 ====")
# Filter: true label==4 and prediction==1 in beiden Fällen
mask_single = (df_single_test['label'] == 4) & (df_single_test['y_pred'] == 1)
mask_concat = (df_concat_test['label'] == 4) & (df_concat_test['y_pred'] == 1)

print(f"\nDEBUG: In single_patch: {mask_single.sum()} datapoints with label==4 and y_pred==1.")
print(f"DEBUG: In concat_patch: {mask_concat.sum()} datapoints with label==4 and y_pred==1.")

df_single_fluct = df_single_test[mask_single][merge_cols].copy()
df_concat_fluct = df_concat_test[mask_concat][merge_cols].copy()

set_single = set([tuple(row) for row in df_single_fluct.values])
set_concat = set([tuple(row) for row in df_concat_fluct.values])

both = set_single & set_concat
only_single = set_single - set_concat
only_concat = set_concat - set_single

print(f"\nNumber of test data points with label 4 classified as 1:")
print(f"  single_patch: {len(set_single)}")
print(f"  multiple patches: {len(set_concat)}")
print(f"  identical data points in both cases: {len(both)}")
print(f"  only in single_patch: {len(only_single)}")
print(f"  only in multiple patches: {len(only_concat)}")

if len(both) > 0:
    print("\nData points with label 4 classified as 1 in BOTH cases (Patch_X, Patch_Y):")
    print(sorted(list(both)))
else:
    print("\nNo identical data points misclassified as 1 in both cases.")

if len(only_single) > 0:
    print("\nData points with label 4 classified as 1 ONLY in single_patch (Patch_X, Patch_Y):")
    print(sorted(list(only_single)))

if len(only_concat) > 0:
    print("\nData points with label 4 classified as 1 ONLY in multiple patches (Patch_X, Patch_Y):")
    print(sorted(list(only_concat)))
else:
    print("\nNo unique data points for this misclassification in multiple patches.")


