import scanpy as sc
import pandas as pd
import csv
import math
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from PIL import Image

spot_file="Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix_adenocarcinoma.h5"
cluster_file = 'clusters_leiden_0.5.csv'
output_file = 'patches_with_Majority_Cluster_4x4-grid_standard-224_adenocarcinoma.csv'
embedding_file = 'WSI_patch_embeddings_adenocarcinoma.csv'



#adata = #sc.read_10x_h5("./data/filtered_feature_bc_matrix.h5") # Reading the 10x h5 file
adata = sc.read_visium("./",count_file=spot_file) # Reading the 10x h5 file
adata.var_names_make_unique() # Making the data unique
adata.obs_names_make_unique() # Making the obs names 


# functions
def circle_patch_overlap_percentage_grid(patch_x, patch_y, x, y, radius=188.5979256687412 / 2, patch_size=224, grid_size=4):
    """
    Deterministically estimate the percentage of the circle overlapping with the patch using a grid.
    """
    inside_circle = 0
    inside_both = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Grid point in bounding box of circle
            gx = x - radius + 2 * radius * i / (grid_size - 1)
            gy = y - radius + 2 * radius * j / (grid_size - 1)
            # Check if inside circle
            if (gx - x) ** 2 + (gy - y) ** 2 <= radius ** 2:
                inside_circle += 1
                # Check if also inside patch
                if patch_x <= gx < patch_x + patch_size and patch_y <= gy < patch_y + patch_size:
                    inside_both += 1
    if inside_circle == 0:
        return 0.0
    return inside_both / inside_circle

def find_patch_for_coordinates(cluster_rows, patch_rows):
    """
    cluster_rows: list of dicts, each with 'X' and 'Y' keys (from combined_cluster_tissue.csv)
    patch_rows: list of dicts, each with 'X' and 'Y' keys (from WSI_patch_embeddings_Prostate_Anicar_Cancer_1.csv)
    Returns: dict with patch (X, Y) tuple as key and list of (cluster_row, overlap_percentage) tuples as value, and a dict mapping patch_key to patch_row
    """
    patch_dict = {}
    patch_data = {}
    for patch in patch_rows:
        patch_x = int(patch['X'])
        patch_y = int(patch['Y'])
        patch_key = (patch_x, patch_y)
        patch_data[patch_key] = patch


    spot_num = 0
    for idx, path_row in cluster_rows.iterrows():
        x = int(path_row['X'])
        y = int(path_row['Y'])
        for patch_key, patch in patch_data.items():
            patch_x, patch_y = patch_key
            overlap_percentage = circle_patch_overlap_percentage_grid(
                patch_x, patch_y, x, y
            )
            if overlap_percentage > 0:
                if patch_key not in patch_dict:
                    patch_dict[patch_key] = []
                patch_dict[patch_key].append((path_row, overlap_percentage))

        print(f"Spot {spot_num} of {len(cluster_rows)}")
        spot_num += 1

    return patch_dict, patch_data

def majority_vote_cluster(entries, patch_center):
    """
    Determine the majority cluster label for a patch based on weighted votes from overlapping cluster entries.
    Tie-breaking is done by selecting the cluster entry closest to the patch center.
    
    entries: list of (cluster_entry, overlap_percentage)
    patch_center: (patch_x, patch_y)
    
    Returns: (label, confidence_score)
    """
    votes = defaultdict(float)
    total_overlap = 0
    for entry, overlap in entries:
        votes[entry['Cluster']] += overlap
        total_overlap += overlap
    
    if not votes:
        return None, 0.0
    
    max_vote = max(votes.values())
    second_max_vote = sorted(votes.values(), reverse=True)[1] if len(votes) > 1 else 0
    
    vote_ratio = max_vote / total_overlap if total_overlap > 0 else 0
    vote_margin = (max_vote - second_max_vote) / total_overlap if total_overlap > 0 else 0
    
    # Patch and circle geometry
    patch_size = 224
    patch_area = patch_size * patch_size
    circle_diameter = 188.5979256687412
    circle_radius = circle_diameter / 2
    circle_area = math.pi * (circle_radius ** 2)
    
    # Theoretical maximum overlap percentage for a patch (patch area / circle area)
    max_total_percentage = min(patch_area / circle_area, 1.0)
    # Normalize by this value
    overlap_factor = min(total_overlap / max_total_percentage, 1.0) if max_total_percentage > 0 else 0
    
    confidence = ((vote_ratio + vote_margin) / 2) * overlap_factor
    
    candidates = [k for k, v in votes.items() if v == max_vote]
    if len(candidates) == 1:
        return candidates[0], confidence
    
    min_dist = float('inf')
    closest_cluster = None
    for entry, overlap in entries:
        if entry['Cluster'] in candidates:
            x = int(entry['X'])
            y = int(entry['Y'])
            dist = (x - patch_center[0]) ** 2 + (y - patch_center[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_cluster = entry['Cluster']
    return closest_cluster, confidence * 0.7




# Read in h5 and cluster data


# createa a pandas dataframe from the adata object
adata.obs['X'] = adata.obsm['spatial'][:, 0] #* 0.07277491
adata.obs['Y'] = adata.obsm['spatial'][:, 1] #* 0.07277491
# Convert to DataFrame
adata = adata.obs.reset_index()
# Rename columns for clarity
adata.rename(columns={'index': 'Barcode'}, inplace=True)
# Convert to a DataFrame
adataDF = pd.DataFrame(adata)



print("reading in cluster_file")

cluster_dict = {}
with open(cluster_file, newline='', encoding='utf-8') as pf:
    reader = csv.reader(pf)
    header_p = next(reader)
    for row in reader:
        if len(row) > 1 and row[1].strip():
            cluster_dict[row[0]] = row


print("match barcodes")
            
# for each entry in adataDF, add a new column with the cluster information by matching the Barcode
for index, row in adataDF.iterrows():
    barcode = row['Barcode']
    if barcode in cluster_dict:
        cluster_info = cluster_dict[barcode]  # Skip the first column (Barcode)
        adataDF.at[index, "Cluster"] = cluster_info[1]
    else:
        #  delte the row if the barcode is not found in the cluster_dict
        adataDF.drop(index, inplace=True)


cluster_rows = adataDF

print("reading embedding file")

# Read WSI_patch_embeddings_Prostate_Anicar_Cancer_1.csv
with open(embedding_file, newline='', encoding='utf-8') as f:
    patch_reader = csv.reader(f)
    patch_header = next(patch_reader)
    patch_rows = [dict(zip(patch_header, row)) for row in patch_reader]

print("find patch coordinates")
# Find which cluster coordinates are in which patch
patch_dict, patch_data = find_patch_for_coordinates(cluster_rows, patch_rows)

# Write the results to a new CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as outf:
    # Embedding columns are index 0 to 1535 in the patch file
    embedding_cols = patch_header[0:1536]
    fieldnames = ['Patch_X', 'Patch_Y'] + embedding_cols + ['label', 'confidence']
    writer = csv.DictWriter(outf, fieldnames=fieldnames)
    writer.writeheader()

    patch_num = 0
    for patch_key, entries in patch_dict.items():
        patch_entry = patch_data[patch_key]
        patch_x, patch_y = patch_key
        patch_center = (patch_x + 112, patch_y + 112)
        label, confidence = majority_vote_cluster(entries, patch_center)
        row = {'Patch_X': patch_x, 'Patch_Y': patch_y, 'label': label, 'confidence': confidence}
        for col in embedding_cols:
            row[col] = patch_entry.get(col, '')
        writer.writerow(row)

        print(f"Patch {patch_num} of {len(patch_dict.items())}")
        patch_num += 1


