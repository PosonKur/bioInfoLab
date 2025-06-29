pathology_rows = adataDF

# Read WSI_patch_embeddings_Prostate_Anicar_Cancer_1.csv
with open('WSI_patch_embeddings_adenocarcinoma.csv', newline='', encoding='utf-8') as f:
    patch_reader = csv.reader(f)
    patch_header = next(patch_reader)
    patch_rows = [dict(zip(patch_header, row)) for row in patch_reader]

# Find which pathology coordinates are in which patch
patch_dict, patch_data = find_patch_for_coordinates(pathology_rows, patch_rows)

# Write the results to a new CSV file
output_file = 'patches_with_majority_pathology.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as outf:
    # Embedding columns are index 0 to 1535 in the patch file
    embedding_cols = patch_header[0:1536]
    fieldnames = ['Patch_X', 'Patch_Y'] + embedding_cols + ['Majority_Pathology']
    writer = csv.DictWriter(outf, fieldnames=fieldnames)
    writer.writeheader()
    for patch_key, entries in patch_dict.items():
        patch_entry = patch_data[patch_key]
        patch_x, patch_y = patch_key
        patch_center = (patch_x + 112, patch_y + 112)
        majority_pathology = majority_vote_pathology(entries, patch_center)
        row = {'Patch_X': patch_x, 'Patch_Y': patch_y, 'Majority_Pathology': majority_pathology}
        for col in embedding_cols:
            row[col] = patch_entry.get(col, '')
        writer.writerow(row)
