
import pandas as pd
import os

# Define file paths

clusters_csv_path = "clusters_leiden_0.5.csv"
wsi_csv_path = "WSI_patch_embeddings_centered-224_adenocarcinoma.csv"
output_path = "WSI_patch_embeddings_centered-224_adenocarcinoma_training-data.csv"
tissue_csv_path = "spatial/tissue_positions_list.csv"

def create_barcode_mapping(tissue_csv_path, clusters_csv_path):
    """
    Create a pandas DataFrame that maps barcodes from clusters CSV to tissue positions CSV.
    
    Args:
        tissue_csv_path (str): Path to the tissue_positions_list.csv file
        clusters_csv_path (str): Path to the clusters CSV file
    
    Returns:
        pd.DataFrame: DataFrame with columns [barcode, cluster, X, Y]
    """
    
    # Read tissue positions CSV
    # Columns: barcode, col1, col2, col3, Y, X (based on the sample data)
    tissue_df = pd.read_csv(tissue_csv_path, header=None, 
                           names=['barcode', 'col1', 'col2', 'col3', 'Y', 'X'])
    
    # Read clusters CSV
    clusters_df = pd.read_csv(clusters_csv_path)
    
    # Merge the dataframes on barcode
    merged_df = pd.merge(clusters_df, tissue_df[['barcode', 'X', 'Y']], 
                        on='barcode', how='inner')
    
    # Reorder columns for clarity
    result_df = merged_df[['barcode', 'cluster', 'X', 'Y']]
    
    print(f"Total barcodes in clusters CSV: {len(clusters_df)}")
    print(f"Total barcodes in tissue CSV: {len(tissue_df)}")
    print(f"Matched barcodes: {len(result_df)}")
    
    return result_df

def map_to_wsi_embeddings(barcode_mapping_df, wsi_csv_path):
    """
    Map the barcode/cluster DataFrame to WSI embeddings via X and Y coordinates.
    
    Args:
        barcode_mapping_df (pd.DataFrame): DataFrame with barcode, cluster, X, Y columns
        wsi_csv_path (str): Path to the WSI embeddings CSV file
    
    Returns:
        pd.DataFrame: DataFrame with X, Y, embedding columns (0-1535), and cluster as label
    """
    
    print("Reading WSI embeddings CSV...")
    # Read WSI CSV
    wsi_df = pd.read_csv(wsi_csv_path)
    
    print(f"WSI CSV shape: {wsi_df.shape}")
    print(f"Barcode mapping shape: {barcode_mapping_df.shape}")
    
    # Merge on X and Y coordinates
    merged_df = pd.merge(barcode_mapping_df[['cluster', 'X', 'Y']], 
                        wsi_df, 
                        on=['X', 'Y'], 
                        how='inner')
    
    print(f"Matched rows: {len(merged_df)}")
    
    # Select the desired columns: X, Y, embedding features (0-1535), and cluster as label
    embedding_columns = [str(i) for i in range(1536)]  # columns 0-1535
    result_columns = ['X', 'Y'] + embedding_columns + ['cluster']
    
    # Rename columns for final output
    final_df = merged_df[['X', 'Y'] + embedding_columns + ['cluster']].copy()
    final_df.rename(columns={'cluster': 'label', 'X': 'Patch_X', 'Y': 'Patch_Y'}, inplace=True)
    
    return final_df



# Create the mapping
mapping_df = create_barcode_mapping(tissue_csv_path, clusters_csv_path)

# Map to WSI embeddings
final_df = map_to_wsi_embeddings(mapping_df, wsi_csv_path)

# Display results
print(f"\nFinal DataFrame shape: {final_df.shape}")
print(f"Columns: {list(final_df.columns)}")
print(f"\nFirst 5 rows:")
print(final_df.head())

# Save the result

final_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
    
