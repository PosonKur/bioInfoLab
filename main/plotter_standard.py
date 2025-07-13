import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

def visualize_tissue_image_with_samples(image_path, data, width, height):
    """
    Visualize a tissue image in a plot scaled to fit exactly in the specified coordinate system
    and overlay data samples from a CSV file.

    Args:
        image_path (str): Path to the PNG image.
        csv_path (str): Path to the CSV file containing sample data.
        width (int): Width of the coordinate system.
        height (int): Height of the coordinate system.
    """
    # Load the image
    img = mpimg.imread(image_path)

    # Normalize the image if it's too dark or black
    if img.max() > 1:  # Assuming the image might have pixel values in 0-255 range
        img = img / 255.0

    # Load the CSV data
    x_coords = data['Patch_X']  # Read x-coordinates from column 'X'
    y_coords = data['Patch_Y']  # Read y-coordinates from column 'Y'

    # Create a figure with the specified dimensions
    fig, ax = plt.subplots(figsize=(10, 10))  # Fixed figure size for display

    # Display the image with aspect ratio set to 'auto'
    ax.imshow(img, extent=[0, width, height, 0], aspect='auto')  # Flip Y-axis to place (0,0) at the top-left

    # Set axis limits to match the coordinate system
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert Y-axis

    # Calculate the resolution of the axis in pixels per unit
    x_resolution = width / fig.get_size_inches()[0]  # Pixels per inch in the x-axis
    y_resolution = height / fig.get_size_inches()[1]  # Pixels per inch in the y-axis

    # Calculate the spot size based on the resolution
    spot_diameter_in_pixels = 188  # Diameter in pixels
    spot_size = (spot_diameter_in_pixels ** 2) / ((x_resolution + y_resolution) / 2)  # Adjust for resolution

    # Overlay the sample points with adjusted size
    ax.scatter(x_coords, y_coords, c='red', s=spot_size, label='Sample Points')

    # Add a fine grid where each rectangle equals 224x224
    for x in range(0, width, 224):
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in range(0, height, 224):
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

    # Set axis labels
    ax.set_xlabel("X-axis (pixels)")
    ax.set_ylabel("Y-axis (pixels)")

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()




def visualize_tissue_image_with_samples_color_labels(image_path, data, width, height):

    patch_size = 448  # Size of the patches in pixels

    # Load the image
    img = mpimg.imread(image_path)

    # Normalize the image if it's too dark or black
    if img.max() > 1:  # Assuming the image might have pixel values in 0-255 range
        img = img / 255.0

    # Get colors
    unique_labels = data['label'].unique()
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # Load the CSV data
    x_coords = data['Patch_X']  # Read x-coordinates from column 'X'
    y_coords = data['Patch_Y']  # Read y-coordinates from column 'Y'

    # Create a figure with the specified dimensions
    fig, ax = plt.subplots(figsize=(10, 10))  # Fixed figure size for display

    # Display the image with aspect ratio set to 'auto'
    ax.imshow(img, extent=[0, width, height, 0], aspect='auto')  # Flip Y-axis to place (0,0) at the top-left

    # Set axis limits to match the coordinate system
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert Y-axis

    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        label = data['label'].iloc[i]
        color = color_map[label]
        
        # Create rectangle with (x,y) as upper-left corner
        rect = Rectangle((x, y), patch_size, patch_size, 
                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

    # Add a fine grid where each rectangle equals 224x224
    for x in range(0, width, patch_size):
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in range(0, height, patch_size):
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

    # Set axis labels
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    
    # Create legend
    legend_patches = [mpatches.Patch(color=color_map[label], label=label) for label in unique_labels]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster")

    # Show the plot
    plt.show()