import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np
import tifffile
from uni import get_encoder
import tifffile
import numpy as np
import os
import pandas as pd
from PIL import Image
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics
from uni.downstream.utils import concat_images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################

import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################

device

##############################################

method = "centered_448" #Method to extract the patches
tissue = "adenocarcinoma" #Type of tissue

wsi_folder = f"imgs/{tissue}/" #Where the img is located
output_base_folder = f"{method}/Patches/" #Where you want to save the patches
embedding_file = f"{method}/{tissue}/WSI_patch_embeddings.csv" #name of the file with the embeddings
metadata_file = f"{method}/{tissue}/WSI_patch_metadata.csv"   #name of the fiel(optional)
local_dir = 'checkpoint_path' #Where the model is located
spots_positions_file = f"spot_positions/{tissue}/tissue_positions_list.csv" #Where the spot positions are located

os.makedirs(f"{method}", exist_ok=True)
os.makedirs(f"{method}/{tissue}", exist_ok=True)
os.makedirs(f"{method}/Patches", exist_ok=True)

#########################################################


# Load the data
tissue_positions_df = pd.read_csv(spots_positions_file, header=0, names=["Barcode", 0, 1, 2, "Y", "X"])
tissue_positions = tissue_positions_df[["Barcode", "X", "Y"]]

# Use .loc to explicitly set values on the DataFrame slice
tissue_positions.loc[:, 'X'] = pd.to_numeric(tissue_positions['X'], errors='coerce').fillna(0).astype(int)
tissue_positions.loc[:, 'Y'] = pd.to_numeric(tissue_positions['Y'], errors='coerce').fillna(0).astype(int)

###########################################################

# Adjusted code to take 448x448 patches into account
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

#login()

#os.makedirs(local_dir, exist_ok=True)
#hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
timm_kwargs = {
    'model_name': 'vit_giant_patch14_224',
    'img_size': 448,  # Updated to match new patch size
    'patch_size': 14,
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5,
    'embed_dim': 1536,
    'mlp_ratio': 2.66667 * 2,
    'num_classes': 0,
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked,
    'act_layer': torch.nn.SiLU,
    'reg_tokens': 8,
    'dynamic_img_size': True
}

model = timm.create_model(**timm_kwargs)

model.eval().to(device)

# Load weights
state_dict = torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu")
# Remove old pos_embed to avoid size mismatch
old_pos_embed = state_dict.pop('pos_embed')

# Load other weights
missing, unexpected = model.load_state_dict(state_dict, strict=False)

# Interpolate pos_embed to new grid
def interpolate_pos_embed(old_pos_embed, model, new_img_size):
    # Exclude class/reg tokens if any
    num_extra_tokens = model.pos_embed.shape[1] - model.patch_embed.num_patches
    extra_tokens = old_pos_embed[:, :num_extra_tokens]
    old_grid = old_pos_embed[:, num_extra_tokens:]

    gs_old = int(old_grid.shape[1] ** 0.5)
    gs_new = new_img_size // model.patch_embed.patch_size[0]

    old_grid = old_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # [1, dim, H, W]
    new_grid = torch.nn.functional.interpolate(old_grid, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
    new_grid = new_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)

    new_pos_embed = torch.cat((extra_tokens, new_grid), dim=1)
    return new_pos_embed

# Interpolate and assign
model.pos_embed = torch.nn.Parameter(
    interpolate_pos_embed(old_pos_embed, model, 448)
)

# Define transform
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


#######################################################################

patch_size = (448, 448)  
stride = 224  
root_path = output_base_folder


wsi_files = [f for f in os.listdir(wsi_folder) if f.endswith(".tif")]
failed_wsi_files = []
all_patch_metadata = []  

for wsi_file in wsi_files:
    wsi_path = os.path.join(wsi_folder, wsi_file)
    
    try:
    
        wsi_image = tifffile.imread(wsi_path)
        
      
        wsi_image = Image.fromarray(wsi_image)  

  
        slide_name = os.path.splitext(wsi_file)[0]
        slide_output_folder = os.path.join(output_base_folder, slide_name)
        os.makedirs(slide_output_folder, exist_ok=True)

    
        wsi_width, wsi_height = wsi_image.size
        print(f"Processing {slide_name} - Size: {wsi_width}x{wsi_height}")

        saved_patches = 0
        # Modify the patch generation logic to center patches around X and Y coordinates from tissue_positions_filtered
        for index, row in tissue_positions.iterrows():
            x_center, y_center = row['X'], row['Y']
            x_start = max(0, x_center - patch_size[0] // 2)
            y_start = max(0, y_center - patch_size[1] // 2)
            x_end = min(wsi_width, x_start + patch_size[0])
            y_end = min(wsi_height, y_start + patch_size[1])

            patch = wsi_image.crop((x_start, y_start, x_end, y_end))

            patch_filename = f"patch_{saved_patches+1}.png"
            patch_path = os.path.join(slide_output_folder, patch_filename)
            patch.save(patch_path)

            all_patch_metadata.append([slide_name, patch_filename, x_center, y_center, patch_path])
            saved_patches += 1

        print(f"Saved {saved_patches} patches for {slide_name}")

    except Exception as e:
        print(f"Error processing {wsi_file}: {e}")
        failed_wsi_files.append(wsi_file)


metadata_df = pd.DataFrame(all_patch_metadata, columns=['Slide_ID', 'Patch_ID', 'X', 'Y', 'Path'])
metadata_df.to_csv(metadata_file, index=False)

print("All WSIs processed successfully!")
print("Failed WSIs:", failed_wsi_files)
print(f"Metadata saved in {metadata_file}")

##########################################################33

from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import os
import pandas as pd
import torch

class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, "*.png"))  # Load all PNGs
        self.transform = transform
        self.patch_ids = [os.path.basename(img) for img in self.image_paths]  # Store Patch_IDs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels

        if self.transform:
            image = self.transform(image)
            
        return image, -1  




valid_folders = [f for f in sorted(os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, f)) and not f.startswith(".")]

all_embeddings = []
all_sample_ids = []
all_patch_ids = []  

# Ensure all tensors and the model are moved to the same device
model.to(device)

# Update the patch generation logic to move tensors to the correct device
for folder in valid_folders:
    folder_path = os.path.join(root_path, folder)

    test_dataset = PatchDataset(root_dir=folder_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    test_features = extract_patch_features_from_dataloader(model, test_dataloader)
    test_feats = torch.Tensor(test_features['embeddings']).to(device)  # Move embeddings to the same device

    all_embeddings.append(test_feats.cpu().numpy())  # Convert to NumPy after moving to CPU
    all_sample_ids.extend([folder] * test_feats.shape[0])  # Assign folder name to each patch
    all_patch_ids.extend(test_dataset.patch_ids)  


df_embeddings = pd.DataFrame(
    torch.cat([torch.tensor(arr) for arr in all_embeddings], dim=0).numpy()  # Flatten list of arrays
)
df_embeddings["Slide_ID"] = all_sample_ids 
df_embeddings["Patch_ID"] = all_patch_ids  


df_embeddings['match_id']=df_embeddings['Slide_ID']+'_'+df_embeddings['Patch_ID']
metadata_df['match_id']=metadata_df['Slide_ID']+'_'+metadata_df['Patch_ID']
df_embeddings=df_embeddings.merge(metadata_df[['match_id','X','Y','Patch_ID']],on='match_id')
df_embeddings.to_csv(embedding_file,index=False) #final file

