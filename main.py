import os
import random
import glob
import numpy as np
# You will need to install flwr, torch, torchvision, scikit-learn in Colab
# !pip install flwr torch torchvision scikit-learn

# Import model and dataset utilities
from model import ClientDataset, get_model # Assuming you moved the utilities here

# --- CONFIGURATION ---
# CRITICAL: SET THE CORRECT PATH FOR YOUR MOUNTED DRIVE DATA
# Based on your Google Drive structure, this is the expected path:
DRIVE_BASE_PATH = "/content/drive/MyDrive/data/PetImages" # You may need to verify this path
VAL_SPLIT_RATIO = 0.2
NUM_CLIENTS = 4

# --- DATA PARTITIONING LOGIC (Extreme Label Skew) ---
def partition_data(data_root_dir, val_split_ratio=VAL_SPLIT_RATIO, num_clients=NUM_CLIENTS):
    
    # 1. Load all file paths
    cat_dir = os.path.join(data_root_dir, 'Cat')
    dog_dir = os.path.join(data_root_dir, 'Dog')
    
    # Use glob to find all JPG files quickly
    all_cats = glob.glob(os.path.join(cat_dir, '*.jpg'))
    all_dogs = glob.glob(os.path.join(dog_dir, '*.jpg'))
    
    # Simple check for corrupted files (file size > 0)
    all_cats = [f for f in all_cats if os.path.getsize(f) > 0]
    all_dogs = [f for f in all_dogs if os.path.getsize(f) > 0]

    random.shuffle(all_cats)
    random.shuffle(all_dogs)

    # 2. Extreme Label Skew Partition (4 non-overlapping shards)
    half_cats = len(all_cats) // 2
    half_dogs = len(all_dogs) // 2

    # Clients 1 & 2 are 100% Cat; Clients 3 & 4 are 100% Dog
    client_paths = {
        1: all_cats[:half_cats],      
        2: all_cats[half_cats:],      
        3: all_dogs[:half_dogs],      
        4: all_dogs[half_dogs:],      
    }
    
    # 3. Create Train/Validation Splits for each client
    client_datasets = {}
    for client_id, paths in client_paths.items():
        if not paths: continue

        # Split using list slicing (paths are already shuffled)
        split_idx = int(len(paths) * (1 - val_split_ratio))
        train_paths = paths[:split_idx]
        val_paths = paths[split_idx:]

        # Store (Training Dataset, Validation Dataset) tuple
        client_datasets[client_id] = (ClientDataset(train_paths), ClientDataset(val_paths))
        
    print(f"Data Partitioned: {len(client_datasets)} clients created.")
    return client_datasets


# --- MAIN EXECUTION LOGIC ---
if __name__ == "__main__":
    
    # NOTE: You MUST execute the Drive mount cell in Colab first.
    
    if not os.path.exists(DRIVE_BASE_PATH):
        print(f"FATAL ERROR: Data not found at {DRIVE_BASE_PATH}")
    else:
        # 1. Partition the data for all clients
        all_client_data = partition_data(DRIVE_BASE_PATH)

        # 2. Run the simulations (Case1_Fedavg, Case2_FedDrift_Eager)
        # We will implement these cases next.
        
        # Example of how you will call the simulation:
        # fedavg_results = run_flwr_simulation(Case1_Fedavg, all_client_data)
        # feddrift_results = run_flwr_simulation(Case2_FedDrift_Eager, all_client_data)
        pass # Placeholder for the final Flower calls