# main.py

import os
import random
import glob
import numpy as np
import torch
from tqdm import tqdm # Import tqdm for a progress bar during filtering

# --- NOTE: Ensure you have installed these libraries in your Colab session ---
# !pip install flwr torch torchvision scikit-learn tqdm

# Import necessary components from other project files
from model import ClientDataset, get_model 
import Case1_Fedavg # FedAvg
import Case2_FedDriftEager # FedDrift-Eager
import Case3_CentralizedBaseline # Centralized

# --- I. CONFIGURATION ---

# CRITICAL: Adjust this path to where your PetImages folder is located on your mounted Drive
DRIVE_BASE_PATH = "/content/drive/MyDrive/F25/COMS 4776 - NNDL/NNDL Project/data/PetImages" 

VAL_SPLIT_RATIO = 0.2
NUM_CLIENTS = 4

# --- II. DATA UTILITIES FOR CLEANUP ---

def test_corrupted_image(path):
    """Returns True if the image loads successfully, False otherwise."""
    try:
        # Use the same image loading logic as the ClientDataset
        from PIL import Image
        Image.open(path).convert('RGB')
        return True
    except Exception:
        return False

def filter_corrupted_images(paths, label):
    """Filters out all images that fail to load from a list of paths."""
    print(f"Cleaning {label} data ({len(paths)} paths)...")
    clean_paths = []
    # Use tqdm to show a progress bar during this slow filtering step
    for path in tqdm(paths):
        if os.path.getsize(path) > 0 and test_corrupted_image(path):
            clean_paths.append(path)
    print(f"Cleaned {label} paths: {len(clean_paths)}")
    return clean_paths


# --- III. DATA PARTITIONING LOGIC (Extreme Label Skew) ---

def partition_data(data_root_dir, val_split_ratio=VAL_SPLIT_RATIO, num_clients=NUM_CLIENTS):
    """
    Loads all Cat/Dog image file paths, implements Extreme Label Skew (2 Clients per class), 
    and splits each client's unique shard into non-overlapping training and validation sets.
    """
    print(f"Loading data paths from: {data_root_dir}")
    
    # 1. Load all file paths
    cat_dir = os.path.join(data_root_dir, 'Cat')
    dog_dir = os.path.join(data_root_dir, 'Dog')
    
    all_cats = glob.glob(os.path.join(cat_dir, '*.jpg'))
    all_dogs = glob.glob(os.path.join(dog_dir, '*.jpg'))
    
    # 2. FILTER CORRUPTED FILES (New Step for Efficiency)
    all_cats = filter_corrupted_images(all_cats, 'Cat')
    all_dogs = filter_corrupted_images(all_dogs, 'Dog')

    np.random.seed(42)
    np.random.shuffle(all_cats)
    np.random.shuffle(all_dogs)

    # 3. Extreme Label Skew Partition (4 non-overlapping shards)
    half_cats = len(all_cats) // 2
    half_dogs = len(all_dogs) // 2

    # Clients 1 & 2 are 100% Cat; Clients 3 & 4 are 100% Dog
    client_paths = {
        1: all_cats[:half_cats],      # C1: Cats, Subset A (Train/Val)
        2: all_cats[half_cats:],      # C2: Cats, Subset B (Train/Val)
        3: all_dogs[:half_dogs],      # C3: Dogs, Subset A (Train/Val)
        4: all_dogs[half_dogs:],      # C4: Dogs, Subset B (Train/Val)
    }
    
    # 4. Create Train/Validation Splits for each client
    client_datasets = {}
    for client_id, paths in client_paths.items():
        if not paths: continue

        split_idx = int(len(paths) * (1 - VAL_SPLIT_RATIO))
        train_paths = paths[:split_idx]
        val_paths = paths[split_idx:]

        client_datasets[client_id] = (ClientDataset(train_paths), ClientDataset(val_paths))
        
    print(f"--- Partition Summary ---")
    print(f"Total Clean Images Loaded: {len(all_cats) + len(all_dogs)}")
    print(f"Clients 1 & 2 (Cat) Train Samples: {len(client_datasets[1][0])} each.")
    print(f"Clients 3 & 4 (Dog) Train Samples: {len(client_datasets[3][0])} each.")
    print(f"Data partitioning complete.")
    
    return client_datasets


# --- IV. MAIN EXECUTION LOGIC (Unchanged) ---

def run_flwr_simulation(client_datasets, case_module):
    """Orchestrates the simulation based on the imported case module."""
    
    if case_module.__name__ == 'Case1_Fedavg':
        return case_module.start_fedavg_simulation(client_datasets, DRIVE_BASE_PATH)
    elif case_module.__name__ == 'Case2_FedDriftEager':
        return case_module.start_feddrift_simulation(client_datasets, DRIVE_BASE_PATH)
    elif case_module.__name__ == 'Case3_CentralizedBaseline':
        return case_module.start_centralized_baseline(client_datasets, DRIVE_BASE_PATH)
    

if __name__ == "__main__":
    
    # ... (Execution Block remains the same) ...
    if not os.path.exists(DRIVE_BASE_PATH):
        print(f"FATAL ERROR: Data not found at {DRIVE_BASE_PATH}")
        print("Please ensure Google Drive is mounted and the DRIVE_BASE_PATH is correct.")
    else:
        # 1. Partition the data for all clients (SLOW filtering happens here once)
        all_client_data = partition_data(DRIVE_BASE_PATH)

        # 2. Run Centralized Baseline (Case 3) - Should now be much faster
        centralized_results = run_flwr_simulation(all_client_data, Case3_CentralizedBaseline)

        # 3. Run FedAvg Baseline (Case 1)
        fedavg_results = run_flwr_simulation(all_client_data, Case1_Fedavg)
        
        # 4. Run FedDrift-Eager (Case 2)
        feddrift_results = run_flwr_simulation(all_client_data, Case2_FedDriftEager)
        
        print("\nAll required simulations executed.")