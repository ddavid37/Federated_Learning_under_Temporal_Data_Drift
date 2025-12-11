import os
import random
import glob
import numpy as np
import torch

# --- NOTE: Ensure you have installed these libraries in your Colab session ---
# !pip install flwr torch torchvision scikit-learn

# Import necessary components from other project files
from model import ClientDataset, get_model 
import Case1_Fedavg # FedAvg
import Case2_FedDrift-Eager # New: FedDrift-Eager
import Case3_Centralized # Centralized

# --- I. CONFIGURATION ---

# CRITICAL: Adjust this path to where your PetImages folder is located on your mounted Drive
DRIVE_BASE_PATH = "/content/drive/MyDrive/COMSW4776_Fall25/Project/data/PetImages" 

VAL_SPLIT_RATIO = 0.2
NUM_CLIENTS = 4

# --- II. DATA PARTITIONING LOGIC (Extreme Label Skew) ---

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
    
    # Simple file cleanup (filtering out zero-byte files)
    all_cats = [f for f in all_cats if os.path.getsize(f) > 0 and 'Cat' in f]
    all_dogs = [f for f in all_dogs if os.path.getsize(f) > 0 and 'Dog' in f]

    np.random.seed(42)
    np.random.shuffle(all_cats)
    np.random.shuffle(all_dogs)

    # 2. Extreme Label Skew Partition (4 non-overlapping shards)
    half_cats = len(all_cats) // 2
    half_dogs = len(all_dogs) // 2

    # Clients 1 & 2 are 100% Cat; Clients 3 & 4 are 100% Dog
    client_paths = {
        1: all_cats[:half_cats],      # C1: Cats, Subset A (Train/Val)
        2: all_cats[half_cats:],      # C2: Cats, Subset B (Train/Val)
        3: all_dogs[:half_dogs],      # C3: Dogs, Subset A (Train/Val)
        4: all_dogs[half_dogs:],      # C4: Dogs, Subset B (Train/Val)
    }
    
    # 3. Create Train/Validation Splits for each client
    client_datasets = {}
    for client_id, paths in client_paths.items():
        if not paths: continue

        split_idx = int(len(paths) * (1 - VAL_SPLIT_RATIO))
        train_paths = paths[:split_idx]
        val_paths = paths[split_idx:]

        client_datasets[client_id] = (ClientDataset(train_paths), ClientDataset(val_paths))
        
    print(f"--- Partition Summary ---")
    print(f"Total Images Loaded: {len(all_cats) + len(all_dogs)}")
    print(f"Clients 1 & 2 (Cat) Train Samples: {len(client_datasets[1][0])} each.")
    print(f"Clients 3 & 4 (Dog) Train Samples: {len(client_datasets[3][0])} each.")
    print(f"Data partitioning complete.")
    
    return client_datasets


# --- III. MAIN EXECUTION LOGIC ---

def run_flwr_simulation(client_datasets, case_module):
    """Orchestrates the simulation based on the imported case module."""
    
    if case_module.__name__ == 'Case1_Fedavg':
        return case_module.start_fedavg_simulation(client_datasets, DRIVE_BASE_PATH)
    elif case_module.__name__ == 'Case2_FedDrift-Eager':
        return case_module.start_feddrift_simulation(client_datasets, DRIVE_BASE_PATH)
    elif case_module.__name__ == 'Case3_Centralized':
        return case_module.start_centralized_baseline(client_datasets, DRIVE_BASE_PATH)
    

if __name__ == "__main__":
    
    # 1. Check Data Path (Must be manually mounted in Colab before running)
    if not os.path.exists(DRIVE_BASE_PATH):
        print(f"FATAL ERROR: Data not found at {DRIVE_BASE_PATH}")
        print("Please ensure Google Drive is mounted and the DRIVE_BASE_PATH is correct.")
    else:
        # 2. Partition the data for all clients
        all_client_data = partition_data(DRIVE_BASE_PATH)

        # 3. Run Centralized Baseline (Case 3) - Get the upper bound
        centralized_results = run_flwr_simulation(all_client_data, Case3_Centralized)

        # 4. Run FedAvg Baseline (Case 1) - Get the lower bound (standard FL failure)
        fedavg_results = run_flwr_simulation(all_client_data, Case1_Fedavg)
        
        # 5. Run FedDrift-Eager (Case 2) - Test the multi-model solution
        feddrift_results = run_flwr_simulation(all_client_data, Case2_FedDrift-Eager)
        
        print("\nAll required simulations executed.")
        # Final step: Save all results for plotting/analysis
        # save_results(centralized_results, fedavg_results, feddrift_results)
