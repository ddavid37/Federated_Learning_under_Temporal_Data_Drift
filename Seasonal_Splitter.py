"""
Seasonal Data Splitter for Fashion-MNIST
Creates non-IID data distributions simulating seasonal concept drift.

Drift Schedule:
- Phase 0 (Init): IID data from all 10 classes
- Phase 1 (Winter): Coat, Ankle boot, Pullover
- Phase 2 (Spring): Trouser, Shirt, Bag
- Phase 3 (Summer): T-shirt, Dress, Sandal  
- Phase 4 (Fall): Sneaker, Pullover, Shirt, Trouser
"""

import torch
from torchvision import datasets
import numpy as np
import os

# --- Configuration ---
NUM_CLIENTS = 10
DATA_DIR = './data_seasonal'
NUM_TRAIN_SAMPLES = 60000

# Class mapping: 
# 0: T-shirt, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat
# 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
SEASONS = {
    "1_winter": [4, 9, 2],      # Coat, Ankle boot, Pullover
    "2_spring": [1, 6, 8],      # Trouser, Shirt, Bag
    "3_summer": [0, 3, 5],      # T-shirt, Dress, Sandal
    "4_fall":   [7, 2, 6, 1]    # Sneaker, Pullover, Shirt, Trouser
}

def split_and_save():
    print("1. Downloading Fashion-MNIST...")
    train_dataset = datasets.FashionMNIST(root='./data_raw', train=True, download=True)
    data = train_dataset.data.numpy().reshape(-1, 784)
    labels = train_dataset.targets.numpy()

    # 1. Init Phase (IID - Balanced)
    print("2. Creating Phase 0: Initialization (IID)...")
    init_idx = np.random.permutation(NUM_TRAIN_SAMPLES)[:10000]
    rem_idx = np.setdiff1d(np.arange(NUM_TRAIN_SAMPLES), init_idx)
    save_phase(data[init_idx], labels[init_idx], "0_init_iid")

    # 2. Seasonal Phases (Drift)
    print("3. Creating Seasonal Drift Phases...")
    rem_data = data[rem_idx]
    rem_labels = labels[rem_idx]

    for season, class_ids in SEASONS.items():
        mask = np.isin(rem_labels, class_ids)
        save_phase(rem_data[mask], rem_labels[mask], season)

def save_phase(data, labels, phase_name):
    phase_dir = os.path.join(DATA_DIR, phase_name)
    if not os.path.exists(phase_dir): 
        os.makedirs(phase_dir)
    
    chunk = len(data) // NUM_CLIENTS
    for i in range(NUM_CLIENTS):
        start = i * chunk
        end = start + chunk
        path = os.path.join(phase_dir, f'client_{i+1}.npz')
        np.savez_compressed(path, data=data[start:end], labels=labels[start:end])
    print(f"   -> Saved {phase_name}: {len(data)} samples ({chunk} per client)")

if __name__ == "__main__":
    split_and_save()
    
    # Save Global Test Set
    print("4. Creating Global Test Set...")
    test_ds = datasets.FashionMNIST(root='./data_raw', train=False, download=True)
    t_data = test_ds.data.numpy().reshape(-1, 784)
    t_labels = test_ds.targets.numpy()
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    np.savez_compressed(os.path.join(DATA_DIR, 'global_test_set.npz'), data=t_data, labels=t_labels)
    print(f"   -> Saved global_test_set.npz: {len(t_data)} samples")
    
    print("\n✅ Seasonal Data Created Successfully!")
