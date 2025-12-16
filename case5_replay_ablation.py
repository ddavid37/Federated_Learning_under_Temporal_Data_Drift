"""
Case 5: Replay Buffer Size Ablation Study
Tests buffer sizes: 0, 10, 25, 50, 100 samples per class
Demonstrates impact of buffer capacity on forgetting mitigation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
import os
import copy
import json
import matplotlib.pyplot as plt
from model import get_model

# --- Configuration ---
DATA_DIR = './data_seasonal'
SEASONS = ["0_init_iid", "1_winter", "2_spring", "3_summer", "4_fall"]
ROUNDS_PER_SEASON = 5
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ablation buffer sizes
BUFFER_SIZES = [0, 10, 25, 50, 100]

def load_client_data(phase, client_id):
    path = os.path.join(DATA_DIR, phase, f'client_{client_id}.npz')
    with np.load(path) as data:
        X = torch.tensor(data['data'], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
        y = torch.tensor(data['labels'], dtype=torch.long)
    return X, y

def load_test_set():
    path = os.path.join(DATA_DIR, 'global_test_set.npz')
    with np.load(path) as data:
        X = torch.tensor(data['data'], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
        y = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(X, y)

class ReplayBuffer:
    def __init__(self, capacity_per_class):
        self.capacity = capacity_per_class
        self.data = {}

    def add(self, X, y):
        if self.capacity == 0:
            return
        
        unique_labels = torch.unique(y)
        for label in unique_labels:
            lbl = label.item()
            if lbl not in self.data:
                self.data[lbl] = []
            
            indices = (y == label).nonzero(as_tuple=True)[0]
            x_subset = X[indices]
            
            current_len = len(self.data[lbl])
            needed = self.capacity - current_len
            
            if needed > 0:
                select = x_subset[:needed]
                self.data[lbl].extend([t.clone() for t in select])

    def get_dataset(self):
        if not self.data or self.capacity == 0:
            return None
        
        all_x = []
        all_y = []
        
        for lbl, samples in self.data.items():
            if len(samples) > 0:
                all_x.extend(samples)
                all_y.extend([lbl] * len(samples))
        
        if not all_x:
            return None

        X = torch.stack(all_x)
        y = torch.tensor(all_y, dtype=torch.long)
        return TensorDataset(X, y)

def client_update(model, train_loader):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(LOCAL_EPOCHS):
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def aggregate_weights(global_weights, local_weights_list):
    avg_weights = copy.deepcopy(global_weights)
    for key in avg_weights.keys():
        avg_weights[key] = torch.stack([w[key] for w in local_weights_list]).mean(dim=0)
    return avg_weights

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_ablation(buffer_size):
    """Run FedAvg+Replay with a specific buffer size."""
    print(f"\n{'='*60}")
    print(f"Running: Buffer Size = {buffer_size} samples/class")
    print(f"{'='*60}")
    
    global_model = get_model().to(DEVICE)
    test_set = load_test_set()
    test_loader = DataLoader(test_set, batch_size=1000)
    
    accuracy_history = []
    client_buffers = [ReplayBuffer(buffer_size) for _ in range(11)]
    
    round_counter = 0
    for phase in SEASONS:
        print(f"   Phase: {phase}")
        client_data_raw = [load_client_data(phase, cid) for cid in range(1, 11)]
        
        for round_num in range(1, ROUNDS_PER_SEASON + 1):
            round_counter += 1
            local_weights = []
            
            for i in range(CLIENTS_PER_ROUND):
                client_id = i + 1
                
                if round_num == 1:
                    X_new, y_new = client_data_raw[i]
                    client_buffers[client_id].add(X_new, y_new)
                
                X_curr, y_curr = client_data_raw[i]
                current_ds = TensorDataset(X_curr, y_curr)
                buffer_ds = client_buffers[client_id].get_dataset()
                
                if buffer_ds:
                    train_ds = ConcatDataset([current_ds, buffer_ds])
                else:
                    train_ds = current_ds
                
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
                
                local_model = get_model().to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                w = client_update(local_model, train_loader)
                local_weights.append(w)
            
            new_weights = aggregate_weights(global_model.state_dict(), local_weights)
            global_model.load_state_dict(new_weights)
            
            acc = evaluate(global_model, test_loader)
            accuracy_history.append(acc)
    
    print(f"   Final Accuracy: {accuracy_history[-1]:.2f}%")
    return accuracy_history

def plot_ablation_results(results):
    """Create ablation visualization."""
    plt.figure(figsize=(12, 6))
    
    rounds = list(range(1, 26))
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    
    for i, (buf_size, accs) in enumerate(results.items()):
        label = f"Buffer={buf_size}/class" if buf_size > 0 else "No Replay"
        plt.plot(rounds, accs, label=label, color=colors[i], linewidth=2, marker='o', markersize=4)
    
    # Add phase boundaries
    for x in [5.5, 10.5, 15.5, 20.5]:
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    
    # Add phase labels
    phases = ['Init (IID)', 'Winter', 'Spring', 'Summer', 'Fall']
    for i, phase in enumerate(phases):
        plt.text(i*5 + 2.5, 95, phase, ha='center', fontsize=9, alpha=0.7)
    
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Global Test Accuracy (%)', fontsize=12)
    plt.title('Ablation Study: Impact of Replay Buffer Size', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.ylim(20, 100)
    plt.tight_layout()
    plt.savefig('ablation_buffer_size.png', dpi=150)
    plt.close()
    print("\n✅ Saved: ablation_buffer_size.png")

def main():
    print(f"{'='*60}")
    print(f"Case 5: Replay Buffer Ablation Study on {DEVICE}")
    print(f"Testing buffer sizes: {BUFFER_SIZES}")
    print(f"{'='*60}")
    
    results = {}
    final_accuracies = {}
    
    for buf_size in BUFFER_SIZES:
        acc_history = run_ablation(buf_size)
        results[buf_size] = acc_history
        final_accuracies[buf_size] = acc_history[-1]
    
    # Save metrics
    with open("case5_ablation_metrics.json", "w") as f:
        json.dump({
            "buffer_sizes": BUFFER_SIZES,
            "results": {str(k): v for k, v in results.items()},
            "final_accuracies": {str(k): v for k, v in final_accuracies.items()}
        }, f, indent=2)
    
    # Generate plot
    plot_ablation_results(results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Buffer Size':<20} {'Final Accuracy':<15} {'Memory/Client':<15}")
    print("-"*50)
    for buf_size in BUFFER_SIZES:
        memory_kb = buf_size * 10 * 784 / 1024
        print(f"{buf_size} samples/class{'':<5} {final_accuracies[buf_size]:.2f}%{'':<8} {memory_kb:.0f} KB")
    
    print("\n✅ Metrics saved to case5_ablation_metrics.json")
    print("✅ Ablation Study Complete!")

if __name__ == "__main__":
    main()
