"""
Case 2: Standard FedAvg Under Seasonal Drift
Demonstrates catastrophic forgetting when data distribution shifts.

Result: Drops from ~74% to ~28% accuracy after drift
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import copy
import json
from model import get_model

# --- Configuration ---
DATA_DIR = './data_seasonal'
SEASONS = ["0_init_iid", "1_winter", "2_spring", "3_summer", "4_fall"]
ROUNDS_PER_SEASON = 5
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_client_data(phase, client_id):
    path = os.path.join(DATA_DIR, phase, f'client_{client_id}.npz')
    with np.load(path) as data:
        X = torch.tensor(data['data'], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
        y = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(X, y)

def load_test_set():
    path = os.path.join(DATA_DIR, 'global_test_set.npz')
    with np.load(path) as data:
        X = torch.tensor(data['data'], dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
        y = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(X, y)

def client_update(model, train_loader):
    """Local training on client."""
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
    """FedAvg: Average weights from all clients."""
    avg_weights = copy.deepcopy(global_weights)
    for key in avg_weights.keys():
        avg_weights[key] = torch.stack([w[key] for w in local_weights_list]).mean(dim=0)
    return avg_weights

def evaluate_detailed(model, test_loader):
    """Evaluate model and return global + per-class accuracy."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Global accuracy
    matches = [p == l for p, l in zip(all_preds, all_labels)]
    acc = 100 * sum(matches) / len(matches)
    
    # Per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10
    for p, l in zip(all_preds, all_labels):
        if p == l:
            class_correct[l] += 1
        class_total[l] += 1
    
    per_class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                     for i in range(10)]
    
    return acc, per_class_acc

def main():
    print(f"{'='*60}")
    print(f"Case 2: Standard FedAvg (Simulation) on {DEVICE}")
    print(f"{'='*60}")
    
    global_model = get_model().to(DEVICE)
    test_set = load_test_set()
    test_loader = DataLoader(test_set, batch_size=1000)
    
    metrics_history = {
        "rounds": [],
        "accuracy": [],
        "per_class_accuracy": []
    }
    
    round_counter = 0
    for phase in SEASONS:
        print(f"\n=== Phase: {phase.upper()} ===")
        client_datasets = [load_client_data(phase, cid) for cid in range(1, 11)]
        
        for round_num in range(1, ROUNDS_PER_SEASON + 1):
            round_counter += 1
            local_weights = []
            
            for i in range(CLIENTS_PER_ROUND):
                local_model = get_model().to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                train_loader = DataLoader(client_datasets[i], batch_size=BATCH_SIZE, shuffle=True)
                w = client_update(local_model, train_loader)
                local_weights.append(w)
            
            new_weights = aggregate_weights(global_model.state_dict(), local_weights)
            global_model.load_state_dict(new_weights)
            
            acc, class_accs = evaluate_detailed(global_model, test_loader)
            print(f"   Round {round_num}/{ROUNDS_PER_SEASON} | Global Accuracy: {acc:.2f}%")
            
            metrics_history["rounds"].append(round_counter)
            metrics_history["accuracy"].append(acc)
            metrics_history["per_class_accuracy"].append(class_accs)

    # Save results
    with open("case2_metrics.json", "w") as f:
        json.dump(metrics_history, f)
    torch.save(global_model.state_dict(), "fedavg_seasonal.pth")
    
    print(f"\n{'='*60}")
    print(f"RESULT: Final Accuracy = {acc:.2f}% (dropped from ~74% due to forgetting)")
    print(f"{'='*60}")
    print("✅ Metrics saved to case2_metrics.json")
    print("✅ Model saved to fedavg_seasonal.pth")

if __name__ == "__main__":
    main()
