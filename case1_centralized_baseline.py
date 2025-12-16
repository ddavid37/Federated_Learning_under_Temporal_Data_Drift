"""
Case 1: Centralized Training Baseline
Trains a CNN on centralized IID data to establish upper-bound accuracy.

Result: ~88% accuracy (reference ceiling for FL experiments)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from model import get_model

# --- Configuration ---
DATA_DIR = './data_seasonal'
INIT_PHASE = '0_init_iid'
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_phase_0_data():
    """Loads all client data from Phase 0 into one big centralized dataset."""
    print(f"Loading centralized data from {INIT_PHASE}...")
    
    all_data = []
    all_labels = []
    
    phase_dir = os.path.join(DATA_DIR, INIT_PHASE)
    
    for i in range(1, 11):
        path = os.path.join(phase_dir, f'client_{i}.npz')
        if os.path.exists(path):
            with np.load(path) as data:
                all_data.append(data['data'])
                all_labels.append(data['labels'])
    
    X = np.concatenate(all_data)
    y = np.concatenate(all_labels)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0 
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    print(f"   -> Centralized Dataset Size: {len(X_tensor)} samples")
    return TensorDataset(X_tensor, y_tensor)

def load_test_set():
    path = os.path.join(DATA_DIR, 'global_test_set.npz')
    with np.load(path) as data:
        X = data['data']
        y = data['labels']
    X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train():
    print(f"{'='*60}")
    print(f"Case 1: Centralized Training (Baseline) on {DEVICE}")
    print(f"{'='*60}")
    
    train_dataset = load_phase_0_data()
    test_dataset = load_test_set()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    model = get_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {"epochs": [], "accuracy": []}
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")
        
        metrics["epochs"].append(epoch + 1)
        metrics["accuracy"].append(acc)

    # Save model and metrics
    torch.save(model.state_dict(), "centralized_baseline.pth")
    with open("case1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RESULT: Centralized Baseline Accuracy = {acc:.2f}%")
    print(f"{'='*60}")
    print("✅ Model saved to centralized_baseline.pth")
    print("✅ Metrics saved to case1_metrics.json")

if __name__ == "__main__":
    train()
