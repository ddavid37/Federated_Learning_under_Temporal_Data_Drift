import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import copy
from model import get_model

# --- Configuration ---
DATA_DIR = './data_seasonal'
# The order of seasons simulates the "Temporal Drift"
SEASONS = ["0_init_iid", "1_winter", "2_spring", "3_summer", "4_fall"]
ROUNDS_PER_SEASON = 5  # How many communication rounds to spend on each season
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_client_data(phase, client_id):
    path = os.path.join(DATA_DIR, phase, f'client_{client_id}.npz')
    with np.load(path) as data:
        X = torch.tensor(data['data'], dtype=torch.float32).reshape(-1, 1, 28, 28)
        y = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(X, y)

def load_test_set():
    path = os.path.join(DATA_DIR, 'global_test_set.npz')
    with np.load(path) as data:
        X = torch.tensor(data['data'], dtype=torch.float32).reshape(-1, 1, 28, 28)
        y = torch.tensor(data['labels'], dtype=torch.long)
    return TensorDataset(X, y)

def client_update(model, train_loader):
    """Trains the model locally on the client's data."""
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
    """FedAvg: Averages the weights from all clients."""
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

def main():
    print(f"--- Case 2: Standard FedAvg (Simulation) on {DEVICE} ---")
    
    # 1. Initialize Global Model
    global_model = get_model().to(DEVICE)
    test_set = load_test_set()
    test_loader = DataLoader(test_set, batch_size=1000)
    
    # 2. Simulation Loop over Seasons
    for phase in SEASONS:
        print(f"\n=== Entering Phase: {phase.upper()} ===")
        
        # In this simulation, the data "arrives" at the clients at the start of the season
        # We pre-load it here for efficiency
        client_datasets = [load_client_data(phase, cid) for cid in range(1, 11)]
        
        for round_num in range(1, ROUNDS_PER_SEASON + 1):
            local_weights = []
            
            # --- Client Training Step ---
            for i in range(CLIENTS_PER_ROUND):
                # Copy global model to local client
                local_model = get_model().to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                
                # Client trains on their seasonal data
                train_loader = DataLoader(client_datasets[i], batch_size=BATCH_SIZE, shuffle=True)
                w = client_update(local_model, train_loader)
                local_weights.append(w)
            
            # --- Server Aggregation Step ---
            new_weights = aggregate_weights(global_model.state_dict(), local_weights)
            global_model.load_state_dict(new_weights)
            
            # --- Evaluation ---
            acc = evaluate(global_model, test_loader)
            print(f"   Round {round_num}/{ROUNDS_PER_SEASON} | Global Test Accuracy: {acc:.2f}%")

    print("\n✅ FedAvg Simulation Complete.")
    torch.save(global_model.state_dict(), "fedavg_seasonal.pth")

if __name__ == "__main__":
    main()