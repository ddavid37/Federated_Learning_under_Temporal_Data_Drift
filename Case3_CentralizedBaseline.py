import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# Import components from model.py
from model import get_model 

# --- CONFIGURATION ---
NUM_EPOCHS = 10  # Typically more epochs than local FL epochs
BATCH_SIZE = 64  # Can be larger since all data is available
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- 1. Training and Evaluation Functions ---

def train_centralized(model, trainloader):
    """Train the model on the full, combined training set."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Optional: Print loss every epoch for monitoring
        # print(f"  Centralized Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss / len(trainloader):.4f}")


def test_centralized(model, testloader):
    """Evaluate the model on the full, combined validation set."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    loss /= len(testloader)
    accuracy = correct / total
    return loss, accuracy


# --- 2. Main Entry Point ---

def start_centralized_baseline(client_datasets, data_root_dir):
    """
    Combines all client data into single train/test sets and runs centralized training.
    """
    print("\n===== Starting Case 3: Centralized Baseline =====")
    
    # 1. Combine All Training Data and All Validation Data
    
    # Extract all individual client train and validation datasets
    all_trainsets = [train_ds for train_ds, _ in client_datasets.values()]
    all_valsets = [val_ds for _, val_ds in client_datasets.values()]
    
    # Concatenate them into single massive datasets
    full_trainset = ConcatDataset(all_trainsets)
    full_valset = ConcatDataset(all_valsets)
    
    # Create centralized data loaders
    trainloader = DataLoader(full_trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    testloader = DataLoader(full_valset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)
    
    print(f"Total Combined Training Samples: {len(full_trainset)}")
    print(f"Total Combined Validation Samples: {len(full_valset)}")
    
    # 2. Initialize and Train Model
    model = get_model()
    train_centralized(model, trainloader)
    
    # 3. Evaluate Final Performance
    final_loss, final_accuracy = test_centralized(model, testloader)
    
    print(f"\n[Case 3: Centralized Baseline Complete]")
    print(f"Final Centralized Accuracy: {final_accuracy:.4f}")
    
    return {"accuracy": final_accuracy, "loss": final_loss}
    