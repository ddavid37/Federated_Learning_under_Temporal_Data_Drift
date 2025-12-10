import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict
import numpy as np

# Import components from model.py
from model import get_model, ClientDataset, IMAGE_SIZE 

# --- CONFIGURATION ---
NUM_ROUNDS = 5  # Keep rounds low for fast debugging
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- 1. Client Helper Functions ---

def set_parameters(model: nn.Module, parameters):
    """Sets the PyTorch model weights from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_parameters(model: nn.Module):
    """Returns the PyTorch model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def train(model, trainloader, epochs):
    """Train the model on the local client data."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(model, testloader):
    """Evaluate the model on the local client validation data."""
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


# --- 2. Flower Client Class ---

class CatDogClient(fl.client.NumPyClient):
    def __init__(self, client_id, trainset, valset):
        self.client_id = client_id
        self.model = get_model()  # Initialize local model
        # Use more efficient data loaders (num_workers > 0 for Colab GPU speedup)
        self.trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        self.valloader = DataLoader(valset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=LOCAL_EPOCHS)
        # Return parameters and number of examples used for aggregation
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader)
        
        # Return loss and accuracy. The local_loss is key for FedDrift-Eager later.
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "local_loss": loss}


# --- 3. Client Creation and Simulation Entry Point ---

def client_fn(cid: str, client_datasets):
    """Factory function to create a client."""
    client_id = int(cid)
    # Flower's Client IDs start from 0, but our data dict keys start from 1.
    trainset, valset = client_datasets[client_id + 1] 
    return CatDogClient(client_id, trainset, valset).to_client()

def aggregate_eval_metrics(metrics):
    """Server-side function to aggregate evaluation results."""
    # Simple weighted average of accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, m in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def start_fedavg_simulation(client_datasets, data_root_dir):
    """Main function to start the FedAvg FL simulation."""

    # Define the FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all 4 clients every round
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
        # Initialize the global model with weights from a local model
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(get_model())),
    )

    # Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, client_datasets),
        num_clients=len(client_datasets),
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    
    # Extract final accuracy for plotting/reporting
    accuracies = [
        (r, acc) 
        for r, metrics in history.metrics_distributed_global.items() 
        for acc in metrics.values()
    ]
    
    print(f"\n[Case 1: FedAvg Baseline Complete]")
    print(f"Final Global Accuracy: {accuracies[-1][1]:.4f}")
    
    return {"history": history, "accuracies": accuracies}
    