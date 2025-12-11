import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# Import components from model.py
from model import get_model, ClientDataset, IMAGE_SIZE 
from Case1_Fedavg import get_parameters, set_parameters, train, test, CatDogClient, DEVICE # Reuse common functions

# --- CONFIGURATION (Must match Case1 for fair comparison) ---
NUM_ROUNDS = 5 
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MIN_AVAILABLE_CLIENTS = 4

# --- 1. Custom Strategy for Multi-Model Aggregation ---

class FedDriftEagerStrategy(fl.server.strategy.Strategy):
    """
    Simplified Multi-Model Strategy for extreme non-IID conditions.
    Manages separate models for different concepts (Cat and Dog).
    """
    def __init__(self):
        super().__init__()
        # Dict to hold different models: {model_id: model_parameters}
        self.models: Dict[int, fl.common.Parameters] = {}
        # Dict to track which client belongs to which model (cluster)
        # cid is a string '0', '1', etc.
        self.client_cluster: Dict[str, int] = {}
        
        # Initialize two models (one for Cat-Clients, one for Dog-Clients)
        self._initialize_clusters()
        
    def _initialize_clusters(self):
        """Initializes two separate models (clusters) for the two concepts."""
        # Initial Model 1 (e.g., for Cat clients)
        self.models[1] = fl.common.ndarrays_to_parameters(get_parameters(get_model()))
        # Initial Model 2 (e.g., for Dog clients)
        self.models[2] = fl.common.ndarrays_to_parameters(get_parameters(get_model()))
        
        # Initial assignment based on label skew (C0, C1 = Cat; C2, C3 = Dog)
        self.client_cluster['0'] = 1  # Client 0 (Cat) -> Model 1
        self.client_cluster['1'] = 1  # Client 1 (Cat) -> Model 1
        self.client_cluster['2'] = 2  # Client 2 (Dog) -> Model 2
        self.client_cluster['3'] = 2  # Client 3 (Dog) -> Model 2
        print(f"Server initialized with {len(self.models)} models/clusters.")

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        """Returns the initial global parameters for the first round."""
        # Since we use an iterative approach (Model 1, Model 2, etc.), 
        # we return the initial parameters of Model 1 as the default.
        return self.models[1]

    def configure_fit(self, server_round: int, clients: List[fl.server.client_proxy.ClientProxy]) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Sends the correct model parameters to each client based on its cluster."""
        fit_configurations = []
        for client in clients:
            cid = client.cid
            # Get the model ID assigned to this client
            model_id = self.client_cluster.get(cid, 1) # Default to Model 1 if unassigned
            
            # Retrieve the specific parameters for this model/cluster
            parameters = self.models.get(model_id, self.models[1])
            
            # The client needs to know which model it is updating
            fit_ins = fl.common.FitIns(parameters, config={"model_id": model_id})
            fit_configurations.append((client, fit_ins))
            
        return fit_configurations

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregates updates only within the same model/cluster."""
        
        # 1. Group results by model ID (cluster)
        updates_by_model: Dict[int, List[Tuple[fl.common.Parameters, int]]] = defaultdict(list)
        
        for client, fit_res in results:
            # We assume the client returned the model ID in the metrics (fit_res.metrics)
            # which is critical for FedDrift-Eager.
            model_id = fit_res.metrics.get("model_id") 
            if model_id is not None:
                updates_by_model[model_id].append(
                    (fit_res.parameters, fit_res.num_examples)
                )

        # 2. Aggregate each model separately
        for model_id, updates in updates_by_model.items():
            if not updates:
                continue

            # Separate parameters and weights
            params_list = [fl.common.parameters_to_ndarrays(p) for p, _ in updates]
            num_examples_list = [n for _, n in updates]
            
            # Perform FedAvg aggregation
            aggregated_ndarrays = fl.common.aggregate(params_list, num_examples_list)
            
            # Update the global parameters for this specific model/cluster
            self.models[model_id] = fl.common.ndarrays_to_parameters(aggregated_ndarrays)

        # Since we are managing multiple models, we return None, as there is no single 'global' model
        # The true aggregated models are stored in self.models
        return None, {} 

    def configure_evaluate(self, server_round: int, clients: List[fl.server.client_proxy.ClientProxy]) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """Sends the current cluster model to the clients for evaluation."""
        eval_configurations = []
        for client in clients:
            cid = client.cid
            model_id = self.client_cluster.get(cid, 1)
            parameters = self.models.get(model_id, self.models[1])
            
            # Clients use the model assigned to their cluster to evaluate their local validation set
            eval_configurations.append((client, fl.common.EvaluateIns(parameters, config={})))
        
        return eval_configurations

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregates all client accuracies to get a global performance view."""
        
        total_examples = sum(res.num_examples for _, res in results)
        
        if total_examples == 0:
            return None, {}
            
        # Weighted average of accuracy across all clients (and thus, all models)
        weighted_accuracy = sum(res.num_examples * res.metrics.get("accuracy", 0) for _, res in results) / total_examples
        
        print(f"Round {server_round}: Aggregated Global Accuracy (All Models): {weighted_accuracy:.4f}")
        return weighted_accuracy, {"accuracy": weighted_accuracy}


# --- 2. Flower Client Class (Modified for Cluster ID) ---

class FedDriftEagerClient(fl.client.NumPyClient):
    # Reuses init, trainloader, valloader from Case1
    def __init__(self, client_id, trainset, valset):
        self.client_id = client_id
        self.model = get_model() 
        self.trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        self.valloader = DataLoader(valset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)
        self.model_id: int = 1 # Tracks which model/cluster the client is currently using

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Store the model ID for the aggregation step
        self.model_id = config.get("model_id", 1) 
        
        train(self.model, self.trainloader, epochs=LOCAL_EPOCHS)
        
        # Return parameters AND the model_id so the server knows which cluster to update
        return get_parameters(self.model), len(self.trainloader.dataset), {"model_id": self.model_id}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader)
        
        # Return local accuracy and loss for global tracking
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "local_loss": loss}


# --- 3. Simulation Entry Point ---

def client_fn(cid: str, client_datasets):
    """Factory function to create a client."""
    client_id = int(cid)
    # Flower's Client IDs start from 0, but our data dict keys start from 1.
    trainset, valset = client_datasets[client_id + 1] 
    return FedDriftEagerClient(client_id, trainset, valset).to_client()

def start_feddrift_simulation(client_datasets, data_root_dir):
    """Main function to start the FedDrift-Eager FL simulation."""

    # Define the custom strategy
    strategy = FedDriftEagerStrategy()

    # Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, client_datasets),
        num_clients=len(client_datasets),
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 1} if DEVICE.type == 'cuda' else None,
    )
    
    # Extract final accuracy for plotting/reporting
    accuracies = [
        (r, acc) 
        for r, metrics in history.metrics_distributed.items() 
        for acc in metrics.values()
    ]
    
    print(f"\n[Case 2: FedDrift-Eager Complete]")
    print(f"Final Global Accuracy: {accuracies[-1][1]:.4f}")
    
    return {"history": history, "accuracies": accuracies}


# --- 4. Wrapper for main.py ---

def start_fl_process(client_datasets, data_root_dir):
    """Wrapper function to align with main.py structure."""
    return start_feddrift_simulation(client_datasets, data_root_dir)