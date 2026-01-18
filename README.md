# Federated Learning Under Extreme Label Skew

This project investigates and successfully mitigates the severe performance degradation experienced by Federated Learning (FL) models when operating in environments characterized by **extreme, persistent label skew** across clients.

The methodology involves implementing a multi-model solution, FedDrift-Eager, and comparing its stability and convergence against standard FedAvg and an upper-bound Centralized baseline.

## 💾 Imported Data & Extreme Partitioning

The simulation uses the Kaggle Cats vs. Dogs PetImages dataset. The data is accessed from the local machine's file system for speed and partitioned to create the necessary conditions for testing the FedDrift-Eager solution.

### Data Overview

| Directory | Original Count | Cleaned Count | Label |
| :--- | :--- | :--- | :--- |
| `data/PetImages/Cat` | 12,490 | 12,490 | 0 (Cat) |
| `data/PetImages/Dog` | 12,469 | 12,469 | 1 (Dog) |
| **Total Images** | $\mathbf{24,959}$ | $\mathbf{24,959}$ | |

### Extreme Label Skew Design

The 25,000 images are partitioned into four non-overlapping client shards to create the necessary *concept drift* where no single global model can satisfy all clients:

| Client ID | Data Content | Samples (Train/Val Split) | Challenge |
| :--- | :--- | :--- | :--- |
| **Client 1** | **$100\%$ Cat Images** | $\approx 4,996$ Training Samples | Requires a Cat-specialized model. |
| **Client 2** | **$100\%$ Cat Images** | $\approx 4,996$ Training Samples | Requires a Cat-specialized model. |
| **Client 3** | **$100\%$ Dog Images** | $\approx 4,987$ Training Samples | Requires a Dog-specialized model. |
| **Client 4** | **$100\%$ Dog Images** | $\approx 4,987$ Training Samples | Requires a Dog-specialized model. |

## 🚀 Current Project Scope & Benchmarks

| Feature | Scope Implemented |
| :--- | :--- |
| **Data Challenge** | **Extreme Label Skew (Data Heterogeneity)** |
| **Execution** | Local Machine (VS Code/PowerShell) for maximum speed and stability. |

### The Three Benchmark Cases

We executed three comparative cases to validate the necessity and effectiveness of the multi-model approach:

1.  **Case 1: FedAvg Baseline (Lower Bound)**
    * **Purpose:** Demonstrates the failure mode of standard FedAvg when forced to aggregate conflicting, non-IID models.
2.  **Case 2: FedDrift-Eager (Multi-Model Solution)**
    * **Goal:** The novel implementation using a **custom Flower strategy** to maintain two separate, specialized models (one for Cat data, one for Dog data) to prevent convergence to a single, generalized poor   state.
3.  **Case 3: Centralized Baseline (Upper Bound)**
    * **Purpose:** Establishes the theoretical maximum achievable accuracy by training a single model on all data centrally.


## 📂 Project Structure

* **`main.py`**: Orchestrates the entire simulation, handles data integrity checks, performs the **Extreme Label Skew partition**, and launches the three benchmark cases sequentially.
* **`model.py`**: Defines the lightweight `SimpleCNN` and the robust `ClientDataset` with corrupted file handling.
* **`Case1_Fedavg.py`**: Implements the standard FedAvg strategy with corrected metric aggregation.
* **`Case2_FedDriftEager.py`**: Implements the custom multi-model aggregation strategy for the FedDrift-Eager solution.
* **`Case3_CentralizedBaseline.py`**: Contains the local PyTorch training loop used to establish the upper bound and monitor epoch-by-epoch loss.
