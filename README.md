# Federated Learning under Temporal Data Drift

This repository implements **FedAvg with Experience Replay** to mitigate catastrophic forgetting in federated learning systems experiencing temporal data drift.

## 📋 Overview

In real-world federated learning deployments, client data distributions shift over time (e.g., seasonal trends in retail, evolving user behavior). Standard FedAvg suffers from **catastrophic forgetting** when faced with such drift—new knowledge overwrites old, causing accuracy collapse.

Our solution: **Client-side Experience Replay Buffers** that retain representative samples from past distributions, enabling continual learning without centralized data storage.

## 🔬 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | Fashion-MNIST (60k train / 10k test) |
| Model | CNN (~1.7M parameters) |
| Clients | 10 (full participation) |
| Local Epochs | 1 |
| Batch Size | 32 |
| Optimizer | SGD (lr=0.01, momentum=0.9) |
| Rounds | 25 (5 phases × 5 rounds) |
| Replay Buffer | 50 samples/class (default) |

### Seasonal Drift Simulation

We simulate temporal drift using Fashion-MNIST's 10 classes across 5 seasonal phases:

| Phase | Classes Available |
|-------|-------------------|
| Init (IID) | All 10 classes |
| Winter | Coat, Ankle Boot, Pullover |
| Spring | Trouser, Shirt, Bag |
| Summer | T-shirt, Dress, Sandal |
| Fall | Sneaker, Pullover, Shirt, Trouser |

## 📊 Results

| Experiment | Final Accuracy | Description |
|------------|----------------|-------------|
| Case 1: Centralized | **87.41%** | Upper bound (full data access) |
| Case 2: FedAvg + Drift | **35.90%** | Catastrophic forgetting (from 74%) |
| Case 3: FedAvg + Replay | **78.23%** | Forgetting mitigated (+42 pts) |
| Case 6: FedAvg IID | **84.73%** | No drift baseline |

### Ablation Study (Buffer Size)

| Buffer (samples/class) | Final Accuracy |
|------------------------|----------------|
| 0 | 35.32% |
| 10 | 60.88% |
| 25 | 68.65% |
| 50 | 77.58% |
| 100 | 83.33% |

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib reportlab
```

### Run All Experiments

```bash
python main.py
```

### Run Individual Experiments

```bash
# 1. Prepare seasonal data splits
python Seasonal_Splitter.py

# 2. Centralized baseline
python case1_centralized_baseline.py

# 3. FedAvg under drift (shows catastrophic forgetting)
python case2_fedavg.py

# 4. FedAvg with replay buffer (mitigates forgetting)
python case3_fedavg_replay.py

# 5. FedAvg without drift (IID baseline)
python case6_fedavg_iid.py

# 6. Buffer size ablation study
python case5_replay_ablation.py

# 7. Generate visualizations
python visualize_advanced.py

# 8. Generate PDF report
python generate_report.py
```

## 📁 File Structure

```
├── model.py                    # CNN architecture
├── Seasonal_Splitter.py        # Data preparation with drift simulation
├── case1_centralized_baseline.py   # Centralized training (upper bound)
├── case2_fedavg.py             # Standard FedAvg under drift
├── case3_fedavg_replay.py      # FedAvg with experience replay
├── case5_replay_ablation.py    # Buffer size ablation study
├── case6_fedavg_iid.py         # FedAvg without drift (IID)
├── visualize_advanced.py       # Generate all plots
├── generate_report.py          # PDF report generator
├── main.py                     # Run all experiments
├── *.json                      # Experiment metrics
├── *.png                       # Visualization outputs
├── *.pth                       # Trained model weights
└── Final_Project_Report.pdf    # Complete project report
```

## 📈 Generated Visualizations

- `heatmap_case2.png` - Per-class accuracy without replay (shows forgetting)
- `heatmap_case3.png` - Per-class accuracy with replay (stable)
- `recovery_curve.png` - Accuracy over rounds comparison
- `radar_robustness.png` - Multi-metric comparison
- `ablation_buffer_size.png` - Buffer size vs accuracy

## 🔧 Model Architecture

```
CNN(
  Conv2d(1, 32, kernel_size=5) → ReLU → MaxPool2d(2)
  Conv2d(32, 64, kernel_size=5) → ReLU → MaxPool2d(2)
  Flatten → Linear(3136, 512) → ReLU
  Linear(512, 10)
)
```

## 📚 References

1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017
2. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," PNAS 2017
3. Shoham et al., "Overcoming Forgetting in Federated Learning on Non-IID Data," NeurIPS Workshop 2019
4. Yoon et al., "Federated Continual Learning with Weighted Inter-client Transfer," ICML 2021

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 👥 Authors

Neural Networks and Deep Learning Project - December 2025
