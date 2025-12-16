"""
Visualization Script for FL Under Concept Drift
Generates: heatmaps, recovery curves, radar charts, ablation plots
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import pi
import os

# --- Configuration ---
CLASS_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_metrics(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠ File not found: {filename}")
        return None

def plot_heatmap(metrics_data, title, filename):
    """Generate per-class accuracy heatmap."""
    per_class_acc = metrics_data.get("per_class_accuracy", [])
    if not per_class_acc: 
        return
    
    per_class_acc = np.array(per_class_acc).T
    
    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(per_class_acc, cmap="RdYlGn", vmin=0, vmax=100,
                     yticklabels=CLASS_LABELS, cbar_kws={'label': 'Accuracy (%)'})
    
    # Add phase boundaries
    for x in [5, 10, 15, 20]:
        plt.axvline(x=x, color='black', linewidth=2, alpha=0.7)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Fashion Classes")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")

def plot_radar(case2, case3, case6=None):
    """Generate radar chart for per-class comparison."""
    if not case2 or not case3: 
        return

    c2_final = case2["per_class_accuracy"][-1]
    c3_final = case3["per_class_accuracy"][-1]
    c1_final = [88.0] * 10  # Baseline approximation
    c6_final = case6["per_class_accuracy"][-1] if case6 else None
    
    N = len(CLASS_LABELS)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    c2_vals = c2_final + c2_final[:1]
    c3_vals = c3_final + c3_final[:1]
    c1_vals = c1_final + c1_final[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], CLASS_LABELS, size=10)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
    plt.ylim(0, 100)
    
    ax.plot(angles, c1_vals, linewidth=1, linestyle='dashed', color='blue', label='Centralized Baseline')
    ax.plot(angles, c2_vals, linewidth=2, linestyle='dotted', color='red', label='FedAvg (with drift)')
    ax.plot(angles, c3_vals, linewidth=2, linestyle='solid', color='green', label='FedAvg + Replay')
    
    if c6_final:
        c6_vals = c6_final + c6_final[:1]
        ax.plot(angles, c6_vals, linewidth=2, linestyle='--', color='orange', label='FedAvg (no drift)')

    plt.title("Per-Class Robustness Profile (Final Round)", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=9)
    plt.tight_layout()
    plt.savefig("radar_robustness.png", dpi=150)
    plt.close()
    print("✅ Saved: radar_robustness.png")

def plot_recovery(case2, case3, case6=None):
    """Generate accuracy trajectory plot."""
    plt.figure(figsize=(14, 6))
    
    r2 = case2["rounds"]
    a2 = case2["accuracy"]
    r3 = case3["rounds"]
    a3 = case3["accuracy"]
    
    window = 3
    a2_smooth = pd.Series(a2).rolling(window=window, min_periods=1).mean()
    a3_smooth = pd.Series(a3).rolling(window=window, min_periods=1).mean()
    
    plt.plot(r2, a2, alpha=0.3, color='red')
    plt.plot(r2, a2_smooth, color='red', linewidth=2, label='FedAvg (with drift)')
    
    plt.plot(r3, a3, alpha=0.3, color='green')
    plt.plot(r3, a3_smooth, color='green', linewidth=2, label='FedAvg + Replay')
    
    if case6:
        r6 = case6["rounds"]
        a6 = case6["accuracy"]
        a6_smooth = pd.Series(a6).rolling(window=window, min_periods=1).mean()
        plt.plot(r6, a6, alpha=0.3, color='orange')
        plt.plot(r6, a6_smooth, color='orange', linewidth=2, linestyle='--', label='FedAvg (no drift, IID)')
    
    plt.axhline(y=88.0, color='blue', linestyle=':', linewidth=2, label='Centralized Baseline (88%)')
    
    # Add phase boundaries
    for x in [5.5, 10.5, 15.5, 20.5]:
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    
    # Add phase labels
    phases = ['Init (IID)', 'Winter', 'Spring', 'Summer', 'Fall']
    for i, phase in enumerate(phases):
        plt.text(i*5 + 2.5, 92, phase, ha='center', fontsize=10, fontweight='bold', alpha=0.7)
    
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Global Test Accuracy (%)', fontsize=12)
    plt.title('Accuracy Trajectory Under Concept Drift', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(20, 95)
    plt.xlim(0.5, 25.5)
    plt.tight_layout()
    plt.savefig("recovery_curve.png", dpi=150)
    plt.close()
    print("✅ Saved: recovery_curve.png")

def plot_ablation():
    """Plot ablation study results."""
    ablation_file = "case5_ablation_metrics.json"
    if not os.path.exists(ablation_file):
        print(f"⚠ Ablation file not found: {ablation_file}")
        return
    
    with open(ablation_file, "r") as f:
        data = json.load(f)
    
    results = data["results"]
    
    plt.figure(figsize=(14, 6))
    
    rounds = list(range(1, 26))
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    
    for i, (buf_size, accs) in enumerate(results.items()):
        label = f"Buffer={buf_size}/class" if int(buf_size) > 0 else "No Replay (Buffer=0)"
        plt.plot(rounds, accs, label=label, color=colors[i], linewidth=2, marker='o', markersize=4)
    
    # Add phase boundaries
    for x in [5.5, 10.5, 15.5, 20.5]:
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    
    # Add phase labels
    phases = ['Init (IID)', 'Winter', 'Spring', 'Summer', 'Fall']
    for i, phase in enumerate(phases):
        plt.text(i*5 + 2.5, 95, phase, ha='center', fontsize=10, fontweight='bold', alpha=0.7)
    
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Global Test Accuracy (%)', fontsize=12)
    plt.title('Ablation Study: Impact of Replay Buffer Size', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(20, 100)
    plt.xlim(0.5, 25.5)
    plt.tight_layout()
    plt.savefig("ablation_buffer_size.png", dpi=150)
    plt.close()
    print("✅ Saved: ablation_buffer_size.png")

def main():
    print("="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # Load all metrics
    c2 = load_metrics("case2_metrics.json")
    c3 = load_metrics("case3_metrics.json")
    c6 = load_metrics("case6_iid_metrics.json")
    
    # Generate plots
    if c2: 
        plot_heatmap(c2, "Standard FedAvg: Catastrophic Forgetting Under Drift", "heatmap_case2.png")
    if c3: 
        plot_heatmap(c3, "FedAvg + Experience Replay: Forgetting Mitigated", "heatmap_case3.png")
    
    if c2 and c3:
        plot_radar(c2, c3, c6)
        plot_recovery(c2, c3, c6)
    
    plot_ablation()
    
    print("\n✅ All visualizations generated!")

if __name__ == "__main__":
    main()
