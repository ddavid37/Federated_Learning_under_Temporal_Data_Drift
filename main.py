"""
Main Runner Script for FL Under Concept Drift Project
Runs all experiments in sequence and generates the final report.

Usage: python main.py
"""

import subprocess
import sys
import os

def run_script(name, script):
    """Run a Python script and handle errors."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {name}")
    print(f"{'='*70}")
    
    result = subprocess.run([sys.executable, script], capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ {name} failed with return code {result.returncode}")
        return False
    return True

def main():
    print("="*70)
    print("FL UNDER CONCEPT DRIFT - COMPLETE EXPERIMENT PIPELINE")
    print("="*70)
    
    # Check if data exists
    if not os.path.exists('./data_seasonal/global_test_set.npz'):
        print("\n⚠ Data not found. Running Seasonal_Splitter.py first...")
        if not run_script("Data Preparation", "Seasonal_Splitter.py"):
            print("Failed to create data. Exiting.")
            return
    
    # Run experiments in order
    experiments = [
        ("Case 1: Centralized Baseline", "case1_centralized_baseline.py"),
        ("Case 2: FedAvg with Drift", "case2_fedavg.py"),
        ("Case 3: FedAvg + Replay", "case3_fedavg_replay.py"),
        ("Case 6: FedAvg IID (No Drift)", "case6_fedavg_iid.py"),
        ("Case 5: Ablation Study", "case5_replay_ablation.py"),
    ]
    
    for name, script in experiments:
        if not run_script(name, script):
            print(f"Stopping due to error in {name}")
            return
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    run_script("Visualizations", "visualize_advanced.py")
    
    # Generate report
    print("\n" + "="*70)
    print("GENERATING FINAL REPORT")
    print("="*70)
    run_script("PDF Report", "generate_report.py")
    
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - case1_metrics.json     (Centralized baseline)")
    print("  - case2_metrics.json     (FedAvg with drift)")
    print("  - case3_metrics.json     (FedAvg + Replay)")
    print("  - case5_ablation_metrics.json (Buffer size ablation)")
    print("  - case6_iid_metrics.json (FedAvg no drift)")
    print("  - heatmap_case2.png      (Forgetting visualization)")
    print("  - heatmap_case3.png      (Replay visualization)")
    print("  - recovery_curve.png     (Accuracy trajectory)")
    print("  - radar_robustness.png   (Per-class comparison)")
    print("  - ablation_buffer_size.png (Buffer size impact)")
    print("  - Final_Project_Report.pdf (Complete report)")

if __name__ == "__main__":
    main()
