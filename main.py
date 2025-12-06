import os
import sys
import time
import subprocess

# --- Configuration ---
# Ensure these filenames match exactly what you saved
FILES = {
    "splitter": "Seasonal_Splitter.py",
    "case1": "case1_centralized_baseline.py",
    "case2": "case2_fedavg.py",
    "model": "model.py"
}

def run_script(script_name, description):
    """Runs a python script and handles errors."""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {description}")
    print(f"   Running {script_name}...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run the script using the current python interpreter
    result = subprocess.run([sys.executable, script_name])
    
    end_time = time.time()
    duration = end_time - start_time
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed with exit code {result.returncode}.")
        print("   Please fix the error in that file before continuing.")
        sys.exit(1)
    else:
        print(f"\n✅ FINISHED: {description}")
        print(f"   Time taken: {duration:.2f} seconds")

def check_files():
    """Checks if all necessary files exist."""
    missing = []
    for key, filename in FILES.items():
        if not os.path.exists(filename):
            missing.append(filename)
    
    if missing:
        print("❌ CRITICAL ERROR: Missing files!")
        print("   The following files are missing from this directory:")
        for m in missing:
            print(f"   - {m}")
        print("\n   Please create them before running main.py.")
        sys.exit(1)

def main():
    print("\nStarting NNDL Project: Federated Learning under Temporal Drift")
    print("Mentors: Saivignesh Venkatraman")
    print("Team: Daniel David, Sahasra Kokkula, Aaditya Bhaskar Baruah")
    
    # 1. Verify Files
    check_files()
    
    # 2. Data Preparation
    # Check if data exists, if not, run the splitter
    if not os.path.exists("./data_seasonal"):
        print("\n[!] Seasonal data not found. Generating now...")
        run_script(FILES["splitter"], "Data Generation (Seasonal Drift)")
    else:
        print("\n[i] Seasonal data found. Skipping generation.")
        # Optional: Uncomment line below to force regeneration
        # run_script(FILES["splitter"], "Data Generation (Seasonal Drift)")

    # 3. Case 1: Centralized Baseline
    # This establishes the "Gold Standard" accuracy
    run_script(FILES["case1"], "Case 1: Centralized Baseline Training")

    # 4. Case 2: Federated Averaging (The "Struggle")
    # This demonstrates how standard FL fails when seasons change
    run_script(FILES["case2"], "Case 2: Federated Averaging (Standard)")

    print(f"\n{'='*60}")
    print("🎉 EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print("\nNext Steps for your Report:")
    print("1. Compare the Final Accuracy of Case 1 vs. Case 2.")
    print("2. Look at the Case 2 logs: Did accuracy drop when the season changed?")
    print("   (e.g., transition from INIT -> WINTER or WINTER -> SPRING)")
    print("3. If accuracy dropped significantly, you have successfully demonstrated")
    print("   that standard FL struggles with temporal drift!")

if __name__ == "__main__":
    main()