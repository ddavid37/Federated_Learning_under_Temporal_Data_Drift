import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = './data_seasonal'
PHASES = ["0_init_iid", "1_winter", "3_summer"] # We'll compare Init vs Winter vs Summer
CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def main():
    fig, axes = plt.subplots(len(PHASES), 10, figsize=(15, 6))
    plt.subplots_adjust(hspace=0.5)
    
    print("Generating visualization of Client 1's data across seasons...")

    for row, phase in enumerate(PHASES):
        # Load Client 1's data for this phase
        path = os.path.join(DATA_DIR, phase, 'client_1.npz')
        try:
            loaded = np.load(path)
            images = loaded['data']
            labels = loaded['labels']
        except FileNotFoundError:
            print(f"Skipping {phase} (File not found)")
            continue

        # Get 10 random samples
        indices = np.random.choice(len(images), 10, replace=False)
        
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            img = images[idx].reshape(28, 28)
            lbl = labels[idx]
            
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Label only the first column
            if col == 0:
                ax.set_ylabel(phase.upper(), fontsize=12, fontweight='bold')

            # Title only on first row
            if row == 0:
                ax.set_title(f"Sample {col+1}")
                
            # Add class name as small label
            ax.text(0.5, -0.2, CLASS_NAMES[lbl], transform=ax.transAxes, 
                    ha='center', fontsize=8)

    output_file = 'seasonal_drift_visualization.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_file}")
    [print(f"   Visualized phase: {p}") for p in PHASES if 'init' not in p]

if __name__ == "__main__":
    main()