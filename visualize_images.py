import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
data_dir = os.path.join(os.path.dirname(__file__), 'data')
train_csv_path = os.path.join(data_dir, 'fashion_mnist_train.csv')

# Read the CSV
df = pd.read_csv(train_csv_path)

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display 10 random images (one from each class)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

# Get one image from each class
for class_idx in range(10):
    # Find an image with this label
    class_images = df[df['label'] == class_idx]
    if len(class_images) > 0:
        # Get the first image of this class
        image_data = class_images.iloc[0, :-1].values  # All columns except label
        image = image_data.reshape(28, 28)  # Reshape to 28x28
        
        # Plot
        axes[class_idx].imshow(image, cmap='gray')
        axes[class_idx].set_title(f'{class_names[class_idx]}')
        axes[class_idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'fashion_mnist_samples.png'), dpi=150, bbox_inches='tight')
print(f"Sample images saved to {os.path.join(data_dir, 'fashion_mnist_samples.png')}")
plt.show()

# Display a grid of 25 random images
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.flatten()

np.random.seed(42)
random_indices = np.random.choice(len(df), 25, replace=False)

for idx, random_idx in enumerate(random_indices):
    image_data = df.iloc[random_idx, :-1].values
    label = df.iloc[random_idx, -1]
    image = image_data.reshape(28, 28)
    
    axes[idx].imshow(image, cmap='gray')
    axes[idx].set_title(f'{class_names[int(label)]}', fontsize=8)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'fashion_mnist_grid.png'), dpi=150, bbox_inches='tight')
print(f"Grid of images saved to {os.path.join(data_dir, 'fashion_mnist_grid.png')}")
plt.show()
