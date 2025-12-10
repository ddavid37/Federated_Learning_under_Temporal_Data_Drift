import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import random

# --- CONFIGURATION ---
IMAGE_SIZE = 64
NUM_CLASSES = 2
CAT_LABEL = 0
DOG_LABEL = 1

# --- 1. DATA UTILITIES ---

def get_transform():
    """Standard transformations for CNN training."""
    return transforms.Compose([
        # All images must be resized/cropped to a uniform size (64x64 for speed)
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), # Converts image to a PyTorch tensor (0-255 -> 0.0-1.0)
        # Use standard normalization for color images
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class ClientDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset to load images just-in-time from the mounted Drive path.
    This saves Colab memory and is highly efficient.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transform = get_transform()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        
        # Determine the label based on the folder name (Cat/Dog)
        parent_dir = os.path.basename(os.path.dirname(img_path))
        label = CAT_LABEL if parent_dir == 'Cat' else DOG_LABEL
        
        # Load, convert, and transform the image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception:
            # Handle corrupted files (common in PetImages): skip and return a random valid sample
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx) 

        return image, label

# --- 2. MODEL ARCHITECTURE ---
class SimpleCNN(nn.Module):
    """A lightweight CNN designed for efficient Colab training."""
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        # 64x64 input image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        # Output size calculation: ~64x64 -> 30x30 -> 13x13 -> 13x13 * 64 features
        self.fc1 = nn.Linear(64 * 13 * 13, 256) 
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model():
    return SimpleCNN(num_classes=NUM_CLASSES)
    