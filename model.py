"""
Model Architecture for Fashion-MNIST Classification
CNN with 2 convolutional layers and 2 fully connected layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionMNIST_CNN(nn.Module):
    """
    Standard CNN for Fashion-MNIST.
    Input: 28x28 grayscale image.
    Output: 10 classes (clothing types).
    
    Architecture:
    1. Conv2d (1 -> 32 filters, 5x5 kernel) + ReLU + MaxPool (2x2)
    2. Conv2d (32 -> 64 filters, 5x5 kernel) + ReLU + MaxPool (2x2)
    3. Flatten
    4. Linear (3136 -> 512) + ReLU
    5. Linear (512 -> 10)
    
    Total Parameters: ~1.7M
    """
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        
        # Fully Connected Layers
        # Image: 28x28 -> Pool1: 14x14 -> Pool2: 7x7 -> Flatten: 64*7*7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten for Dense Layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

def get_model():
    """Helper function to instantiate the model."""
    return FashionMNIST_CNN()

if __name__ == '__main__':
    model = get_model()
    print("Model Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
