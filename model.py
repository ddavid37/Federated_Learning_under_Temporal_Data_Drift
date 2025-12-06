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
    """
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        
        # Convolutional Block 1
        # Padding=2 ensures the output size remains 28x28 before pooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        
        # Fully Connected Layers
        # Image reduction logic:
        # Input: 28x28
        # After Pool 1: 14x14
        # After Pool 2: 7x7
        # Final Flattened Vector: 64 channels * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10) # 10 Output Classes

    def forward(self, x):
        # x shape: [Batch_Size, 1, 28, 28]
        
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten for Dense Layers
        # -1 automatically infers the batch size
        x = x.view(-1, 64 * 7 * 7)
        
        # Dense Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # Note: We return raw logits. PyTorch's CrossEntropyLoss includes Softmax.
        return x

def get_model():
    """Helper function to instantiate the model."""
    return FashionMNIST_CNN()

if __name__ == '__main__':
    # Test the model dimensions when running this script directly
    model = get_model()
    print("Model Architecture:")
    print(model)
    
    # Create a dummy input tensor (Batch Size 1, 1 Channel, 28x28 Image)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    try:
        output = model(dummy_input)
        print(f"\nTest Pass Successful.")
        print(f"Input Shape: {dummy_input.shape}")
        print(f"Output Shape: {output.shape} (Expected: [1, 10])")
    except Exception as e:
        print(f"\nTest Pass Failed: {e}")