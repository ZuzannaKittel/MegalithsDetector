import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from pathlib import Path
import time

# Import your custom dataset
from dataset import MegalithDataset

def get_adapted_resnet():
    """Loads a pre-trained ResNet18 and adapts it for 1-channel LRM data."""
    # Load standard pre-trained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 1. Adapt the first convolutional layer for 1 channel (instead of 3 RGB channels)
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Preserve pre-trained weights by averaging the RGB weights into the new single channel
    with torch.no_grad():
        model.conv1.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
        
    # 2. Adapt the final classification head for Binary Output (Megalith vs. Background)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def train_model():
    # Setup MPS (Metal Performance Shaders) for Apple Silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device.type.upper()}")

    # Paths
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data" / "dataset" / "train"
    
    # Load Dataset
    full_dataset = MegalithDataset(data_dir, is_train=True)
    
    # 80/20 Train/Validation Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Batch size of 16 is safe for 16GB Unified Memory with 512x512 images
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Initialize Model, Loss, and Optimizer
    model = get_adapted_resnet().to(device)
    
    # BCEWithLogitsLoss is perfect for binary classification 
    # (combines Sigmoid and BCELoss for numerical stability)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    epochs = 10
    best_val_loss = float('inf')
    
    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1) # Shape to (batch, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()
            
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()
                
        # --- METRICS ---
        train_loss /= train_size
        train_acc = train_correct / train_size
        val_loss /= val_size
        val_acc = val_correct / val_size
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
              
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), project_dir / "best_megalith_model.pth")
            
    print("\nTraining Complete! Best model saved to 'best_megalith_model.pth'")

if __name__ == "__main__":
    train_model()