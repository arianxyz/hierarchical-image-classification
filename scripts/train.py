import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

project_root = Path(".").resolve()
src_dir = project_root / "src"
results_dir = project_root / "results"

sys.path.append(str(src_dir))
results_dir.mkdir(exist_ok=True)

from data_loader import FashionDataset

CSV_PATH = os.path.join(project_root, "data/df_balanced.csv")

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(NeuralNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # Downsamples by 2

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional block 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Convolutional block 2
        x = self.pool(F.relu(self.conv2(x)))

        # Convolutional block 3
        x = self.pool(F.relu(self.conv3(x)))

        # Convolutional block 4
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        #print(f"Flattened shape: {x.shape}")
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits (no softmax here; use CrossEntropyLoss)

        return x


def main():    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = pd.read_csv(CSV_PATH)
    dataset = FashionDataset(device=device, df=df)

    BATCH_SIZE = 64
    EPOCHS = 30

    # ---------------- Weighted Sampling ----------------
    targets = dataset.encoded_labels
    class_sample_counts = np.bincount(targets)
    weights = 1.0 / class_sample_counts
    sample_weights = [weights[label] for label in targets]

    # ---------------- DataLoader ----------------
    # split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Extract corresponding sample weights for train subset
    train_indices = train_dataset.indices
    train_weights = [sample_weights[i] for i in train_indices]
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  

    # Initialize model, loss, optimizer
    model = NeuralNetwork(num_classes=len(dataset.index_to_classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        model.train()
        training_running_loss = 0.0
        correct, total = 0, 0

        for i, (images, labels) in enumerate(train_dataloader):            
            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_losses.append(training_running_loss / len(train_dataloader))
        train_accuracies.append(train_acc)
        print(f"[{epoch + 1}, {i + 1}] | Train Loss: {train_losses[-1]:.3f} | Accuracy: {train_acc:.2f}%")
        
        model.eval()
        test_running_loss = 0.0
        correct, total = 0, 0

        # Validation loop
        with torch.no_grad():
            for images, labels in val_dataloader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Validation loss after epoch {epoch + 1}: {test_running_loss / len(val_dataloader):.3f}")

        val_acc = 100 * correct / total
        val_losses.append(test_running_loss / len(val_dataloader))
        val_accuracies.append(val_acc)
        print(f"After epoch {epoch+1} Val Loss: {val_losses[-1]:.3f} | Accuracy: {val_acc:.2f}%")
            
    torch.save(model.state_dict(), results_dir / "model_weights.pth")
    print("Model weights saved to results/model_weights.pth")

    training_logs = {
        "train_loss": train_losses,
        "train_acc": train_accuracies,
        "val_loss": val_losses,
        "val_acc": val_accuracies
    }
    with open(results_dir / "training_logs.json", "w") as f:
        json.dump(training_logs, f, indent=4)

    print("Training logs saved to results/training_logs.json")        
    print("Finished Training")



if __name__ == "__main__":
    main()