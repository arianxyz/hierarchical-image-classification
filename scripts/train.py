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

src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from dataloader import FashionDataset


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

    project_root = Path("..").resolve()
    CSV_PATH = os.path.join(project_root, "project/data/labels.csv")
    df = pd.read_csv(CSV_PATH)

    BATCH_SIZE = 128
    EPOCHS = 30

    dataset = FashionDataset(device=device, df=df)
    # split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # DataLoader for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)    

    # Initialize model, loss, optimizer
    model = NeuralNetwork(num_classes=len(dataset.index_to_classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        training_running_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader):            
            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_running_loss += loss.item()
            
        print(f"[{epoch + 1}, {i + 1}] loss: {training_running_loss / 100:.3f}")
        
        # Validation loop
        test_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dataloader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
            print(f"Validation loss after epoch {epoch + 1}: {test_running_loss / len(val_dataloader):.3f}")
            
    print("Finished Training")



if __name__ == "__main__":
    main()