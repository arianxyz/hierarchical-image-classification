import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import numpy as np
import sys


project_root = Path(".").resolve()
src_dir = project_root / "src"
results_dir = project_root / "results"

weights_path = results_dir / "model_weights.pth"
plots_dir = results_dir / "plots"
plots_dir.mkdir(exist_ok=True)

project_root = Path(".").resolve()
csv_path = project_root / "data/df_balanced.csv"

sys.path.append(str(src_dir))
results_dir.mkdir(exist_ok=True)

from data_loader import FashionDataset
from train import NeuralNetwork

# Load CSV and dataset
df = pd.read_csv(csv_path)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dataset = FashionDataset(device=device, df=df)

# Load model
model = NeuralNetwork(num_classes=len(dataset.index_to_classes))
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# Prepare validation split
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# DataLoader
from torch.utils.data import DataLoader
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Predict
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
labels = list(dataset.index_to_classes.values())

# Plot using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(plots_dir / "confusion_matrix.png")
print(f"Confusion matrix saved to {plots_dir / 'confusion_matrix.png'}")