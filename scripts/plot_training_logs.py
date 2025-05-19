import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load logs
results_dir = Path("results")
log_path = results_dir / "training_logs.json"
plots_dir = results_dir / "plots"
plots_dir.mkdir(exist_ok=True)

with open(log_path, "r") as f:
    logs = json.load(f)

train_loss = logs["train_loss"]
val_loss = logs["val_loss"]
train_acc = logs["train_acc"]
val_acc = logs["val_acc"]
epochs = range(1, len(train_loss) + 1)

# --- 1. Loss: Train vs Validation ---
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Train Loss", marker="o")
plt.plot(epochs, val_loss, label="Val Loss", marker="o", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "loss_comparison.png")

# --- 2. Accuracy: Train vs Validation ---
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
plt.plot(epochs, val_acc, label="Val Accuracy", marker="o", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "accuracy_comparison.png")

print(f"Plots saved to {plots_dir.resolve()}")