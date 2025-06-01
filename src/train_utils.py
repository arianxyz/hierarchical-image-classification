import torch
import numpy as np

def train_model(
    model, 
    optimizer, 
    criterion, 
    X_train, 
    y_train, 
    device, 
    epochs, 
    batch_size,
    X_val=None, 
    y_val=None, 
    print_every=1
):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size].to(device)
            y_batch = y_train[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        
        acc = correct / total * 100
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(acc)

        # Validation
        if X_val is not None and y_val is not None:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    x_batch = X_val[i:i+batch_size].to(device)
                    y_batch = y_val[i:i+batch_size].to(device)
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)
            val_acc = val_correct / val_total * 100
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if (epoch+1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.3f} | Acc: {acc:.2f}% | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%")
        else:
            if (epoch+1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.3f} | Accuracy: {acc:.2f}%")
    return history

def evaluate_model(model, X, y, device, batch_size=32):
    """
    Evaluates the model and returns predicted and true labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size].to(device)
            y_batch = y[i:i+batch_size].to(device)
            outputs = model(x_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)