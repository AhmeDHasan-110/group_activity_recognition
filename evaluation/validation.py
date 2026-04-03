import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np

def validate(model, val_loader, device, criterion=None):
    """
    Validate the model on the validation set after every epoch in training.

    Args:
        model: The model to validate
        val_loader: DataLoader for the validation set
        device: Device to use for validation
        criterion: Loss function

    Returns:
        loss: Validation loss
        acc: Validation accuracy
        f1_macro: Macro F1 score
        f1_per_class: Per-class F1 scores
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction='sum')

    total_loss, correct = 0, 0
    all_preds, all_labels = [], []

    model.eval()
    with torch.inference_mode():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            pred = model(data)
            loss = criterion(pred, labels)

            total_loss += loss.item()
            pred_idx = torch.argmax(pred, dim=1)
            correct += torch.sum(labels == pred_idx).item()

            all_preds.append(pred_idx.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        loss = total_loss / len(all_labels)
        acc = correct / len(all_labels) * 100

        f1_per_class = f1_score(all_labels, all_preds, average=None)
        f1_macro = f1_score(all_labels, all_preds, average='macro')

    return loss, acc, f1_macro, f1_per_class
