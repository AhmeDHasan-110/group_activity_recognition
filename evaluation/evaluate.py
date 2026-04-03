import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    output_path: str, 
    target_names: list, 
    baseline_name: str, 
    device: torch.device=None
    ):
    """
    Evaluate a trained model on a test dataloader and save the results.

    Args:
        model: torch.nn.Module
        data_loader: DataLoader for the test set
        output_path (str): Directory where results will be saved
        target_names (list): List of class names
        baseline_name (str): Name of the baseline for reporting
        device (torch.device, optional): The device to use for evaluation. Defaults to None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print("NOTE: The given output_path couldn't be found so it was created automatically.")

    eval_dir_path = os.path.join(output_path, 'evaluation_results')
    os.makedirs(eval_dir_path, exist_ok=True)

    model = model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    total_loss = 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            pred = model(data)

            loss = criterion(pred, labels)
            total_loss += loss.item()

            pred_idx = torch.argmax(pred, dim=1)

            all_preds.append(pred_idx.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    loss, accuracy, cr, cm = calc_metrics(total_loss, all_labels, all_preds, target_names)
    save_results(eval_dir_path, target_names, baseline_name, loss, accuracy, cr, cm)


def calc_metrics(total_loss, targets, preds, target_names):
    """
    Calculate metrics for the given predictions and targets.

    Args:
        total_loss: Total loss
        targets: Ground truth labels
        preds: Predicted labels
        target_names: List of class names

    Returns:
        loss: Loss
        accuracy: Accuracy
        cr: Classification report
        cm: Confusion matrix
    """
    loss = total_loss / len(targets)
    accuracy = accuracy_score(targets, preds)

    cr = classification_report(targets, preds, target_names=target_names, digits=4)
    cm = confusion_matrix(targets, preds, normalize='true')

    return round(loss, 4), 100 * round(accuracy, 4), cr, cm


def save_results(eval_dir_path, target_names, baseline_name, *res):
    """
    Save the evaluation results to a file.

    Args:
        eval_dir_path: Directory where results will be saved
        target_names: List of class names
        baseline_name: Name of the baseline for reporting
        res: Results to save(coming from calc_metrics)
    """
    loss, acc, cr, cm = res

    cm_array = np.array(cm)
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm_array, 
        annot=True, 
        fmt='.2g',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        linewidths=0.01,
        linecolor='gray'
    )
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for {baseline_name}')
    plt.tight_layout()

    plt.savefig(
        os.path.join(eval_dir_path, "confusion_matrix.png"), 
        dpi=300, 
        bbox_inches="tight"
    )
    
    plt.close()

    res_f = os.path.join(eval_dir_path, 'metrics.txt')

    with open(res_f, 'w') as f:
        f.write(f"Evaluation results on the test set for baseline: {baseline_name}\n\n")
        f.write("Accuracy: " + str(acc) + "%" + "\n")
        f.write("Loss: " + str(loss) + "\n\n")
        f.write("Classification Report:\n")
        f.write(cr)
    
    print('Evaluation metrics saved successfully.')