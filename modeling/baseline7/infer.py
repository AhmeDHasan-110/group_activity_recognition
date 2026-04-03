import os
import argparse
import torch
from torch.utils.data import DataLoader

from generic_utils.config import ModelConfig, initialize_data_config, DEFAULT_DEVICE
from evaluation.evaluate import calc_metrics, save_results
from data_utils.feature_extraction import extract_features
from data_utils.data_cls import EmbeddingsDataset
import generic_utils.checkpoint as cp
from modeling.baseline7.model import MultiHeadClassifier

def evaluate(model, data_loader, output_path, target_names, baseline_name, device):
    """Evaluate the model on the test set."""

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
        for data, g_idx, sizes in data_loader:
            data, g_idx = data.to(device), g_idx.to(device)

            p_logits, g_logits = model(data, sizes)
            g_loss = criterion(g_logits, g_idx)

            total_loss += g_loss.item()
            pred_idx = torch.argmax(g_logits, dim=1)
            
            all_preds.append(pred_idx.cpu())
            all_labels.append(g_idx.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    loss, accuracy, cr, cm = calc_metrics(total_loss, all_labels, all_preds, target_names)
    save_results(eval_dir_path, target_names, baseline_name, loss, accuracy, cr, cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=str(DEFAULT_DEVICE),
                        help='Force a device (e.g., cpu, cuda:0)')
    args = parser.parse_args()
    device = torch.device(args.device)

    b3_cfg = ModelConfig('baseline3').get_stage('stage1')
    b7_cfg = ModelConfig('baseline7').get_stage('stage1')

    test_data = extract_features(
        extractor_cfg=b3_cfg,
        extraction_info=b7_cfg.extraction_info,
        device=device,
        subset_name='test_set',
    )
    test_set = EmbeddingsDataset(data=test_data)
    
    def collate_fn(batch):
        data, labels = tuple(zip(*batch))
        g_idx, p_idx = tuple(zip(*labels))
        
        sizes = [item.size(0) for item in data]
        
        data = torch.cat(data)
        p_idx = torch.cat(p_idx)

        return data, torch.tensor(g_idx), sizes
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=b7_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = MultiHeadClassifier(b7_cfg.model) 

    cp_path = os.path.join(b7_cfg.out_dir, 'checkpoints', 'last_checkpoint.pth')
    trained_model, *_ = cp.load_checkpoint(
        cp_path=cp_path,
        model=model,
        device=device
    )

    data_cfg = initialize_data_config()
    target_names = sorted(data_cfg.group_labels.keys(), key = lambda x: data_cfg.group_labels[x])

    kwargs = {
        'model' : trained_model,
        'data_loader' : test_loader,
        'output_path' : b7_cfg.out_dir,
        'target_names' : target_names,
        'baseline_name' : 'baseline7',
        'device' : device
    }
    evaluate(**kwargs)