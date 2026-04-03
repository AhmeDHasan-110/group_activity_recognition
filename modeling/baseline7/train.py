import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn 
from torch import optim 
from torch.utils.data import DataLoader 
from sklearn.metrics import f1_score

from generic_utils.config import (
    DEFAULT_DEVICE,
    SAVED_EMBEDDINGS_DIR,
    set_seed,
    ModelConfig,
    StageConfig,
    )

from modeling.baseline7.model import MultiHeadClassifier
from data_utils.feature_extraction import extract_features
from data_utils.data_cls import EmbeddingsDataset
from generic_utils.log import Logger, log_curves
from data_utils.functions import get_scheduler
import generic_utils.checkpoint as cp


def train(config: StageConfig, logger: Logger, data_loaders: tuple):
    logger.info('Training Logs for Baseline7\n')
    logger.info(f'Device used: {device}')
    
    model = MultiHeadClassifier(config.model).to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    logger.info(f'Criterion used: {criterion}')

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training['lr'],
        weight_decay=config.training['weight_decay']
    )
    
    scheduler = get_scheduler(config, optimizer)

    train_loader, val_loader = data_loaders
    logger.info(f'Train data size = {len(train_loader.dataset)} ; \
Validation data size = {len(val_loader.dataset)}')
    
    train_loss_curve, val_loss_curve = [], []
    train_acc_curve, val_acc_curve = [], []
    next_check = len(train_loader) // 2 # report metrics once in an epoch

    alpha = config.training['alpha']
    for epoch in tqdm(range(1, config.training['epochs'] + 1), desc='Training..'):
        logger.info(f'Epoch ({epoch} / {config.training["epochs"]}) ::')
        logger.info(f'Current lr = {optimizer.param_groups[0]["lr"]}\n')

        model.train()
        total_loss, correct, total_samples = 0, 0, 0

        for i, (data, g_idx, p_idx, sizes) in enumerate(train_loader, 1):
            data, g_idx, p_idx = data.to(device), g_idx.to(device), p_idx.to(device)
            
            p_logits, g_logits = model(data, sizes)

            p_loss = criterion(p_logits, p_idx)
            g_loss = criterion(g_logits, g_idx)
            
            loss = g_loss + (alpha * p_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # calc group metrics only
            total_loss += g_loss.item()
            output_idx = torch.argmax(g_logits, dim=1)
            correct += torch.sum(g_idx == output_idx).item()
            total_samples += g_idx.size(0)

            if i == next_check:
                logger.info(
                    f'Batch ({i} / {len(train_loader)}):\
Average batch loss of group head = {g_loss.item() / g_idx.size(0):.4f}  |\
  running batch accuracy of group head = {correct / total_samples * 100:.2f}%'
    )
        
        loss = total_loss / total_samples
        acc = correct / total_samples * 100

        val_loss, val_acc, val_f1_macro = validate(model, val_loader)

        train_loss_curve.append(loss)
        train_acc_curve.append(acc)
        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_acc)

        logger.info(f'End of epoch {epoch}: train loss: {loss:.4f}  -  train accuracy: {acc:.2f}%')
        logger.info(f'End of epoch {epoch}: val loss: {val_loss:.4f}  -  \
val accuracy: {val_acc:.2f}%  -  val macro f1 score: {val_f1_macro:.4f}\n')
        
        scheduler.step()

    cp.save_checkpoint(model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=config.training['epochs'],
                    val_acc=val_acc, config_path=config.config_path,
                    output_path=config.out_dir, cp_name='last_checkpoint')
    logger.info('Last model checkpoint saved')

    curves = (train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve)
    names = ('group head train loss', 'group head validation loss', 
            'group head train accuracy', 'group head validation accuracy')
    txt = log_curves(curves, names, config.log_dir)
    logger.info(txt)

def validate(model, val_loader):
    criterion = nn.CrossEntropyLoss(reduction='sum')

    total_loss, correct = 0, 0
    all_preds, all_labels = [], []

    model.eval()
    with torch.inference_mode():
        for data, g_idx, _, sizes in val_loader:
            data, g_idx = data.to(device), g_idx.to(device)

            p_logits, g_logits = model(data, sizes)
            g_loss = criterion(g_logits, g_idx)

            total_loss += g_loss.item()
            pred_idx = torch.argmax(g_logits, dim=1)
            correct += torch.sum(g_idx == pred_idx).item()

            all_preds.append(pred_idx.cpu())
            all_labels.append(g_idx.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        loss = total_loss / len(all_labels)
        acc = correct / len(all_labels) * 100
        
        f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    return loss, acc, f1_macro


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=str(DEFAULT_DEVICE), 
                    help='Force a device (e.g., cpu, cuda:0)')
    parser.add_argument(
        '-e', '--extract',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable or disable the feature extraction process (default: disabled).'
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    set_seed()
    b7_cfg = ModelConfig('baseline7').get_stage('stage1')

    embds_path = os.path.join(SAVED_EMBEDDINGS_DIR, 'b7_embds')
    if args.extract and os.path.exists(embds_path): # if already extracted once before
        train_data = torch.load(os.path.join(embds_path, 'train_set.pth'))
        val_data = torch.load(os.path.join(embds_path, 'validation_set.pth'))
    else:
        b3_st2_cfg = ModelConfig('baseline3').get_stage('stage2')

        train_data = extract_features(
            extractor_cfg=b3_st2_cfg,
            extraction_info=b7_cfg.extraction_info,
            device=device,
            subset_name='train_set'
        )
        val_data = extract_features(
            extractor_cfg=b3_st2_cfg,
            extraction_info=b7_cfg.extraction_info,
            device=device,
            subset_name='validation_set'
        )
    
    train_set = EmbeddingsDataset(data=train_data)
    val_set = EmbeddingsDataset(data=val_data)

    def collate_fn(batch):
        data, labels = tuple(zip(*batch))
        g_idx, p_idx = tuple(zip(*labels))
        
        sizes = [item.size(0) for item in data]
        data = torch.cat(data)
        p_idx = torch.cat(p_idx)

        return data, torch.tensor(g_idx), p_idx, sizes

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=b7_cfg.training['train_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=b7_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    logger = Logger(b7_cfg.log_dir, 'baseline7')
    train(b7_cfg, logger, (train_loader, val_loader))
