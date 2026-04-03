import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import optim
import numpy as np
import argparse
from sklearn.metrics import f1_score
from tqdm import tqdm

from generic_utils.focal_loss import FocalLoss
from generic_utils.config import (
    initialize_data_config,
    ModelConfig,
    StageConfig,
    DEFAULT_DEVICE,
    set_seed,
    )
from generic_utils import checkpoint as cp 
from generic_utils.log import Logger, log_curves
from data_utils.functions import player_lvl_transforms
from data_utils.functions import get_scheduler
from data_utils.data_cls import VolleyballPlayerDataset
from modeling.baseline1.model import ResnetClassifier

def train(logger: Logger, config: StageConfig, data_loaders: tuple):
    logger.info('Training Logs for Baseline3 - Stage1\n')
    logger.info(f'Device used: {device}')

    train_loader, val_loader = data_loaders
    cls_names = sorted(train_loader.dataset.mapping.keys(), key=lambda x: train_loader.dataset.mapping[x])

    model = ResnetClassifier(
        num_cls=config.model['num_classes'], 
        freeze_backbone=False
    ).to(device)

    logger.info(f'The starting lr = {config.training["lr"]}')

    criterion = FocalLoss(
        gamma=2.0,
        alpha=None,
        reduction='sum'
    )
    criterion = criterion.to(device)
    logger.info(f'Criterion used: {criterion}')
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training['lr'],
        weight_decay=config.training['weight_decay']
    )
    
    scheduler = get_scheduler(config, optimizer)

    logger.info(f'Train data size = {len(train_loader.dataset)} ; \
    Validation data size = {len(val_loader.dataset)}')
    
    train_loss_curve, val_loss_curve = [], []
    train_acc_curve, val_acc_curve = [], []

    logger.info('Training loop logs:\n')

    for epoch in tqdm(range(1, config.training['epochs'] + 1), desc='Training'):
        logger.info(f'Epoch ({epoch} / {config.training["epochs"]}) ::')
        logger.info(f'Current lr = {optimizer.param_groups[0]["lr"]}\n')

        model.train()
        total_loss, correct, total_samples = 0, 0, 0
        offset = next_check = len(train_loader) // 2

        for i, (data, labels) in enumerate(train_loader, 1):
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            output_idx = torch.argmax(output, dim=1)
            correct += torch.sum(labels == output_idx).item()
            total_samples += labels.size(0)

            if i == next_check:
                logger.info(
                    f'Batch ({i} / {len(train_loader)}):\
    Average batch loss = {loss.item() / labels.size(0):.4f}  |\
    running batch accuracy = {correct / total_samples * 100:.2f}%'
    )
                next_check += offset

        loss = total_loss / total_samples
        acc = correct / total_samples * 100

        val_loss, val_acc, val_f1_macro, val_f1_weighted, val_f1_all = validate(model, val_loader)

        train_loss_curve.append(loss)
        train_acc_curve.append(acc)
        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_acc)

        logger.info(f'End of epoch {epoch}: train loss: {loss:.4f}  -  train accuracy: {acc:.2f}%')
        logger.info(f'\t\t val loss: {val_loss:.4f}  -  val accuracy: {val_acc:.2f}%')
        logger.info(f'\t\t val macro f1 score: {val_f1_macro:.4f}  -  val weighted f1 score: {val_f1_weighted:.4f}')
        
        print(f'class f1 scores: {dict(zip(cls_names, val_f1_all))}\n')

        scheduler.step(val_loss)
    
    # only save the last model
    cp.save_checkpoint(model=model, optimizer=optimizer,
                        scheduler=scheduler, epoch=config.training['epochs'],
                        val_acc=val_acc, config_path=config.config_path,
                        output_path=config.out_dir, cp_name='st1_last_checkpoint')
    logger.info('Last model checkpoint saved')
    
    curves = (train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve)
    names = ('train loss', 'validation loss', 'train accuracy', 'validation accuracy')
    txt = log_curves(curves, names, config.log_dir, 'stage1')
    logger.info(txt)

def validate(model, val_loader):
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

            all_labels.append(labels.cpu())
            all_preds.append(pred_idx.cpu())
        
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        loss = total_loss / len(all_labels)
        acc = correct / len(all_labels) * 100

        f1_per_class = f1_score(all_labels, all_preds, average=None)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    return loss, acc, f1_macro, f1_weighted, np.round(f1_per_class, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=str(DEFAULT_DEVICE), 
                    help='Force a device (e.g., cpu, cuda:0)')
    args = parser.parse_args()
    device = torch.device(args.device)

    set_seed()
    
    model_cfg = ModelConfig('baseline3')
    stage_cfg = model_cfg.get_stage('stage1')

    train_trans, val_trans = player_lvl_transforms(stage_cfg.data['resize_dims'])

    data_cfg = initialize_data_config()
    train_ds = VolleyballPlayerDataset(
        data_cfg=data_cfg,
        subset_dir_name='train_set',
        transform=train_trans,
        undersample_maj=stage_cfg.data['undersample_maj'],
        undersample_precent=stage_cfg.data['undersample_precent']
    )

    data_cfg = initialize_data_config()
    val_ds = VolleyballPlayerDataset(
        data_cfg=data_cfg,
        subset_dir_name='validation_set',
        transform=val_trans,
        undersample_maj=False,
    )

    wr_sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=stage_cfg.training['train_batch_size'],
        shuffle=False if stage_cfg.data['use_wrs'] else True,
        sampler=wr_sampler if stage_cfg.data['use_wrs'] else None,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=stage_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger = Logger(stage_cfg.log_dir, 'baseline3_st1')
    train(logger, stage_cfg, (train_loader, val_loader))
