import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim 
from torch.utils.data import DataLoader

from generic_utils.config import (
    DEFAULT_DEVICE,
    initialize_data_config,
    set_seed,
    ModelConfig,
    StageConfig,
    )

from modeling.baseline1.model import ResnetClassifier
from modeling.baseline8.model import ResnetLSTMClassifier
from data_utils.data_cls import VolleyballTemporalDataset
from data_utils.functions import get_scheduler, player_lvl_transforms
import generic_utils.checkpoint as cp
from generic_utils.log import Logger, log_curves
from evaluation.validation import validate


def train(config: StageConfig, logger: Logger, data_loaders: tuple):
    logger.info('Training Logs for Baseline8 - Stage1\n')
    logger.info(f'Device used: {device}')
    
    train_loader, val_loader = data_loaders
    
    model = ResnetLSTMClassifier(
        config.model,
        backbone=ResnetClassifier(num_cls=4), # 4 is an arbitrary number.
        imgs_per_clip=train_loader.dataset.imgs_per_clip,
        extraction_mode=False
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
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
    next_check = len(train_loader) // 2
    
    for epoch in tqdm(range(1, config.training['epochs'] + 1), desc='Training..'):
        logger.info(f'Epoch ({epoch} / {config.training["epochs"]}) ::')
        logger.info(f'Current lr = {optimizer.param_groups[0]["lr"]}\n')

        correct = total_loss = total_samples = 0 

        model.train()
        for i, (data, labels) in enumerate(train_loader, 1):
            data, labels = data.to(device), labels.to(device)

            logits = model(data)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            output_idx = torch.argmax(logits, dim=1)
            correct += torch.sum(labels == output_idx).item()
            total_samples += labels.size(0)

            if i == next_check:
                logger.info(f'Batch ({i} / {len(train_loader)}): \
Average batch loss = {loss.item() / labels.size(0):.4f} | running batch accuracy = {correct / total_samples * 100:.2f}%')
        
        loss = total_loss / total_samples
        acc = correct / total_samples * 100

        val_loss, val_acc, val_f1_macro, _ = validate(model, val_loader, device)

        train_loss_curve.append(loss)
        train_acc_curve.append(acc)
        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_acc)

        logger.info(f'End of epoch {epoch}: train loss: {loss:.4f}  -  train accuracy: {acc:.2f}%')
        logger.info(f'End of epoch {epoch}: val loss: {val_loss:.4f}  -\
  val accuracy: {val_acc:.2f}%  -  val macro f1 score: {val_f1_macro:.4f}\n')
        
        scheduler.step(val_loss)

    cp.save_checkpoint(model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=config.training['epochs'],
                    val_acc=val_acc, config_path=config.config_path,
                    output_path=config.out_dir, cp_name='st1_last_checkpoint')
    logger.info('Last model checkpoint saved')

    curves = (train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve)
    names = ('train loss', 'validation loss', 
            'train accuracy', 'validation accuracy')
    txt = log_curves(curves, names, config.log_dir, file_name='st1')
    logger.info(txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=str(DEFAULT_DEVICE), 
                    help='Force a device (e.g., cpu, cuda:0)')
    args = parser.parse_args()
    device = torch.device(args.device)

    set_seed()
    b8_cfg = ModelConfig('baseline8').get_stage('stage1')

    t_transform, v_transform = player_lvl_transforms(b8_cfg.data['resize_dims'])
    b8_cfg.data.pop('data_cls')
    b8_cfg.data.pop('resize_dims')
    
    train_set = VolleyballTemporalDataset(
        data_cfg=initialize_data_config(),
        subset_dir_name='train_set',
        transform=t_transform,
        **b8_cfg.data
    )
    val_set = VolleyballTemporalDataset(
        data_cfg=initialize_data_config(),
        subset_dir_name='validation_set',
        transform=v_transform,
        **b8_cfg.data
    )
    def collate_fn(batch):
        data, labels = tuple(zip(*batch))
        _, p_idx = tuple(zip(*labels))
        
        data = torch.cat(data, dim=0)
        c, h, w = data[0].size()[-3:]
        data = data.view((-1, c, h, w))

        p_idx = torch.cat(p_idx)

        return data, p_idx
        
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=b8_cfg.training['train_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=b8_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    logger = Logger(b8_cfg.log_dir, 'baseline8_st1')
    train(b8_cfg, logger, (train_loader, val_loader))
