import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim 
from torch.utils.data import DataLoader

from generic_utils.config import (
    DEFAULT_DEVICE,
    SAVED_EMBEDDINGS_DIR,
    set_seed,
    ModelConfig,
    StageConfig,
    )
from generic_utils.log import Logger, log_curves
from generic_utils import checkpoint as cp
from data_utils.feature_extraction import extract_features
from data_utils.data_cls import EmbeddingsDataset
from data_utils.functions import get_scheduler
from evaluation.validation import validate
from modeling.baseline4.model import TemporalClassifier


def train(config: StageConfig, logger: Logger, data_loaders: tuple):
    logger.info('Training Logs for Baseline4\n')
    logger.info(f'Device used: {device}')

    model = TemporalClassifier(**config.model) 
    model = model.to(device)

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

    for epoch in tqdm(range(1, config.training['epochs'] + 1), desc='Training..'):
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

        val_loss, val_acc, val_f1_macro, val_f1_all = validate(model, val_loader, device)

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
                    output_path=config.out_dir, cp_name='last_checkpoint')
    logger.info('Last model checkpoint saved')

    curves = (train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve)
    names = ('train loss', 'validation loss', 'train accuracy', 'validation accuracy')
    txt = log_curves(curves, names, config.log_dir)
    logger.info(txt)


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
    
    b4_cfg = ModelConfig('baseline4').get_stage('stage1')
    
    path = os.path.join(SAVED_EMBEDDINGS_DIR, 'b4_embds')
    if args.extract and os.path.exists(path): # if already extracted once before
        train_data = torch.load(os.path.join(path, 'train_set.pth'))
        val_data = torch.load(os.path.join(path, 'validation_set.pth'))
    else:
        b1_cfg = ModelConfig('baseline1').get_stage('stage1') # the extractor model

        train_data = extract_features(
            extractor_cfg=b1_cfg,
            extraction_info=b4_cfg.extraction_info,
            device=device,
            subset_name='train_set',
        )
        val_data = extract_features(
            extractor_cfg=b1_cfg,
            extraction_info=b4_cfg.extraction_info,
            device=device,
            subset_name='validation_set',
        )
        
    train_set = EmbeddingsDataset(data=train_data)
    val_set = EmbeddingsDataset(data=val_data)
        
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=b4_cfg.training['train_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=b4_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger = Logger(b4_cfg.log_dir, 'baseline4')
    train(b4_cfg, logger, (train_loader, val_loader))