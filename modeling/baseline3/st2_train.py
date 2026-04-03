import os
import argparse
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch 
import torch.nn as nn 
from torch import optim 
from torch.utils.data import DataLoader, TensorDataset

from generic_utils.config import (
    DEFAULT_DEVICE, 
    SAVED_EMBEDDINGS_DIR, 
    set_seed
    )
from generic_utils import checkpoint as cp 
from generic_utils.config import ModelConfig, StageConfig, initialize_data_config
from generic_utils.log import Logger, log_curves
from data_utils.data_cls import EmbeddingsDataset
from data_utils.feature_extraction import extract_features
from data_utils.functions import player_lvl_transforms, get_scheduler
from st2_model import nnClassifier
from evaluation.validation import validate

def train(logger: Logger, config: StageConfig, data_loaders: tuple):
    logger.info('Training Logs for Baseline3 - Stage2\n')
    logger.info(f'Device used: {device}')

    model = nnClassifier(
        input_sz=config.model['input_size'],
        hidden_sz=config.model['hidden_layer_sizes'],
        p_dropout=config.model['p_dropout'],
        num_classes=config.model['num_classes']
    ).to(device)

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
                    output_path=config.out_dir, cp_name='st2_last_checkpoint')
    logger.info('Last model checkpoint saved')

    curves = (train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve)
    names = ('train loss', 'validation loss', 'train accuracy', 'validation accuracy')
    txt = log_curves(curves, names, config.log_dir, 'stage2')
    logger.info(txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--device', type=str, default=str(DEFAULT_DEVICE), 
        help='Force a device (e.g., cpu, cuda:0)'
    )
    parser.add_argument(
        '-e', '--extract',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable or disable the feature extraction process (default: disabled).'
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    set_seed()

    model_cfg = ModelConfig('baseline3')
    stage1_cfg = model_cfg.get_stage('stage1')
    stage2_cfg = model_cfg.get_stage('stage2')

    path = os.path.join(SAVED_EMBEDDINGS_DIR, 'b3_embds')
    if args.extract and os.path.exists(path):
        train_data = torch.load(os.path.join(path, 'train_set.pth'))
        val_data = torch.load(os.path.join(path, 'validation_set.pth'))
    else:
        train_data = extract_features(
            extractor_cfg=stage1_cfg,
            extraction_info=stage2_cfg.extraction_info,
            device=device,
            subset_name='train_set',
        )

        val_data = extract_features(
            extractor_cfg=stage1_cfg,
            extraction_info=stage2_cfg.extraction_info,
            device=device,
            subset_name='validation_set',
        )

    train_set = EmbeddingsDataset(data=train_data)
    val_set = EmbeddingsDataset(data=val_data)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=stage2_cfg.training['train_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=stage2_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger = Logger(stage2_cfg.log_dir, 'baseline3_st2')
    train(logger, stage2_cfg, (train_loader, val_loader))