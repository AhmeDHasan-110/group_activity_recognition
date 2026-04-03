import argparse
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim 

from generic_utils.config import (
    initialize_data_config,
    ModelConfig, 
    StageConfig, 
    DEFAULT_DEVICE, 
    set_seed
)
from data_utils.functions import group_lvl_transforms
from data_utils.data_cls import VolleyballGroupDataset
from data_utils.functions import get_scheduler
from generic_utils import checkpoint as cp
from generic_utils.log import Logger, log_curves
from evaluation.validation import validate
from modeling.baseline1.model import ResnetClassifier


def train(logger: Logger, config: StageConfig, data_loaders: tuple):
    logger.info(f'Device used: {device}')

    train_loader, val_loader = data_loaders

    model = ResnetClassifier(config.model['num_classes'], False).to(device)

    logger.info(f'The starting lr = {config.training["lr"]}')

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.training['lr'],
                            weight_decay=config.training['weight_decay'])
    
    scheduler = get_scheduler(config, optimizer)

    logger.info(f'Train data size = {len(train_loader.dataset)} ; \
Validation data size = {len(val_loader.dataset)}')
    
    best_acc = 0
    train_loss_curve, val_loss_curve = [], []
    train_acc_curve, val_acc_curve = [], []

    logger.info('Training loop logs:\n')

    for epoch in tqdm(range(1, config.training['epochs'] + 1), desc='Training'):
        logger.info(f'Epoch ({epoch} / {config.training["epochs"]}) ::')
        logger.info(f'Current lr = {optimizer.param_groups[0]["lr"]}\n')

        model.train()
        total_loss, correct, total_samples = 0, 0, 0
        offset = next_check = len(train_loader) // 3    # log training loss and accuracy 3 times per epoch

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
        logger.info(f'End of epoch {epoch}: val loss: {val_loss:.4f}  - \
    val accuracy: {val_acc:.2f}%  -  val macro f1 score: {val_f1_macro:.4f}\n')

        if val_acc > best_acc:
            best_acc = val_acc
            cp.save_checkpoint(model=model, optimizer=optimizer,
                                scheduler=scheduler, epoch=epoch,
                                val_acc=val_acc, config_path=config.config_path,
                                output_path=config.out_dir, cp_name='best_model')

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
    
    args = parser.parse_args()
    device = torch.device(args.device)

    set_seed()
    model_cfg = ModelConfig('baseline1').get_stage('stage1')

    train_trans, val_trans = group_lvl_transforms(model_cfg.data['resize_dims'])

    data_cfg = initialize_data_config()
    train_ds = VolleyballGroupDataset(
        data_cfg=data_cfg,
        subset_dir_name='train_set',
        group_level=model_cfg.data['group_level'],
        imgs_per_clip=model_cfg.data['imgs_per_clip'],
        transform=train_trans
    )

    data_cfg = initialize_data_config()
    val_ds = VolleyballGroupDataset(
        data_cfg=data_cfg,
        subset_dir_name='validation_set',
        group_level=model_cfg.data['group_level'],
        imgs_per_clip=model_cfg.data['imgs_per_clip'],
        transform=val_trans
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=model_cfg.training['train_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=model_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger = Logger(model_cfg.log_dir, 'baseline1')
    train(logger, model_cfg, (train_loader, val_loader))