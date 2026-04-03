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
    initialize_data_config,
    set_seed,
    ModelConfig,
    StageConfig,
    )

from modeling.baseline8.model import GroupLSTMClassifier, ResnetLSTMClassifier
from data_utils.data_cls import EmbeddingsDataset, VolleyballTemporalDataset
from data_utils.functions import get_scheduler, player_lvl_transforms
from modeling.baseline1.model import ResnetClassifier
from generic_utils.log import Logger, log_curves
from evaluation.validation import validate
import generic_utils.checkpoint as cp
from modeling.baseline8.eval_functions import extract


def train(model: GroupLSTMClassifier, config: StageConfig, logger: Logger, data_loaders: tuple):
    logger.info('Training Logs for Baseline8 - Stage2\n')
    logger.info(f'Device used: {device}')

    train_loader, val_loader = data_loaders
    
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

        val_loss, val_acc, val_f1_macro, val_f1_all = validate(model, val_loader, device)

        train_loss_curve.append(loss)
        train_acc_curve.append(acc)
        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_acc)

        logger.info(f'End of epoch {epoch}: train loss: {loss:.4f}  -  train accuracy: {acc:.2f}%')
        logger.info(f'End of epoch {epoch}: val loss: {val_loss:.4f}  -\
  val accuracy: {val_acc:.2f}%  -  val macro f1 score: {val_f1_macro:.4f}\n')
        
        scheduler.step()

    cp.save_checkpoint(model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=config.training['epochs'],
                    val_acc=val_acc, config_path=config.config_path,
                    output_path=config.out_dir, cp_name='st2_last_checkpoint')
    logger.info('Last model checkpoint saved')

    curves = (train_loss_curve, val_loss_curve, train_acc_curve, val_acc_curve)
    names = ('train loss', 'validation loss', 
            'train accuracy', 'validation accuracy')
    txt = log_curves(curves, names, config.log_dir, file_name='st2')
    logger.info(txt)

def call_extract(st1_cfg, st2_cfg, device):
    st1_model = ResnetLSTMClassifier(
        st1_cfg.model,
        backbone=ResnetClassifier(num_cls=4),
        imgs_per_clip=st2_cfg.extraction_info['imgs_per_clip'],
        extraction_mode=True
    ).to(device)

    st1_trained_model, *_ = cp.load_checkpoint(
        cp_path=os.path.join(st1_cfg.out_dir, 'checkpoints', 'st1_last_checkpoint.pth'),
        model=st1_model,
        device=device
    )

    _, v_transform = player_lvl_transforms(st1_cfg.data['resize_dims'])
    kwargs = {
        'data_cfg' : initialize_data_config(),
        'transform' : v_transform,
        'imgs_per_clip': st2_cfg.extraction_info['imgs_per_clip'],
        'group_level': st2_cfg.extraction_info['group_level'],
        'player_label': st2_cfg.extraction_info['player_label'],
        'enforce_sorting': st2_cfg.extraction_info['enforce_sorting']
    }
    train_set = VolleyballTemporalDataset(**kwargs, subset_dir_name='train_set')
    val_set = VolleyballTemporalDataset(**kwargs, subset_dir_name='validation_set')

    def collate_fn(batch):
        data, labels = tuple(zip(*batch))
        g_idx, _ = tuple(zip(*labels))

        sizes = torch.tensor([x.size(0) for x in data])
        
        data = torch.cat(data, dim=0)
        c, h, w = data[0].size()[-3:]
        data = data.view((-1, c, h, w))

        return data, torch.tensor(g_idx), sizes
        
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=st2_cfg.extraction_info['batch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=st2_cfg.extraction_info['batch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn
    )
        
    train_dict = extract(
        model=st1_trained_model, 
        data_loader=train_loader,
        device=device,
        split_groups=st2_cfg.extraction_info['split_groups'],
        subset_name='train_set',
    )
    val_dict = extract(
        model=st1_trained_model, 
        data_loader=val_loader, 
        device=device,
        split_groups=st2_cfg.extraction_info['split_groups'],
        subset_name='validation_set'
    )
    return train_dict, val_dict


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
    
    st1_cfg = ModelConfig('baseline8').get_stage('stage1')
    st2_cfg = ModelConfig('baseline8').get_stage('stage2')

    embds_path = os.path.join(SAVED_EMBEDDINGS_DIR, 'b8_embds')
    if args.extract and os.path.exists(embds_path):
        train_data = torch.load(os.path.join(embds_path, 'train_set.pth'))
        val_data = torch.load(os.path.join(embds_path, 'validation_set.pth'))
    else:
        train_data, val_data = call_extract(st1_cfg, st2_cfg, device)

    train_set = EmbeddingsDataset(train_data)
    val_set = EmbeddingsDataset(val_data)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=st2_cfg.training['train_batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=st2_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    input_size = ResnetLSTMClassifier(
        st1_cfg.model, backbone=ResnetClassifier(num_cls=4),
        imgs_per_clip=st2_cfg.extraction_info['imgs_per_clip'], extraction_mode=True
    ).get_output_size
    
    model = GroupLSTMClassifier(
        st2_cfg.model, 
        input_size=input_size * (st2_cfg.extraction_info['split_groups'] + 1)
    ).to(device)
    
    logger = Logger(st2_cfg.log_dir, 'baseline8_st2')
    train(model, st2_cfg, logger, (train_loader, val_loader))
