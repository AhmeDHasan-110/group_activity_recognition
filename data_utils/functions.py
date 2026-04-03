import torch
from torchvision.transforms import v2

def group_lvl_transforms(resize_dims: list):
    dims = resize_dims[0] if len(resize_dims) == 1 else resize_dims

    train_transform= v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=dims, scale=(0.5, 0.9)),
        v2.RandomRotation(degrees=5),
        v2.ColorJitter(brightness=0.1, contrast=0.1,
                        saturation=0.1, hue=0.02),
        
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(256),

        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform

def player_lvl_transforms(resize_dims: list):
    dims = resize_dims[0] if len(resize_dims) == 1 else resize_dims
    
    train_transform = v2.Compose([
        v2.ToImage(), 
        
        v2.Resize(dims),
        v2.RandomHorizontalFlip(p=0.5),
        v2.TrivialAugmentWide(),
        
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(dims),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_scheduler(cfg, optimizer):
    sched_type = cfg.scheduler['type']

    if sched_type == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=cfg.scheduler['factor'],
            patience=cfg.scheduler['patience'],
            threshold=cfg.scheduler['threshold'],
            min_lr=cfg.scheduler['min_lr']
        )
    elif sched_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler['T_max'],
            eta_min=cfg.scheduler['eta_min']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

