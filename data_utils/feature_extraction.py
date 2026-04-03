import os
import numpy as np
from itertools import chain
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

from generic_utils.config import SAVED_EMBEDDINGS_DIR
from generic_utils import checkpoint as cp 
from generic_utils.config import initialize_data_config, StageConfig
from data_utils.data_cls import VolleyballGroupDataset, VolleyballTemporalDataset
from data_utils.functions import group_lvl_transforms, player_lvl_transforms
from modeling.baseline1.model import ResnetClassifier


def extract_features(
        extractor_cfg: StageConfig, 
        extraction_info: dict, 
        device: torch.device, 
        subset_name: str
    ):
    """
    This function initializes the model and data loader via the `prepare()`
    utility, runs inference to extract features, optionally pools them, and returns
    a new loader containing the processed features.

    Args:
        extractor_cfg (StageConfig, required): Config object of the extractor model.
        extraction_info (dict, required): Dictionary contains the extraction process information.
        subset_name (str, required): Name of the dataset subset (e.g., 'train_set', 'test_set').
        device (str, required): (Keyword Arg) The used device.
    
    Returns:
        dict[torch.Tensor, int | list[int]]: A dictionary containing the extracted and pooled feature tensors and their labels.
    """
    model, data_loader = _prepare(extractor_cfg, extraction_info, device, subset_name)
    
    model.model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    all_sizes = []
    
    with torch.no_grad():
        for data, sizes, labels in data_loader:
            data = data.to(device)
            features = model(data)
            all_features.append(features.cpu())
            all_labels.append(labels)
            all_sizes.append(sizes)
        
    all_features = torch.cat(all_features)
    all_labels = list(chain.from_iterable(all_labels))
    sizes = np.vstack(all_sizes).T
    
    if extraction_info['pool']:
        data = _split_and_pool(all_features, sizes)
    else:
        data = _split(all_features, sizes)

    if len(all_labels) != len(data):
        raise ValueError("data and labels must have the same length.")
    data_dict = dict(zip(data, all_labels))

    if subset_name != 'test_set': # only save the train and val datasets
        path = os.path.join(SAVED_EMBEDDINGS_DIR, extraction_info['save_dir_name'], f'{subset_name}.pth')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(data_dict, path)
        print(f'Data & Labels dict for {subset_name} saved at: {path}')

    return data_dict

def _prepare(extractor_config, extraction_info, device, subset_name):
    """Prepares the model and data loader for feature extraction."""
    model = ResnetClassifier(num_cls=extractor_config.model['num_classes'])
    cp_path = os.path.join(
        extractor_config.out_dir, 
        'checkpoints', 
        'last_checkpoint.pth' if extractor_config.parent.type == 'single-stage' else 'st1_last_checkpoint.pth'
    )

    model, *_ = cp.load_checkpoint(
        cp_path=cp_path,
        model=model,
        device=device
    )

    if extraction_info['group_level']:
        _, transform = group_lvl_transforms(resize_dims=extractor_config.data['resize_dims']) 
    else:
        _, transform = player_lvl_transforms(resize_dims=extractor_config.data['resize_dims']) 
    
    kwargs = {
        'subset_dir_name' : subset_name,
        'imgs_per_clip' : extraction_info['imgs_per_clip'],
        'group_level' : extraction_info['group_level'],
        'transform' : transform,
        'data_cfg' : initialize_data_config(),
    }

    if extraction_info['data_cls'] == 'VolleyballGroupDataset':
        dataset = VolleyballGroupDataset(**kwargs)
    elif extraction_info['data_cls'] == 'VolleyballTemporalDataset':
        kwargs['player_label'] = extraction_info['player_label']
        kwargs['enforce_sorting'] = extraction_info['enforce_sorting']
        dataset = VolleyballTemporalDataset(**kwargs)
    
    def collate_fn(batch):
        crops, labels = zip(*batch)
        sizes = [item.size()[:-3] for item in crops]

        crops = torch.cat(crops, dim=0)
        c, h, w = crops[0].size()[-3:] # one image dimensions
        crops = crops.view((-1, c, h, w))
        
        return crops, np.array(sizes), labels

    loader = DataLoader(
        dataset=dataset,
        batch_size=extraction_info['batch_size'],
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return model, loader

def _split_and_pool(embds: torch.Tensor, sizes: np.ndarray) -> torch.Tensor:
    """Splits and pools the embeddings based on the sizes."""
    if len(sizes) == 1:
        embds = torch.split(embds, sizes[0])
        pooled_embds = torch.max(torch.stack(embds), dim=0).values
    else:
        input_len = embds.shape[-1]
        start_idx = 0
        pooled_embds = []

        for dim1, dim2 in zip(*sizes):
            end_idx = dim1 * dim2 + start_idx
            part = embds[start_idx : end_idx].view(dim1, dim2, input_len)
            pooled_embds.append(torch.max(part, dim=0).values)
            start_idx = end_idx
        
        pooled_embds = torch.stack(pooled_embds) 
    
    return pooled_embds

def _split(embds: torch.Tensor, sizes: np.ndarray) -> list[torch.Tensor]:
    """Splits the embeddings based on the sizes."""
    if len(sizes) > 1:
        s = np.repeat(sizes[1], sizes[0])
        embds = torch.stack(torch.split(embds, s.tolist()))
    return torch.split(embds, sizes[0].tolist())