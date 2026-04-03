import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from imblearn.under_sampling import RandomUnderSampler
from generic_utils.config import SAVED_PICKLES_DIR, DataConfig

class VolleyballGroupDataset(Dataset):
    """
    Dataset for group-level annotations. Each sample is a single frame from a clip,
    returning either the full frame (group-level) or individual player crops (player-level)
    along with the group activity label.

    Args:
        data_cfg (DataConfig): Main configuration object containing data configuration.
        subset_dir_name (str): Subdirectory name (e.g., 'train_set' or 'test_set').
        imgs_per_clip (int): Number of images extracted per clip. Extracted from the middle of the clip.
        group_level (bool): If True, returns group-level labels. If False, returns player-level.
        transform (callable, optional): Image transformations to apply. Defaults to None.
    """
    def __init__(self, data_cfg: DataConfig, subset_dir_name: str, imgs_per_clip: int = 9, group_level: bool = True, transform=None):
        self.data_cfg = data_cfg
        self.imgs_per_clip = imgs_per_clip
        self.group_level = group_level

        self.path = os.path.join(SAVED_PICKLES_DIR, subset_dir_name)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Subset directory does not exist at: {self.path}")
        
        if transform:
            self.transform = transform
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((256, 256)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])


        self._load_labels()
        self._load_img_paths()
        
        self.mapping = self.data_cfg.group_labels
        
        self._calc_cls_weights()
        self.n_cls = len(self.mapping)

    def _load_labels(self):
        """Loads labels from the stored pickle files."""

        with open(os.path.join(self.path, 'labels.pkl'), 'rb') as f:
            self.labels = pickle.load(f)
    
    def _load_img_paths(self):
        """Extracts the required number of image pathes from every clip."""
        
        n = self.imgs_per_clip // 2
        root = self.data_cfg.root
        imgs = []
        
        for label in self.labels:
            clip = int(os.path.basename(label))
            
            # get the middle `imgs_per_clip` frames of the clip
            paths = [ 
                os.path.join(root, label, f'{clip + i}.jpg')
                for i in range(-n, n+1, 1)
            ]
            
            imgs.extend(paths)
        
        self.imgs = imgs
    
    def _calc_cls_weights(self):
        """Computes balanced class weights inversely proportional to their frequency."""

        int_labels = [self.mapping[label['group_label']] for label in self.labels.values()]
        freq = torch.bincount(torch.tensor(int_labels))
        self.cls_weights = len(int_labels) / (len(freq) * freq.float() + 1e-6)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        
        parts = img_path.split("/")
        key = os.path.join(parts[-3], parts[-2])
        
        label_dict = self.labels[key]
        label_idx = self.mapping[label_dict['group_label']]

        if self.group_level:
            img = self.transform(img)
            return img, label_idx
        else:
            crops = []
            for obj in label_dict['player_labels']:
                x, y, w, h = obj.box
                crop = img.crop((x, y, x+w, y+h))

                crop = self.transform(crop)
                crops.append(crop)
            
            return torch.stack(crops), label_idx


class VolleyballPlayerDataset(Dataset):
    """
    Dataset for player-level annotations. Each sample is a single player crop from
    the center frame of a clip, paired with the individual action label. Supports
    majority-class undersampling for handling class imbalance.

    Args:
        data_cfg (DataConfig): Main configuration object containing data information.
        subset_dir_name (str): Subdirectory name (e.g., 'train_set' or 'test_set').
        transform (callable, optional): Image transformations to apply. Defaults to None
        undersample_maj (bool, optional): Whether to undersample the majority class. Defaults to False.
        undersample_precent (float, optional): How many samples to keep. Defaults to 0.5.
    """
    def __init__(self, data_cfg: DataConfig, subset_dir_name: str, transform=None, undersample_maj: bool = False, undersample_precent: float = 0.6):

        self.data_cfg = data_cfg
        self.undersample = undersample_maj
        self.undersample_precent = undersample_precent

        self.path = os.path.join(SAVED_PICKLES_DIR, subset_dir_name)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Subset directory does not exist at: {self.path}")

        if transform:
            self.transform = transform
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((256, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        self._prepare_objs()
        
        self.mapping = self.data_cfg.player_labels

        if self.undersample:
            self._apply_undersampling()

        self._calc_cls_weights()
            
            
    def _prepare_objs(self):
        """Extracts the PlayerBox objects."""
        with open(os.path.join(self.path, 'labels.pkl'), 'rb') as f:
            labels = pickle.load(f)
        self.objs = np.concatenate([label['player_labels'] for label in labels.values()])

    def _apply_undersampling(self):
        """Reduces the majority class by the specified percentage."""
        int_labels = [self.mapping[obj.label] for obj in self.objs]
        
        freq = np.bincount(int_labels) 
        
        strategy = {i: f for i, f in enumerate(freq)}
        
        majority_idx = self.mapping['Standing']
        
        strategy[majority_idx] = int(strategy[majority_idx] * self.undersample_precent)

        X = self.objs.reshape(-1, 1)
        
        sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=12)
        X_resampled, _ = sampler.fit_resample(X, int_labels)
        
        self.objs = X_resampled.flatten()

    def _calc_cls_weights(self):
        """Uses the 'Effective Number of Samples' method to calculate class weights."""
        beta = 0.995
        
        int_labels = [self.mapping[obj.label] for obj in self.objs]
        labels_tensor = torch.tensor(int_labels, dtype=torch.long)
        
        freq = torch.bincount(labels_tensor)
        weights = (1 - beta) / (1 - beta ** freq)
        
        self.cls_weights = weights / weights.sum() # normalize
        self.sample_weights = 1 / freq[int_labels]  # create sample weights in case we used WeightedRandomSampler 
        
    def __len__(self):
        return len(self.objs)

    def __getitem__(self, idx):
        obj = self.objs[idx]
        clip = os.path.basename(obj.clip_dir)
        img_path = os.path.join(self.data_cfg.root, obj.clip_dir, f'{clip}.jpg')
        x, y, w, h = obj.box
        
        with Image.open(img_path, 'r') as img:
            img = img.convert("RGB")
            crop = img.crop((x, y, x+w, y+h))

        crop = self.transform(crop)
            
        label_idx = self.mapping[obj.label]

        return crop, label_idx

class VolleyballTemporalDataset(Dataset):
    """
    Dataset for temporal (multi-frame) clips of volleyball scenes.

    Supports both group-level (full frames) and player-level (per-person crops across frames)
    data loading. Each sample is a clip of consecutive frames, optionally with player bounding
    box crops extracted per frame.

    Args:
        data_cfg (DataConfig): Main configuration object containing data information.
        subset_dir_name (str): Subdirectory name (e.g., 'train_set' or 'test_set').
        group_level (bool): If True, loads full frames per clip. If False, loads per-player crops across frames.
        player_label (bool): If True, returns both group and player-level labels. If False, returns group-level only.
        enforce_sorting (bool): If True, sorts players by their bounding box x-coordinate (left to right).
        imgs_per_clip (int, optional): Number of frames extracted per clip, centered around the middle frame. Defaults to 9.
        transform (callable, optional): Image transformations to apply. Defaults to None.
    """
    def __init__(
            self, 
            data_cfg: DataConfig, 
            subset_dir_name: str, 
            group_level: bool,
            player_label: bool,
            enforce_sorting: bool,
            imgs_per_clip: int = 9, 
            transform= None
        ):
        self.data_cfg = data_cfg
        self.group_level = group_level
        self.player_label = player_label
        self.enforce_sorting = enforce_sorting
        self.imgs_per_clip = imgs_per_clip
        
        self.path = os.path.join(SAVED_PICKLES_DIR, subset_dir_name)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Subset directory does not exist at: {self.path}")
        
        if transform:
            self.transform = transform
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((256, 256)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        self.g_labels_mapping = data_cfg.group_labels
        self.p_labels_mapping = data_cfg.player_labels

        self._prepare_objs()

    def _prepare_objs(self):
        """Loads the labels and extracts the PlayerBox objects."""
        with open(os.path.join(self.path, 'labels.pkl'), 'rb') as f:
            self.labels = pickle.load(f)
        self.objs = [label['player_labels'] for label in self.labels.values()]
    
    def _gen_paths(self, path) -> list[str]:
        """Generates the paths of a clip's frames."""
        clip = int(os.path.basename(path))
        n = self.imgs_per_clip // 2
        root = self.data_cfg.root

        paths = [ 
            os.path.join(root, path, f'{clip + i}.jpg')
            for i in range(-n, n+1, 1)
        ]
        return paths
    
    def __len__(self):
        return len(self.objs)
    
    def __getitem__(self, idx):
        clip_objs = sorted(self.objs[idx], key=lambda x: x.box[0]) if self.enforce_sorting else self.objs[idx]
        clip_dir = clip_objs[0].clip_dir
        clip_imgs = self._gen_paths(clip_dir)

        label_dict = self.labels[clip_dir]
        group_label_idx = self.g_labels_mapping[label_dict['group_label']]

        if self.group_level:
            data = []
            for img_path in clip_imgs:
                with Image.open(img_path, 'r') as img:
                    img = img.convert('RGB')
                    data.append(self.transform(img))
        else:
            all_crops = [[] for _ in clip_objs]

            for img_path in clip_imgs:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")

                    for i, obj in enumerate(clip_objs):
                        x, y, w, h = obj.box
                        crop = img.crop((x, y, x + w, y + h))
                        all_crops[i].append(self.transform(crop))
            
            data = [torch.stack(crops) for crops in all_crops]

        data = torch.stack(data)
        if self.player_label:
            player_label_idx = torch.tensor([self.p_labels_mapping[obj.label] for obj in clip_objs])
            return data, (group_label_idx, player_label_idx)
        else:
            return data, group_label_idx

class EmbeddingsDataset(Dataset):
    """Dataset for pre-extracted embeddings."""
    def __init__(self, data: dict[torch.Tensor, int | list[int]]):
        self.data, self.labels = list(data.keys()), list(data.values())
    def __len__(self):
        return len(self.labels) 
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

