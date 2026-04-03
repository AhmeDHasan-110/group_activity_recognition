import os
import numpy as np
import yaml
import torch # type: ignore
from dataclasses import dataclass

RANDOM_SEED = 12
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

BASELINES_DIR = os.path.join(PROJECT_ROOT, "modeling")
DATA_CONFIG_PATH = os.path.join(PROJECT_ROOT, "data_utils", "data_config.yml")
SAVED_PICKLES_DIR = os.path.join(PROJECT_ROOT, "processed_data", "data_pickles")

SAVED_EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "processed_data", "embeddings")


def set_seed():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ModelConfig:
    """
    Initializes config for a specific baseline
    
    Args:
    baseline_name (str): The name of the specific baseline for which configuration is initialized
    """
    def __init__(self, baseline_name: str):
        self.baseline_dir = os.path.join(BASELINES_DIR, baseline_name)
        if not os.path.exists(self.baseline_dir):
            raise FileNotFoundError(f"No such directory: {self.baseline_dir}")

        self.out_dir = os.path.join(self.baseline_dir, "results")
        self.log_dir = os.path.join(self.out_dir, "logs")
        self.config_path = os.path.join(self.baseline_dir, "config.yml")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.device = data['device']
        self.type = data['type']

        self.stages = {}
        if self.type == 'multi-stage':
            self.stages['stage1'] = StageConfig('stage1', data['stage1'], self)
            self.stages['stage2'] = StageConfig('stage2', data['stage2'], self)
        else:
            self.stages['stage1'] = StageConfig('stage1', data, self)
        
    def get_stage(self, stage_name: str):
        return self.stages[stage_name]

    def __repr__(self):
        return f"ModelConfig of baseline: {os.path.basename(self.baseline_dir)}"

class StageConfig:
    def __init__(self, name: str, data: dict, parent_cfg: ModelConfig):
        self.parent = parent_cfg
        self.name = name

        self.config_path = parent_cfg.config_path
        self.out_dir = parent_cfg.out_dir
        self.log_dir = parent_cfg.log_dir

        self.data = data.get('data', None)
        self.model = data['model']
        self.training = data['training']

        self.extraction_info = data.get('extraction_process', None)
        self.scheduler = data['training'].get('scheduler', None)
        self.device = self.parent.device

    def __repr__(self):
        return f"StageConfig: {self.name} of {os.path.basename(self.parent.baseline_dir)}"


@dataclass
class DataConfig:
    """
    Data structure holding all dataset parameters.
    
    Args:
        root (str): Path to the dataset root directory.
        train_subset (list[int]): List of video IDs for training.
        val_subset (list[int]): List of video IDs for validation.
        test_subset (list[int]): List of video IDs for testing.
        group_labels (dict[str, int]): Dictionary mapping group labels to indexes
        player_labels (dict[str, int]): Dictionary mapping player labels to indexes
    """
    root: str
    train_subset: list[int]
    val_subset: list[int]
    test_subset: list[int]
    group_labels: dict[str, int]
    player_labels: dict[str, int]


def initialize_data_config() -> DataConfig:
    """
    Parses the YAML file and builds the DataConfig object.
    
    Returns:
        DataConfig: The populated configuration object.
    """
    try:
        with open(DATA_CONFIG_PATH, 'r') as cf:
            cfg = yaml.safe_load(cf)
    except yaml.YAMLError as e:
        print(f'Error parsing yaml: {e}')
        return

    group_labels: list[dict] = cfg['categories']['group_labels']
    player_labels: list[dict] = cfg['categories']['player_labels']
    
    g_dict = {}
    for item in group_labels:
        g_dict.update(item)

    p_dict = {}
    for item in player_labels:
        p_dict.update(item)
    
    params = {
        'root' : cfg['root'],
        'train_subset' : cfg['subsets']['train_set'],
        'val_subset' : cfg['subsets']['validation_set'],
        'test_subset' : cfg['subsets']['test_set'],
        'group_labels' : g_dict,
        'player_labels' : p_dict
    }

    return DataConfig(**params)