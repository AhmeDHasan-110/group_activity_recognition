import os
import argparse
import torch
from torch.utils.data import DataLoader

from generic_utils.config import DEFAULT_DEVICE, ModelConfig, initialize_data_config
from generic_utils import checkpoint as cp
from data_utils.feature_extraction import extract_features
from data_utils.data_cls import EmbeddingsDataset
from modeling.baseline4.model import TemporalClassifier
from evaluation.evaluate import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=str(DEFAULT_DEVICE),
                        help='Force a device (e.g., cpu, cuda:0)')
    args = parser.parse_args()
    device = torch.device(args.device)

    b1_cfg = ModelConfig('baseline1').get_stage('stage1')
    b4_cfg = ModelConfig('baseline4').get_stage('stage1')

    test_data = extract_features(
        extractor_cfg=b1_cfg,
        extraction_info=b4_cfg.extraction_info,
        device=device,
        subset_name='test_set'
    )

    test_set = EmbeddingsDataset(data=test_data)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=b4_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    b4_model = TemporalClassifier(**b4_cfg.model)
    b4_cp_path = os.path.join(b4_cfg.out_dir, 'checkpoints', 'last_checkpoint.pth')

    b4_trained_model, *_ = cp.load_checkpoint(
        cp_path=b4_cp_path,
        model=b4_model,
        device=device
    )

    data_cfg = initialize_data_config()
    target_names = sorted(data_cfg.group_labels.keys(), key = lambda x: data_cfg.group_labels[x])
    name = 'Baseline4'

    evaluate(
        model=b4_trained_model, 
        data_loader=test_loader,
        output_path=b4_cfg.out_dir,
        target_names=target_names,
        baseline_name=name,
        device=device
    )
