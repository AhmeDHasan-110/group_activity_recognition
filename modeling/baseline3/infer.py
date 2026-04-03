import os
import argparse
import torch
from torch.utils.data import DataLoader

from evaluation.evaluate import evaluate
from generic_utils.config import ModelConfig, initialize_data_config, DEFAULT_DEVICE
from generic_utils import checkpoint as cp
from data_utils.feature_extraction import extract_features
from data_utils.data_cls import EmbeddingsDataset
from modeling.baseline3.st2_model import nnClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default=str(DEFAULT_DEVICE),
                        help='Force a device (e.g., cpu, cuda:0)')
    args = parser.parse_args()
    device = torch.device(args.device)

    model_cfg = ModelConfig('baseline3')

    stage1_cfg = model_cfg.get_stage('stage1')
    stage2_cfg = model_cfg.get_stage('stage2')

    test_data = extract_features(
        extractor_cfg=stage1_cfg,
        extraction_info=stage2_cfg.extraction_info,
        device=device,
        subset_name='test_set'
    )
    
    test_set = EmbeddingsDataset(data=test_data)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=stage2_cfg.training['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    st2_model = nnClassifier(
        input_sz=stage2_cfg.model['input_size'],
        hidden_sz=stage2_cfg.model['hidden_layer_sizes'],
        p_dropout=stage2_cfg.model['p_dropout'],
        num_classes=stage2_cfg.model['num_classes']
    )
    
    st2_cp_path = os.path.join(stage2_cfg.out_dir, 'checkpoints', 'st2_last_checkpoint.pth')
    trained_model, *_ = cp.load_checkpoint(
            cp_path=st2_cp_path,
            model=st2_model,
            device=device
    )
    
    data_cfg = initialize_data_config()
    target_names = sorted(data_cfg.group_labels.keys(), key = lambda x: data_cfg.group_labels[x])
    name = 'Baseline3'

    evaluate(
        model=trained_model, 
        data_loader=test_loader,
        output_path=stage2_cfg.out_dir,
        target_names=target_names,
        baseline_name=name,
        device=device
    )