import os
import argparse
import torch
from torch.utils.data import DataLoader 

from generic_utils.config import ModelConfig, initialize_data_config, DEFAULT_DEVICE
from generic_utils import checkpoint as cp
from data_utils.data_cls import VolleyballGroupDataset 
from data_utils.functions import group_lvl_transforms
from evaluation.evaluate import evaluate
from modeling.baseline1.model import ResnetClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=str(DEFAULT_DEVICE), 
                    help='Force a device (e.g., cpu, cuda:0)')
    
    args = parser.parse_args()
    device = torch.device(args.device)

    model_cfg = ModelConfig("baseline1").get_stage('stage1')

    model = ResnetClassifier(
        num_cls=model_cfg.model["num_classes"],
        freeze_backbone=False
    )

    cp_path = os.path.join(model_cfg.out_dir, "checkpoints", "last_checkpoint.pth")
    model, *_ = cp.load_checkpoint(
        cp_path=cp_path,
        model=model,
        device=device
    )

    model = model.to(device)

    _, test_transform = group_lvl_transforms(model_cfg.data['resize_dims'])

    data_cfg = initialize_data_config()

    test_dataset = VolleyballGroupDataset(
        data_cfg=data_cfg,
        subset_dir_name="test_set",
        group_level=True,
        transform=test_transform
    )

    target_names = sorted(test_dataset.mapping.keys(), key= lambda x: test_dataset.mapping[x])

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    evaluate(
        model=model,
        data_loader=test_loader,
        output_path=model_cfg.out_dir,
        target_names=target_names,
        baseline_name="B1 - Finetuned ResNet50 Classifier",
        device=device
    )
