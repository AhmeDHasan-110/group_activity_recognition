import os
import numpy as np
import pandas as pd
import time
import tempfile
import torch
from torch.utils.data import DataLoader
from torch.ao.quantization import quantize_dynamic
from torch.utils.data.sampler import SubsetRandomSampler

from generic_utils.config import SAVED_EMBEDDINGS_DIR, StageConfig, DataConfig
from evaluation.evaluate import calc_metrics, save_results
from data_utils.functions import player_lvl_transforms
from data_utils.data_cls import VolleyballTemporalDataset, EmbeddingsDataset
from modeling.baseline1.model import ResnetClassifier
from .model import GroupLSTMClassifier, ResnetLSTMClassifier
from generic_utils import checkpoint as cp 

def extract(model, data_loader, split_groups: bool, subset_name: str, device):
    model = model.to(device)
    model.eval()
    all_outputs, all_labels, all_sizes = [], [], []
    total_time = 0.0
    
    with torch.inference_mode():
        for data, labels, sizes in data_loader:
            data = data.to(device)
            
            start_time = time.perf_counter()
            output = model(data)
            end_time = time.perf_counter()
            total_time += (end_time - start_time)
            
            all_outputs.append(output.cpu())
            all_labels.append(labels.cpu())
            all_sizes.append(sizes)
            
    all_outputs = torch.split(torch.cat(all_outputs), torch.cat(all_sizes).tolist()) 
    all_outputs = pool(all_outputs, split_groups)
    all_labels = torch.cat(all_labels)

    if len(all_outputs) != len(all_labels):
        raise ValueError("data and labels must have the same length.")
    data_dict = dict(zip(all_outputs, all_labels))

    path = os.path.join(SAVED_EMBEDDINGS_DIR, 'b8_embds', f'{subset_name}.pth')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(data_dict, path)
    print(f'Data & Labels dict for {subset_name} saved at: {path}')
    
    avg_latency_per_sample = (total_time / len(data_dict)) * 1000
    return data_dict, avg_latency_per_sample
    
def pool(data: list[torch.Tensor], split_groups: bool) -> torch.Tensor:
    if split_groups:
        res = []
        for x in data:
            m = x.shape[0] // 2
            pool1 = torch.max(x[:m], dim=0).values
            pool2 = torch.max(x[m:], dim=0).values
            res.append(torch.hstack((pool1, pool2)))
    else:
        res = [torch.max(x, dim=0).values for x in data]
    return torch.stack(res)

def evaluate(model, data_loader, device, save_metrics, output_path=None, target_names=None, baseline_name=None):
    """Evaluate the model on the test set and calculate the inference time."""
    
    model = model.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    total_loss, total_time = 0, 0.0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for data, labels in data_loader:
            data = data.to(device) 
            labels = labels.to(device)
            
            start_time = time.perf_counter()
            pred = model(data)
            end_time = time.perf_counter()    
            total_time += (end_time - start_time)

            loss = criterion(pred, labels)
            total_loss += loss.item()

            pred_idx = torch.argmax(pred, dim=1)

            all_preds.append(pred_idx.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    loss, accuracy, cr, cm = calc_metrics(total_loss, all_labels, all_preds, target_names)
    if save_metrics:
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            print("NOTE: The given output_path couldn't be found so it was created automatically.")
    
        eval_dir_path = os.path.join(output_path, 'evaluation_results')
        os.makedirs(eval_dir_path, exist_ok=True)
        save_results(eval_dir_path, target_names, baseline_name, loss, accuracy, cr, cm)
    
    return accuracy / 100, total_time / len(all_labels) * 1000


def benchmark_on_test_set(st1_cfg: StageConfig, st2_cfg: StageConfig,
                        data_cfg: DataConfig, quantize: bool, subset_frac: float, device: torch.device):
    """Benchmarks model performance on a test subset with optional quantization.

    Args:
        quantize: Whether to apply dynamic quantization to both stages.
        subset_frac: Fraction of the test set to use for evaluation.
        device: Torch device (CPU) for inference.

    Returns:
        A dictionary containing latency, size, and accuracy metrics for both stages.
    """
    assert st2_cfg.extraction_info is not None
    
    st1_model, st2_model, st1_data_loader = prepare_objs(st1_cfg , st2_cfg, data_cfg, subset_frac, device)
    if quantize:
        st1_model = dynamic_quantize(st1_model)
        st2_model = dynamic_quantize(st2_model)
    
    print("Warming up stage 1...")
    st1_model = warm_up(st1_model,
                        torch.rand(7, *st1_data_loader.dataset.__getitem__(0)[0].shape[-3:]), 10)
    
    print("Running stage 1...")
    test_data, st1_latency = extract(
        model=st1_model, 
        data_loader=st1_data_loader, 
        split_groups=st2_cfg.extraction_info['split_groups'],
        subset_name='test_set',
        device=device
    )
    
    test_set = EmbeddingsDataset(test_data)
    st2_data_loader = DataLoader(
        dataset=test_set,
        batch_size=200,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    target_names = sorted(data_cfg.group_labels.keys(), key = lambda x: data_cfg.group_labels[x])
    
    print("Warming up stage 2...")
    st2_model = warm_up(st2_model, torch.rand(4, *test_set.__getitem__(0)[0].shape), 10)
    print("Running stage 2...")
    accuracy, st2_latency = evaluate(
        model=st2_model, 
        data_loader=st2_data_loader,
        save_metrics=False,
        device=device
    )
    metrics = {
        'stage1_latency': st1_latency,
        'stage1_size': get_model_size(st1_model),   
        'stage2_latency': st2_latency,
        'stage2_size': get_model_size(st2_model),
        'accuracy': accuracy 
    }
    return metrics

def create_quantization_benchmark(unquantized_metrics, quantized_metrics):
    """
    Creates a comparison DataFrame for quantized vs unquantized hierarchical models.
    
    Expected input format for metrics dictionaries:
    {
        'stage1_latency': float, # e.g., in milliseconds
        'stage1_size': float,    # e.g., in MB
        'stage2_latency': float,
        'stage2_size': float,
        'accuracy': float        # e.g., 0.85 for 85%
    }
    """
    
    unq_total_latency = unquantized_metrics['stage1_latency'] + unquantized_metrics['stage2_latency']
    q_total_latency = quantized_metrics['stage1_latency'] + quantized_metrics['stage2_latency']
    
    unq_total_size = unquantized_metrics['stage1_size'] + unquantized_metrics['stage2_size']
    q_total_size = quantized_metrics['stage1_size'] + quantized_metrics['stage2_size']
    
    data = {
        'Metric': [
            'Stage 1 Latency (ms)',
            'Stage 2 Latency (ms)',
            'Total Latency (ms)',
            'Stage 1 Size (MB)',
            'Stage 2 Size (MB)',
            'Total Size (MB)',
            'Test Accuracy (%)'
        ],
        'Unquantized': [
            unquantized_metrics['stage1_latency'],
            unquantized_metrics['stage2_latency'],
            unq_total_latency,
            unquantized_metrics['stage1_size'],
            unquantized_metrics['stage2_size'],
            unq_total_size,
            unquantized_metrics['accuracy'] * 100 
        ],
        'Quantized': [
            quantized_metrics['stage1_latency'],
            quantized_metrics['stage2_latency'],
            q_total_latency,
            quantized_metrics['stage1_size'],
            quantized_metrics['stage2_size'],
            q_total_size,
            quantized_metrics['accuracy'] * 100
        ]
    }
    df = pd.DataFrame(data)
    
    def calculate_delta(row):
        if 'Accuracy' in row['Metric']:
            diff = row['Quantized'] - row['Unquantized']
            return f"{diff:+.2f} pp"
        else:
            if row['Unquantized'] == 0: 
                return "0.00%"
            change = ((row['Quantized'] - row['Unquantized']) / row['Unquantized']) * 100
            return f"{change:+.2f}%"
            
    df['Delta'] = df.apply(calculate_delta, axis=1)
    
    df['Unquantized'] = df['Unquantized'].round(2)
    df['Quantized'] = df['Quantized'].round(2)
    
    df.set_index('Metric', inplace=True)
    
    return df

def prepare_objs(st1_cfg: StageConfig, st2_cfg: StageConfig, data_cfg: DataConfig, subset_frac: float | None, device: torch.device):
    assert st2_cfg.extraction_info is not None
    assert st1_cfg.data is not None

    _, v_transform = player_lvl_transforms(st1_cfg.data['resize_dims'])
    kwargs = {
        'data_cfg' : data_cfg,
        'transform' : v_transform,
        'imgs_per_clip': st2_cfg.extraction_info['imgs_per_clip'],
        'group_level': st2_cfg.extraction_info['group_level'],
        'player_label': st2_cfg.extraction_info['player_label'],
        'enforce_sorting': st2_cfg.extraction_info['enforce_sorting']
    }
    test_set = VolleyballTemporalDataset(**kwargs, subset_dir_name='test_set')

    if subset_frac is not None:
        total_size = len(test_set)
        subset_size = int(subset_frac * total_size)
        indices = np.random.choice(total_size, subset_size, replace=False)
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = None
    
    def collate_fn(batch):
        data, labels = tuple(zip(*batch))
        g_idx, _ = tuple(zip(*labels))
    
        sizes = torch.tensor([x.size(0) for x in data])
        
        data = torch.cat(data, dim=0)
        c, h, w = data[0].size()[-3:]
        data = data.view((-1, c, h, w))
    
        return data, torch.tensor(g_idx), sizes
        
    st1_data_loader = DataLoader(
        dataset=test_set,
        batch_size=st2_cfg.extraction_info['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn
    )
    
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
    
    input_size = ResnetLSTMClassifier(
        st1_cfg.model, backbone=ResnetClassifier(num_cls=4),
        imgs_per_clip=st2_cfg.extraction_info['imgs_per_clip'], extraction_mode=True
    ).get_output_size
    
    st2_model = GroupLSTMClassifier(
        st2_cfg.model, 
        input_size=input_size * (st2_cfg.extraction_info['split_groups'] + 1)
    )
    
    st2_trained_model, *_ = cp.load_checkpoint(
        model=st2_model,
        cp_path=os.path.join(st2_cfg.out_dir, 'checkpoints', 'st2_last_checkpoint.pth'),
        device=device
    )

    st1_trained_model.eval()
    st2_trained_model.eval()
    return st1_trained_model, st2_trained_model, st1_data_loader

def dynamic_quantize(model):
    return quantize_dynamic(
            model,  
            {torch.nn.LSTM, torch.nn.Linear},
            dtype=torch.qint8
        )

def get_model_size(model, path='temp_model.pth'):
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / 1e6
    return size
    
def warm_up(model, dummy_input, runs):
    with torch.inference_mode():
        for _ in range(runs):
            _ = model(dummy_input)
        return model
