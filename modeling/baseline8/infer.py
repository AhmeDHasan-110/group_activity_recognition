import os
import argparse
from torch.utils.data import DataLoader
import torch

from generic_utils.config import DEFAULT_DEVICE, ModelConfig, initialize_data_config
from data_utils.data_cls import EmbeddingsDataset
from modeling.baseline8.eval_functions import (
    create_quantization_benchmark,
    benchmark_on_test_set, 
    prepare_objs, 
    extract, 
    evaluate
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=str(DEFAULT_DEVICE),
                        help='Force a device (e.g., cpu, cuda:0)')
    parser.add_argument('-m', '--mode', type=int, default=1,
                        help='''
                            Chooses the evaluation mode on the test set.
                            options:
                                1 -> infer on test(on GPU, if exists) and save the calculated metircs.
                                2 -> create a simple benchmark between the quantized and unquantized 
                                    versions of the model on a subset of the test set(runs on CPU only).
                            ''')
    
    args = parser.parse_args()
    device = torch.device(args.device)

    st1_cfg = ModelConfig('baseline8').get_stage('stage1')
    st2_cfg = ModelConfig('baseline8').get_stage('stage2')
    data_cfg = initialize_data_config()

    if args.mode == 1:
        st1_model, st2_model, st1_data_loader = prepare_objs(st1_cfg, st2_cfg, data_cfg, None, device)

        test_data, _ = extract(
            model=st1_model, 
            data_loader=st1_data_loader, 
            split_groups=st2_cfg.extraction_info['split_groups'],
            subset_name='test_set',
            device=device
        )

        test_set = EmbeddingsDataset(test_data)
        st2_data_loader = DataLoader(
            dataset=test_set,
            batch_size=st2_cfg.training['val_batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        target_names = sorted(data_cfg.group_labels.keys(), key = lambda x: data_cfg.group_labels[x])
        _ = evaluate(
            model=st2_model, 
            data_loader=st2_data_loader,
            output_path=st2_cfg.out_dir,
            target_names=target_names,
            baseline_name='Baseline8',
            save_metrics=True,
            device=device
        )
    
    elif args.mode == 2:
        print('Unquantized model experiment:-')
        unquant_metrics = benchmark_on_test_set(st1_cfg, st2_cfg, data_cfg, False, 0.5, 'cpu')
        print('\nQuantized model experiment:-')
        quant_metrics = benchmark_on_test_set(st1_cfg, st2_cfg, data_cfg, True, 0.5, 'cpu')

        benchmark_data = create_quantization_benchmark(unquant_metrics, quant_metrics)
        # save the dataframe to excel
        benchmark_data.to_excel(
            os.path.join(st2_cfg.out_dir, 'evaluation_results', 'dynamic_quant_benchmark.xlsx'), 
            index=True
        )