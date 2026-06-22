'''
Purpose: It is optional to evaluate the inference efficiency after training, including time, memory usage and parameter quantity.
Input: Trained model, validation edge, feature graph, PPI graph and profiling parameters.
Output: efficiency_profile.csv; Currently, profiling details are not printed to the console by default.
'''

import math
import os
import time

import numpy as np
import torch

from model.training.helpers import count_trainable_params
from model.utils.utils import print_file

@torch.no_grad()
def profile_inference_efficiency(
    batch,
    p_x_all,
    p_edge_all,
    p_edge_arr,
    p_seq_all,
    model,
    graph,
    device,
    save_path,
    batch_size=11000,
    warmup_steps=5,
    repeat=5,
    use_best_ckpt=True,
):
    profile_path = os.path.join(save_path, 'efficiency_profile.txt')

    ckpt_path = os.path.join(save_path, 'gnn_model_valid_best.ckpt')

    if use_best_ckpt and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        print_file(
            f'[Efficiency] Loaded best checkpoint from {ckpt_path}',
            save_file_path=profile_path
        )
    else:
        print_file(
            '[Efficiency] Best checkpoint not found. Profiling current model state.',
            save_file_path=profile_path
        )

    model.eval()

    edge_ids = list(graph.val_mask)
    total_eval_edges = len(edge_ids)
    steps = math.ceil(total_eval_edges / batch_size)

    edge_batches = []
    for step in range(steps):
        if step == steps - 1:
            valid_edge_id = edge_ids[step * batch_size:]
        else:
            valid_edge_id = edge_ids[step * batch_size: step * batch_size + batch_size]
        edge_batches.append(valid_edge_id)

    params_m = count_trainable_params(model)

    prop_edge_index = graph.edge_index_got

    for i in range(min(warmup_steps, len(edge_batches))):
        valid_edge_id = edge_batches[i]
        pred_edge_index = graph.edge_index[:, valid_edge_id]

        _ = model(
            batch,
            p_x_all,
            p_edge_all,
            p_edge_arr,
            p_seq_all,
            prop_edge_index,
            pred_edge_index,
            graph,
            return_align=False
        )

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    repeat_times = []

    for _ in range(repeat):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        start = time.perf_counter()

        for valid_edge_id in edge_batches:
            pred_edge_index = graph.edge_index[:, valid_edge_id]

            _ = model(
                batch,
                p_x_all,
                p_edge_all,
                p_edge_arr,
                p_seq_all,
                prop_edge_index,
                pred_edge_index,
                graph,
                return_align=False
            )

        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start
        repeat_times.append(elapsed)

    repeat_times = np.array(repeat_times, dtype=float)

    time_s_full_eval_mean = repeat_times.mean()
    time_s_full_eval_std = repeat_times.std(ddof=1) if repeat > 1 else 0.0
    time_ms_per_batch = (time_s_full_eval_mean / steps) * 1000.0
    time_s_per_1k_edges = time_s_full_eval_mean / total_eval_edges * 1000.0

    if device.type == 'cuda':
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    else:
        peak_mem_gb = 0.0

    msg = (
        '\n[Efficiency profiling]\n'
        f'num_eval_edges: {total_eval_edges}\n'
        f'num_batches: {steps}\n'
        f'batch_size: {batch_size}\n'
        f'params_m: {params_m:.6f}\n'
        f'time_s_full_eval_mean: {time_s_full_eval_mean:.6f}\n'
        f'time_s_full_eval_std: {time_s_full_eval_std:.6f}\n'
        f'time_ms_per_batch: {time_ms_per_batch:.6f}\n'
        f'time_s_per_1k_edges: {time_s_per_1k_edges:.6f}\n'
        f'peak_mem_gb: {peak_mem_gb:.6f}\n'
    )

    csv_path = os.path.join(save_path, 'efficiency_profile.csv')
    with open(csv_path, 'w') as f:
        f.write(
            'num_eval_edges,num_batches,batch_size,params_m,'
            'time_s_full_eval_mean,time_s_full_eval_std,'
            'time_ms_per_batch,time_s_per_1k_edges,peak_mem_gb\n'
        )
        f.write(
            f'{total_eval_edges},{steps},{batch_size},{params_m:.6f},'
            f'{time_s_full_eval_mean:.6f},{time_s_full_eval_std:.6f},'
            f'{time_ms_per_batch:.6f},{time_s_per_1k_edges:.6f},{peak_mem_gb:.6f}\n'
        )

    return {
        'num_eval_edges': total_eval_edges,
        'num_batches': steps,
        'batch_size': batch_size,
        'params_m': params_m,
        'time_s_full_eval_mean': time_s_full_eval_mean,
        'time_s_full_eval_std': time_s_full_eval_std,
        'time_ms_per_batch': time_ms_per_batch,
        'time_s_per_1k_edges': time_s_per_1k_edges,
        'peak_mem_gb': peak_mem_gb,
    }
