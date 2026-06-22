"""
用途: Tora-PPI full model 的训练入口脚本，负责读取命令行/YAML 配置、加载数据和特征、构建模型并启动训练。
输入: configs/*.yaml 或命令行参数；PPI 文件、蛋白序列文件、预提取特征、PPR 边文件和 split JSON。
输出: 训练日志、TensorBoard 日志、模型 checkpoint，以及可选的效率评估文件。
"""

import argparse
import os
import re
import time


import numpy as np
import torch
import torch.nn as nn

try:
    import yaml
except ImportError:
    yaml = None

"""
Tora-PPI training script

Key leakage-control design:
1. Ego-subgraphs are extracted only from training PPI edges.
2. Global PPI message passing uses only training PPI edges: graph.edge_index_got.
3. Validation/test edges are used only as prediction targets and are excluded from message passing.
4. The model forward separates:
   - prop_edge_index: train-only propagation graph
   - pred_edge_index: target edges for the current batch
"""




def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")



def validate_split_seed(args):
    """Prevent accidentally evaluating one seed with another seed's split file."""
    if args.split_path is None:
        return
    match = re.search(r"seed(\d+)\.json$", args.split_path)
    if match is None:
        return
    split_seed = int(match.group(1))
    if split_seed != int(args.seed_num):
        raise ValueError(
            "split_path seed does not match seed_num: "
            f"seed_num={args.seed_num}, split_path={args.split_path}"
        )

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(description='Tora-PPI model training')
parser.add_argument('--config', default=None, type=str,
                    help='YAML config path')

parser.add_argument('--ppi_path', default='./data/SHS27k/protein.actions.SHS27k.txt', type=str,
                    help='PPI path')
parser.add_argument('--pseq_path', default='./data/SHS27k/protein.SHS27k.sequences.dictionary.tsv', type=str,
                    help='Protein sequence path')
parser.add_argument('--vec_path', default='./data/SHS27k/vec5_CTC.txt', type=str,
                    help='Protein sequence vector path')

parser.add_argument('--p_feat_matrix', default='./data/SHS27k/FSP_residue_Embedding.pt', type=str,
                    help='Residue-level feature matrix')
parser.add_argument('--p_sequence_feat', default='./data/SHS27k/FSP_sequence_Embedding.pt', type=str,
                    help='Protein-level sequence feature matrix')
parser.add_argument('--p_adj_matrix', default='./data/SHS27k/edge_index_ppr.npy', type=str,
                    help='PPR-enhanced residue graph edge index')
parser.add_argument('--p_adj_arr', default='./data/SHS27k/edge_attr_ppr.npy', type=str,
                    help='PPR-enhanced residue graph edge weights')

parser.add_argument('--split', default='random', type=str,
                    help='Split method: random, bfs, or dfs')

parser.add_argument('--split_path', default=None, type=str,
                    help='Explicit split JSON path. Overrides split/seed naming convention')
parser.add_argument('--save_path', default='./result', type=str,
                    help='Save folder')
parser.add_argument('--epoch_num', default=500, type=int,
                    help='Number of training epochs')

parser.add_argument('--batch_size', default=11000, type=int,
                    help='Training and validation batch size')
parser.add_argument('--learning_rate', default=0.00025, type=float,
                    help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Adam weight decay')
parser.add_argument('--scheduler_factor', default=0.5, type=float,
                    help='ReduceLROnPlateau factor')
parser.add_argument('--scheduler_patience', default=10, type=int,
                    help='ReduceLROnPlateau patience')

parser.add_argument('--seed_num', default=1, type=int,
                    help='Random seed')
parser.add_argument('--run_id', default=1, type=int,
                    help='Run id')
parser.add_argument('--cuda', default=0, type=int,
                    help='CUDA device id, e.g. 0 or 1. Use -1 for CPU')

parser.add_argument('--profile_enabled', default=False, type=str2bool,
                    help='Whether to run inference efficiency profiling after training')
parser.add_argument('--profile_batch_size', default=11000, type=int,
                    help='Profiling batch size')
parser.add_argument('--profile_warmup_steps', default=5, type=int,
                    help='Profiling warmup steps')
parser.add_argument('--profile_repeat', default=5, type=int,
                    help='Profiling repeat count')
parser.add_argument('--profile_use_best_ckpt', default=True, type=str2bool,
                    help='Whether profiling loads gnn_model_valid_best.ckpt')




def _flatten_config(config):
    """Flatten supported YAML sections into argparse-style keys."""
    values = {}
    for section in ("experiment", "paths", "training", "profiling"):
        section_values = config.get(section, {}) or {}
        for key, value in section_values.items():
            if section == "profiling":
                values[f"profile_{key}"] = value
            else:
                values[key] = value
    return values


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, type=str,
                               help="Path to YAML config file")
    config_args, remaining = config_parser.parse_known_args()

    if config_args.config is not None:
        if yaml is None:
            raise ImportError("PyYAML is required when using --config")
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        defaults = _flatten_config(config)
        parser.set_defaults(**defaults)

    args = parser.parse_args(remaining)
    args.config = config_args.config
    validate_split_seed(args)
    return args

# ============================================================
# Main
# ============================================================

def main(args):
    from model.data.gnn_data import GNN_DATA
    from model.models.gnn_models import Tora_PPI
    from model.training.helpers import multi2big_batch, multi2big_edge, multi2big_x
    from model.training.profiler import profile_inference_efficiency
    from model.training.trainer import train


    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    ppi_data.get_feature_origin(
        pseq_path=args.pseq_path,
        vec_path=args.vec_path
    )

    ppi_data.generate_data()

    if args.split_path is not None:
        train_valid_index_path = args.split_path
    else:
        train_valid_index_path = os.path.join('split_data', f'{args.split}')
        train_valid_index_path = os.path.join(
            train_valid_index_path,
            'SHS27k_split_{}_seed{}.json'.format(args.split, args.seed_num)
        )

    ppi_data.split_dataset(
        train_valid_index_path,
        random_new=False
    )

    graph = ppi_data.data
    ppi_list = ppi_data.ppi_list

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['test_index']

    # ========================================================
    # Build train-only ego-subgraph cache
    # ========================================================

    from model.graph.subgraph import SubgraphsData, extract_subgraphs, to_sparse, combine_subgraphs

    graph = SubgraphsData(**{k: v for k, v in graph})

    # Train-only PPI edges for subgraph extraction and propagation.
    train_e = graph.edge_index[:, graph.train_mask]
    train_e = torch.cat([train_e, train_e[[1, 0]]], dim=1)

    subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense = extract_subgraphs(
        train_e,
        graph.num_nodes,
        num_hops=1,
        walk_length=0,
        p=1,
        q=1,
        repeat=1
    )

    subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(
        subgraphs_nodes_mask,
        subgraphs_edges_mask,
        hop_indicator_dense
    )

    combined_subgraphs = combine_subgraphs(
        train_e,
        subgraphs_nodes,
        subgraphs_edges,
        num_selected=graph.num_nodes,
        num_nodes=graph.num_nodes
    )

    graph.subgraphs_batch = subgraphs_nodes[0]
    graph.subgraphs_nodes_mapper = subgraphs_nodes[1]
    graph.subgraphs_edges_mapper = subgraphs_edges[1]
    graph.combined_subgraphs = combined_subgraphs
    graph.hop_indicator = hop_indicator
    graph.__num_nodes__ = graph.num_nodes

    # ========================================================
    # Load residue-level and sequence-level features
    # ========================================================

    p_x_all = torch.load(args.p_feat_matrix)
    p_edge_all = np.load(args.p_adj_matrix, allow_pickle=True)
    p_edge_arr = np.load(args.p_adj_arr, allow_pickle=True)

    p_x_all, x_num_index = multi2big_x(p_x_all)
    p_edge_all, p_edge_arr, edge_num_index = multi2big_edge(
        p_edge_all,
        p_edge_arr,
        x_num_index
    )

    batch = multi2big_batch(x_num_index) + 1

    p_seq_all = torch.load(args.p_sequence_feat)
    if not torch.is_tensor(p_seq_all):
        p_seq_all = np.asarray(p_seq_all)
    p_seq_all = torch.as_tensor(p_seq_all)

    print(
        'train gnn, train_num: {}, valid_num: {}'.format(
            len(graph.train_mask),
            len(graph.val_mask)
        )
    )

    # ========================================================
    # Train-only propagation graph
    # ========================================================

    graph.edge_index_got = torch.cat(
        (
            graph.edge_index[:, graph.train_mask],
            graph.edge_index[:, graph.train_mask][[1, 0]]
        ),
        dim=1
    )

    graph.edge_attr_got = torch.cat(
        (
            graph.edge_attr_1[graph.train_mask],
            graph.edge_attr_1[graph.train_mask]
        ),
        dim=0
    )

    graph.train_mask_got = [i for i in range(len(graph.train_mask))]

    # ========================================================
    # Device
    # ========================================================

    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    graph.to(device)

    # ========================================================
    # Model and optimizer
    # ========================================================

    model = Tora_PPI()
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience
    )

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    # ========================================================
    # Save paths
    # ========================================================

    save_path = args.save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    time_stamp = time.strftime('%Y-%m-%d %H-%M-%S')

    save_path = os.path.join(save_path, 'SHS27k')
    save_path = os.path.join(save_path, f'{args.split}')
    save_path = os.path.join(
        save_path,
        'seed_{}_run_id_{}_{}'.format(
            args.seed_num,
            args.run_id,
            time_stamp
        )
    )

    os.makedirs(save_path, exist_ok=True)

    result_file_path = os.path.join(save_path, 'valid_results.txt')
    config_path = os.path.join(save_path, 'config.txt')

    with open(config_path, 'w') as f:
        f.write('split: {}\n'.format(args.split))
        f.write('seed_num: {}\n'.format(args.seed_num))
        f.write('run_id: {}\n'.format(args.run_id))
        f.write('ppi_path: {}\n'.format(args.ppi_path))
        f.write('pseq_path: {}\n'.format(args.pseq_path))
        f.write('p_feat_matrix: {}\n'.format(args.p_feat_matrix))
        f.write('p_sequence_feat: {}\n'.format(args.p_sequence_feat))
        f.write('p_adj_matrix: {}\n'.format(args.p_adj_matrix))
        f.write('p_adj_arr: {}\n'.format(args.p_adj_arr))
        f.write('config: {}\n'.format(args.config))
        f.write('split_path: {}\n'.format(train_valid_index_path))
        f.write('learning_rate: {}\n'.format(args.learning_rate))
        f.write('weight_decay: {}\n'.format(args.weight_decay))
        f.write('batch_size: {}\n'.format(args.batch_size))
        f.write('max_epochs: {}\n'.format(args.epoch_num))
        f.write('propagation_graph: train-only edges\n')
        f.write('validation_edges: target-only, excluded from message passing\n')

    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        from torch.utils.tensorboard import SummaryWriter

    summary_writer = SummaryWriter(save_path)

    # ========================================================
    # Training
    # ========================================================

    train(
        batch=batch,
        p_x_all=p_x_all,
        p_edge_all=p_edge_all,
        p_edge_arr=p_edge_arr,
        p_seq_all=p_seq_all,
        model=model,
        graph=graph,
        ppi_list=ppi_list,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        result_file_path=result_file_path,
        summary_writer=summary_writer,
        save_path=save_path,
        batch_size=args.batch_size,
        epochs=args.epoch_num,
        scheduler=scheduler
    )

    # ========================================================
    # Inference efficiency profiling
    # ========================================================

    if args.profile_enabled:
        profile_inference_efficiency(
            batch=batch,
            p_x_all=p_x_all,
            p_edge_all=p_edge_all,
            p_edge_arr=p_edge_arr,
            p_seq_all=p_seq_all,
            model=model,
            graph=graph,
            device=device,
            save_path=save_path,
            batch_size=args.profile_batch_size,
            warmup_steps=args.profile_warmup_steps,
            repeat=args.profile_repeat,
            use_best_ckpt=args.profile_use_best_ckpt
        )

    summary_writer.close()


if __name__ == '__main__':
    args = parse_args()
    from model.utils.utils import set_seed
    set_seed(args.seed_num)
    main(args)
