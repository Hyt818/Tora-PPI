'''
Purpose: To provide an auxiliary function for data stitching before training, which combines multiple protein residue graphs into a single large graph and counts the number of model parameters.
Input: Residue features of each protein, PPR edge indices, edge weights, and residue quantities.
Output: Combined node features, edge indices, edge weights, batch indices, and parameter quantity statistics.
'''

import numpy as np
import torch
from tqdm import tqdm

def multi2big_x(x_ori):
    """
    Merge residue-level features of all proteins into a single big graph.

    Returns:
        x_cat: [N_res_total, 320]
        x_num_index: [N_protein], number of residues for each protein
    """
    protein_num = len(x_ori)
    x_cat = torch.zeros(1, 320)
    x_num_index = torch.zeros(protein_num)

    for i in tqdm(range(protein_num), desc='Processing node features', unit='graph'):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), dim=0)

    return x_cat[1:, :], x_num_index


def multi2big_batch(x_num_index):
    """
    Construct residue-to-protein batch indices.
    """
    protein_num = len(x_num_index)
    num_sum = x_num_index.sum().int()
    batch = torch.zeros(num_sum)

    count = 1
    for i in tqdm(range(1, protein_num), desc='Creating batch indices', unit='graph'):
        start = x_num_index[:i].sum().int()
        end = start + x_num_index[i].int()
        batch[start:end] = count * torch.ones(x_num_index[i].int())
        count += 1

    return batch.int()


def multi2big_edge(edge_ori, edge_arr_ori, num_index):
    """
    Merge residue-level PPR graphs of all proteins into a single big graph.

    Returns:
        edge_cat: [2, E_res_total]
        edge_arr_cat: [E_res_total]
        edge_num_index: [N_protein]
    """
    protein_num = len(edge_ori)
    edge_cat = torch.zeros(2, 1)
    edge_arr_cat = torch.zeros(1)
    edge_num_index = torch.zeros(protein_num)

    for i in tqdm(range(protein_num), desc='Processing edges', unit='graph'):
        edge_index_p = np.asarray(edge_ori[i])
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))

        edge_attr_i = torch.tensor(edge_arr_ori[i])

        if i == 0:
            offset = 0
        else:
            offset = torch.tensor(num_index[:i]).sum()

        edge_cat = torch.cat((edge_cat, edge_index_p + offset), dim=1)
        edge_arr_cat = torch.cat((edge_arr_cat, edge_attr_i), dim=0)

    return edge_cat[:, 1:], edge_arr_cat[1:], edge_num_index


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
