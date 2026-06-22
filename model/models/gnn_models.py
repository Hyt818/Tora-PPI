'''
Purpose: To define the core network structure of the Tora-PPI full model, including residue graph encoding, PPI graph propagation, sequence/structure fusion, and multi-label interaction prediction.
Input: Residue-level features and edges, sequence-level features, training propagation edges, edges to be predicted for PPI, and subgraph structure.
Output: Multi-label PPI logits; additional intermediate representations are output during training for the purpose of calculating the alignment loss.
'''

# gnn_models_sag_ppr_subgraph.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, JumpingKnowledge, GCNConv
from torch_geometric.nn import global_mean_pool

# subgraph kernel
from model.models.subgraph_model import SubgraphGNNKernel


# ---------------------------
# Global branch: node encoder
# ---------------------------
class GIN_JK_Encoder(nn.Module):
    """
    Input:
        x: [N, in_dim]
        edge_index: [2, E_prop]

    Output:
        H_global: [N, hidden]
    """
    def __init__(self, in_dim=768, hidden=1024, train_eps=True):
        super().__init__()

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ),
            train_eps=train_eps
        )

        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ),
            train_eps=train_eps
        )

        self.jk = JumpingKnowledge(mode='cat')
        self.lin = nn.Linear(2 * hidden, hidden)

    def forward(self, x, edge_index, p=0.5):
        h1 = self.gin_conv1(x, edge_index)
        h2 = self.gin_conv2(h1, edge_index)

        h = self.jk([h1, h2])
        h = F.relu(self.lin(h))
        h = F.dropout(h, p=p, training=self.training)

        return h


# ---------------------------
# BGNN branch: residue graph encoder
# ---------------------------
class GCN(nn.Module):
    """
    Residue-level GCN on the merged residue graph.

    Input:
        x: [N_res, 320]
        edge_index: [2, E_res]
        edge_attr: [E_res]
        batch: [N_res]

    Output:
        protein-level structural embedding: [N_protein, 320]
    """
    def __init__(self):
        super().__init__()

        hidden = 320

        self.conv1 = GCNConv(320, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)

        return global_mean_pool(x, batch)


# ---------------------------
# Fusion: structure(320) + sequence(320) -> 768
# ---------------------------
class Fusion(nn.Module):
    """
    Fuse protein-level structural representation and sequence representation.

    Input:
        gnn_features: [N, 320]
        p_seq_all: [N, 320]

    Output:
        X_fusion: [N, 768]
    """
    def __init__(self):
        super().__init__()

        self.gnn_transform = nn.Linear(320, 256)
        self.esm_transform = nn.Linear(320, 256)
        self.transform = nn.Linear(512, 768)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, gnn_features, p_seq_all):
        gnn_transformed = F.relu(self.gnn_transform(gnn_features))
        esm_transformed = F.relu(self.esm_transform(p_seq_all))

        cross_feature = torch.cat([gnn_transformed, esm_transformed], dim=-1)

        combined = F.relu(self.transform(cross_feature))
        merged_features = self.bn(combined)

        return merged_features


# ---------------------------
# Main model
# ---------------------------
class Tora_PPI(nn.Module):
    def __init__(self, hidden=1024, class_num=7):
        super().__init__()

        self.BGNN = GCN()

        self.fusion = Fusion()

        self.to_sub = nn.Linear(768, 1024)

        self.subgraph_layers = SubgraphGNNKernel(
            nin=1024,
            nout=768,
            nlayer=1,
            gnn_types='GINConv',
            dropout=0.2,
            hop_dim=16,
            bias=False,
            res=True,
            pooling='mean',
            embs=(0, 1),
            embs_combine_mode='add',
            mlp_layers=1,
            subsampling=False,
            online=True,
            out_dim=768
        )

        self.sub_norm = nn.BatchNorm1d(768)

        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.global_encoder = GIN_JK_Encoder(
            in_dim=768,
            hidden=hidden,
            train_eps=True
        )

        self.fc_edge = nn.Linear(hidden, class_num)

        self.proj_gnn = nn.Linear(320, 128)
        self.proj_seq = nn.Linear(320, 128)

        self.proj_global = nn.Linear(hidden, 1024)
        self.proj_sub = nn.Linear(768, 1024)

    def forward(
        self,
        batch,
        p_x_all,
        p_edge_all,
        p_edge_arr,
        p_seq_all,
        prop_edge_index,
        pred_edge_index,
        graph,
        p=0.5,
        return_align=True
    ):
        dev = next(self.parameters()).device

        batch = batch.to(torch.int64).to(dev)

        x = p_x_all.to(torch.float32).to(dev)
        seq = p_seq_all.to(torch.float32).to(dev)

        edge = torch.as_tensor(
            p_edge_all,
            dtype=torch.long,
            device=dev
        )

        edge_arr = torch.as_tensor(
            p_edge_arr,
            dtype=torch.float32,
            device=dev
        )

        prop_edge_index = torch.as_tensor(
            prop_edge_index,
            dtype=torch.long,
            device=dev
        )

        pred_edge_index = torch.as_tensor(
            pred_edge_index,
            dtype=torch.long,
            device=dev
        )

        embs = self.BGNN(
            x,
            edge,
            edge_arr,
            batch - 1
        )

        X_fusion = self.fusion(
            embs,
            seq
        )

        graph = graph.to(dev)
        graph.x = self.to_sub(X_fusion)  

        H_sub = self.subgraph_layers(graph) 
        H_sub = self.sub_norm(H_sub)
        H_sub = F.relu(H_sub)
        H_sub = F.dropout(H_sub, p=p, training=self.training)

        gamma = torch.clamp(self.gamma, 0.0, 1.0)
        X_aug = X_fusion + gamma * H_sub

        H_global = self.global_encoder(
            X_aug,
            prop_edge_index,
            p=p
        ) 

        u, v = pred_edge_index[0], pred_edge_index[1]

        pair = H_global[u] * H_global[v] 
        final = self.fc_edge(pair)

        if return_align:
            z_gnn = F.normalize(
                self.proj_gnn(embs),
                dim=-1
            )

            z_seq = F.normalize(
                self.proj_seq(seq),
                dim=-1
            )

            g = F.normalize(
                self.proj_global(H_global),
                dim=-1
            )

            s = F.normalize(
                self.proj_sub(H_sub),
                dim=-1
            )

            return final, z_gnn, z_seq, g, s

        return final