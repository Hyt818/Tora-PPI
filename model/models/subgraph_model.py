'''
Purpose: Define the subgraph GNN kernel, which is used to encode the local subgraph structure extracted from the PPI training graph.
Input: Node features, subgraph node/edge mapping, hop indicator, and merged subgraph structure.
Output: Enhanced subgraph nodes or PPI graph representation.
'''

# subgraph_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn.inits import reset
from torch_scatter import scatter

from model.layers.elements import MLP, Identity


class GNN(nn.Module):
    """
    A clean GIN-based node encoder for combined subgraphs.
    - in_dim: input node feature dim (after concat hop embedding if use_hops)
    - hidden_dim: output hidden dim (we will set to 768)
    - nlayer: number of GIN layers
    """
    def __init__(self, in_dim, hidden_dim, nlayer=1,
                 dropout=0.0, bn=True, bias=True, res=True):
        super().__init__()
        assert nlayer >= 1

        self.dropout = dropout
        self.res = res
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.proj_res = nn.ModuleList()

        for layer in range(nlayer):
            layer_in = in_dim if layer == 0 else hidden_dim

            mlp = nn.Sequential(
                nn.Linear(layer_in, hidden_dim, bias=bias),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=bias),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim) if bn else Identity())

            # residual projection if needed
            if res and layer_in != hidden_dim:
                self.proj_res.append(nn.Linear(layer_in, hidden_dim, bias=False))
            else:
                self.proj_res.append(Identity())

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        for pr in self.proj_res:
            if isinstance(pr, nn.Linear):
                pr.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        for conv, norm, pr in zip(self.convs, self.norms, self.proj_res):
            prev = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + pr(prev)
        return x


class SubgraphGNNKernel(nn.Module):
    def __init__(self, nin, nout, nlayer, gnn_types,
                 dropout=0.0,
                 hop_dim=16,
                 bias=True,
                 res=True,
                 pooling='mean',
                 embs=(0, 1),                 
                 embs_combine_mode='add',     
                 mlp_layers=1,
                 subsampling=False,
                 online=True,
                 out_dim=768                  
                 ):
        super().__init__()
        assert max(embs) <= 2 and min(embs) >= 0
        assert embs_combine_mode in ['add', 'concat']
        assert pooling in ['mean', 'sum', 'max']

        self.use_hops = hop_dim > 0
        self.hop_dim = hop_dim
        self.pooling = pooling
        self.embs = embs
        self.embs_combine_mode = embs_combine_mode
        self.dropout = dropout
        self.subsampling = subsampling
        self.online = online

        self.hop_embedder = nn.Embedding(20, hop_dim) if self.use_hops else None

        gnn_in_dim = nin + hop_dim if self.use_hops else nin

        self.gnns = GNN(
            in_dim=gnn_in_dim,
            hidden_dim=out_dim,
            nlayer=nlayer,
            dropout=dropout,
            bn=True,
            bias=bias,
            res=res
        )

        self.subgraph_transform = MLP(out_dim, out_dim, nlayer=mlp_layers, with_final_activation=True)
        self.context_transform = MLP(out_dim, out_dim, nlayer=mlp_layers, with_final_activation=True)

        self.out_encoder = MLP(out_dim if embs_combine_mode == 'add' else out_dim * len(embs),
                               out_dim,
                               nlayer=mlp_layers,
                               with_final_activation=False,
                               bias=bias,
                               with_norm=True)

        if self.use_hops:
            self.gate_mapper_subgraph = nn.Sequential(nn.Linear(hop_dim, out_dim), nn.Sigmoid())
            self.gate_mapper_context = nn.Sequential(nn.Linear(hop_dim, out_dim), nn.Sigmoid())
            self.gate_mapper_centroid = nn.Sequential(nn.Linear(hop_dim, out_dim), nn.Sigmoid())
        else:
            self.gate_mapper_subgraph = None
            self.gate_mapper_context = None
            self.gate_mapper_centroid = None

        if nout != out_dim:
            raise ValueError(f"SubgraphGNNKernel: nout({nout}) must equal out_dim({out_dim}) in scheme-2.")

    def reset_parameters(self):
        if self.hop_embedder is not None:
            self.hop_embedder.reset_parameters()
        self.gnns.reset_parameters()
        self.subgraph_transform.reset_parameters()
        self.context_transform.reset_parameters()
        self.out_encoder.reset_parameters()
        if self.use_hops:
            reset(self.gate_mapper_context)
            reset(self.gate_mapper_subgraph)
            reset(self.gate_mapper_centroid)

    def forward(self, data):
        x = data.x[data.subgraphs_nodes_mapper]              
        edge_index = data.combined_subgraphs                
        sub_batch = data.subgraphs_batch                     

        if self.use_hops:
            hop_emb = self.hop_embedder(data.hop_indicator + 1)  
            x = torch.cat([x, hop_emb], dim=-1)                

        x = self.gnns(x, edge_index, sub_batch)            
        centroid_x = x[(data.subgraphs_nodes_mapper == sub_batch)] 


        if len(self.embs) > 1:
            subgraph_feat = self.subgraph_transform(F.dropout(x, self.dropout, training=self.training))
            context_feat = self.context_transform(F.dropout(x, self.dropout, training=self.training))
        else:
            subgraph_feat = x
            context_feat = x


        if self.use_hops:
            centroid_x = centroid_x * self.gate_mapper_centroid(hop_emb[(data.subgraphs_nodes_mapper == sub_batch)])
            subgraph_feat = subgraph_feat * self.gate_mapper_subgraph(hop_emb)
            context_feat = context_feat * self.gate_mapper_context(hop_emb)

        subgraph_x = scatter(subgraph_feat, sub_batch, dim=0, reduce=self.pooling)  # [N_sub, out_dim]
        context_x  = scatter(context_feat, data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)  # [N_node, out_dim]

        xs = [centroid_x, subgraph_x, context_x]
        xs = [xs[i] for i in self.embs]

        if self.embs_combine_mode == 'add':
            out = xs[0]
            for t in xs[1:]:
                out = out + t
        else:
            out = torch.cat(xs, dim=-1)

        return out
