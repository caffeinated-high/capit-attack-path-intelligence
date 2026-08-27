"""
GCN / GAT / GraphSAGE implemented directly in PyTorch (no PyTorch
Geometric / DGL dependency) using SPARSE, edge-list-based computation.

============================================================================
WHY SPARSE, NOT DENSE ADJACENCY (read this if you hit a slow/frozen run)
============================================================================
An earlier version of this file built a dense n x n adjacency matrix and
did A_norm @ X for GCN, and a dense n x n x heads attention tensor for
GAT. That is fine on the ~70-node synthetic demo graph, but on a real
dataset (e.g. a LANL window with ~9,000 nodes) the GAT attention tensor
alone becomes roughly n * n * heads = 9000 * 9000 * 2 ~= 160 million
elements -- and PyTorch has to keep that (plus its backward-pass
buffers) in memory across 200 training epochs x 3 models. On a laptop
with limited RAM, that looks exactly like the program "getting stuck" --
it isn't actually frozen, it's just extremely slow / thrashing memory.

The fix below computes every layer using ONLY the graph's actual edges
(via `edge_index`, a [2, E] tensor of (source, destination) pairs) --
this is the standard, scalable way real GNN libraries implement these
layers, and its cost scales with the number of EDGES, not nodes^2. It
runs correctly on both the 70-node synthetic demo and a several-thousand
-node real graph without changing any calling code outside this file.

Task: transductive node classification over the CAPG -- predict which
nodes (hosts + identities) are "compromised" given the graph structure
and simple hand-engineered node features. Ground truth labels come from
the dataset's known-compromised set (the synthetic generator's embedded
campaign, or a real dataset's ground-truth file like LANL's
redteam.txt) and are used ONLY for supervised training/evaluation here,
never as an input feature to the graph itself.
"""
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

NODE_TYPES = ["host", "identity", "vuln", "mission_target"]


# ---------------------------------------------------------------------------
# Sparse aggregation primitives (replace dense matrix ops)
# ---------------------------------------------------------------------------
def scatter_add(src, index, dim_size):
    """out[i] = sum of src[j] for every j where index[j] == i.
    This is the sparse equivalent of `A @ X` -- aggregates per-edge
    messages into per-node sums using only the E actual edges, not an
    n x n matrix."""
    shape = (dim_size,) + src.shape[1:]
    out = src.new_zeros(shape)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.index_add_(0, index, src)
    return out


def scatter_softmax(scores, index, dim_size):
    """Softmax of `scores` grouped by `index` (e.g. softmax over each
    node's incoming edges). Numerically-stable via a per-group max
    subtraction using Tensor.scatter_reduce_ (vectorized, available in
    torch>=1.12 -- well within this project's torch>=2.0 requirement),
    so this stays O(E) even on graphs with hundreds of thousands of
    edges, unlike a Python-loop fallback."""
    group_max = scores.new_full((dim_size,) + scores.shape[1:], float("-inf"))
    idx_expanded = index.view(-1, *([1] * (scores.dim() - 1))).expand_as(scores)
    group_max.scatter_reduce_(0, idx_expanded, scores, reduce="amax", include_self=True)
    shifted = scores - group_max[index]
    exp = shifted.exp()
    denom = scatter_add(exp, index, dim_size).clamp(min=1e-12)
    return exp / denom[index]


# ---------------------------------------------------------------------------
# Feature engineering (Data Processing Layer -> Graph Learning Layer input)
# ---------------------------------------------------------------------------
def build_features_and_labels(capg_builder, ground_truth_compromised: set):
    """
    Returns:
      X            : [n, feat_dim] node feature matrix
      y            : [n] binary compromised label (evaluation only)
      edge_index   : [2, E] tensor of (src, dst) node indices, symmetric
                     (undirected) with self-loops added -- this replaces
                     the old dense adjacency matrix
      edge_weight  : [E] symmetric-normalized GCN edge weight
                     (1/sqrt(deg_src * deg_dst)), precomputed once
      nodes, idx   : node id list / id->index map
    """
    g = capg_builder.g
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    feat_dim = len(NODE_TYPES) + 5
    X = np.zeros((n, feat_dim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)

    for node, data in g.nodes(data=True):
        i = idx[node]
        t_onehot = [1.0 if data["type"] == t else 0.0 for t in NODE_TYPES]
        in_deg = g.in_degree(node)
        out_deg = g.out_degree(node)
        n_escalate = sum(1 for _, _, d in g.out_edges(node, data=True) if d["edge_type"] == "escalate")
        is_vuln_host = 1.0 if node in config.VULNERABLE_HOSTS else 0.0
        is_mission = 1.0 if node in config.HIGH_VALUE_HOSTS else 0.0
        X[i] = t_onehot + [in_deg, out_deg, n_escalate, is_vuln_host, is_mission]
        y[i] = 1.0 if node in ground_truth_compromised else 0.0

    # normalize the numeric (non-onehot) columns
    numeric = X[:, len(NODE_TYPES):]
    std = numeric.std(0)
    std[std == 0] = 1.0
    numeric = (numeric - numeric.mean(0)) / std
    X[:, len(NODE_TYPES):] = numeric

    # ---- sparse edge list (undirected + self-loops), built ONCE here --
    g_undirected = g.to_undirected()
    src_list, dst_list = [], []
    for u, v in g_undirected.edges():
        ui, vi = idx[u], idx[v]
        src_list.append(ui); dst_list.append(vi)
        src_list.append(vi); dst_list.append(ui)   # both directions
    for i in range(n):                              # self-loops
        src_list.append(i); dst_list.append(i)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    deg = torch.zeros(n)
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
    deg = deg.clamp(min=1.0)
    edge_weight = 1.0 / (deg[edge_index[0]].sqrt() * deg[edge_index[1]].sqrt())

    return (
        torch.tensor(X),
        torch.tensor(y),
        edge_index,
        edge_weight,
        nodes,
        idx,
    )


# ---------------------------------------------------------------------------
# GCN layer (sparse): out[i] = sum_j in N(i) norm_ij * W * x_j
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X, edge_index, edge_weight, n_nodes):
        h = self.lin(X)
        src, dst = edge_index[0], edge_index[1]
        messages = h[src] * edge_weight.unsqueeze(-1)
        return scatter_add(messages, dst, n_nodes)


# ---------------------------------------------------------------------------
# GAT layer (sparse): attention computed only over real edges, not n x n
# ---------------------------------------------------------------------------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, X, edge_index, edge_weight, n_nodes):
        src, dst = edge_index[0], edge_index[1]
        h = self.lin(X).view(n_nodes, self.n_heads, self.out_dim)        # [n, heads, out]

        h_src = h[src]                                                   # [E, heads, out]
        h_dst = h[dst]
        e_src = (h_src * self.attn_src).sum(-1)                          # [E, heads]
        e_dst = (h_dst * self.attn_dst).sum(-1)                          # [E, heads]
        e = self.leaky(e_src + e_dst)                                    # [E, heads]  -- O(E), not O(n^2)

        alpha = scatter_softmax(e, dst, n_nodes)                         # softmax over each node's incoming edges
        messages = h_src * alpha.unsqueeze(-1)                           # [E, heads, out]
        out = scatter_add(messages, dst, n_nodes)                        # [n, heads, out]
        return out.reshape(n_nodes, self.n_heads * self.out_dim)


# ---------------------------------------------------------------------------
# GraphSAGE layer (sparse): mean-aggregate neighbors via scatter_add
# ---------------------------------------------------------------------------
class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, X, edge_index, edge_weight, n_nodes):
        src, dst = edge_index[0], edge_index[1]
        summed = scatter_add(X[src], dst, n_nodes)
        deg = scatter_add(torch.ones(src.size(0), 1), dst, n_nodes).clamp(min=1.0)
        neighbor_mean = summed / deg
        combined = torch.cat([X, neighbor_mean], dim=1)
        return self.lin(combined)


# ---------------------------------------------------------------------------
# Generic 2-layer node classifier wrapping any of the layer types above
# ---------------------------------------------------------------------------
class NodeClassifier(nn.Module):
    def __init__(self, layer_cls, in_dim, hidden_dim, **layer_kwargs):
        super().__init__()
        self.l1 = layer_cls(in_dim, hidden_dim, **layer_kwargs)
        l1_out_dim = hidden_dim * layer_kwargs.get("n_heads", 1) if layer_cls is GATLayer else hidden_dim
        self.l2 = layer_cls(l1_out_dim, hidden_dim, **({"n_heads": 1} if layer_cls is GATLayer else {}))
        l2_out_dim = hidden_dim
        self.out = nn.Linear(l2_out_dim, 1)

    def forward(self, X, edge_index, edge_weight, n_nodes):
        h = F.relu(self.l1(X, edge_index, edge_weight, n_nodes))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.l2(h, edge_index, edge_weight, n_nodes))
        return self.out(h).squeeze(-1)
