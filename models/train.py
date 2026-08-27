"""
Trains and compares GCN, GAT, and GraphSAGE on the compromised-node
classification task described in models/gnn.py. This is the Phase-1
proof-of-concept referenced in the handbook: "train a 2-layer GCN [and
friends] to classify compromised vs. benign accounts/hosts."

Uses the sparse, edge-list-based layers in models/gnn.py, so this scales
to real graphs with thousands of nodes -- not just the ~70-node
synthetic demo. See models/gnn.py's module docstring for why an earlier
dense-matrix version could look "stuck" on real data.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from capg.graph import CAPGBuilder
from models.gnn import (GCNLayer, GATLayer, SAGELayer, NodeClassifier,
                         build_features_and_labels)

torch.manual_seed(0)
np.random.seed(0)

# On a large real graph, training 3 models x 200 epochs each is still a
# lot of work even with sparse layers -- automatically scale epochs down
# as the graph grows, so a run on a several-thousand-node graph finishes
# in a reasonable time on a laptop CPU instead of taking many minutes per
# model. Tune these thresholds up if you're running on a machine with a
# GPU or want higher-fidelity training and don't mind waiting longer.
_EPOCH_SCALING = [
    (2000, 200),     # up to 2,000 nodes:  full 200 epochs
    (6000, 80),       # up to 6,000 nodes:  80 epochs
    (20000, 40),      # up to 20,000 nodes: 40 epochs
]
_DEFAULT_LARGE_GRAPH_EPOCHS = 20   # beyond 20,000 nodes


def _auto_epochs(n_nodes, requested_epochs):
    for max_nodes, epochs_cap in _EPOCH_SCALING:
        if n_nodes <= max_nodes:
            return min(requested_epochs, epochs_cap)
    return min(requested_epochs, _DEFAULT_LARGE_GRAPH_EPOCHS)


def compare_models(builder, compromised, hidden_dim=16, epochs=200, verbose=True):
    """
    Dataset-agnostic: takes an already-built CAPGBuilder and a
    ground-truth 'compromised node id' set (however you obtained it --
    the synthetic generator's embedded campaign, or a real dataset's
    ground-truth loader) and trains/compares GCN, GAT, GraphSAGE on it.
    """
    X, y, edge_index, edge_weight, nodes, idx = build_features_and_labels(builder, compromised)
    n_nodes = X.size(0)
    train_mask, test_mask = make_masks(n_nodes, y)

    effective_epochs = _auto_epochs(n_nodes, epochs)
    if effective_epochs < epochs and verbose:
        print(f"[models.train] Graph has {n_nodes} nodes -- auto-reducing epochs "
              f"from {epochs} to {effective_epochs} per model so this finishes in a "
              f"reasonable time. Override by calling compare_models(..., epochs=N) "
              f"directly if you want the full count and are willing to wait.")

    if verbose:
        print(f"Graph: {n_nodes} nodes | {int(y.sum().item())} labeled-compromised "
              f"| train={int(train_mask.sum())} test={int(test_mask.sum())}\n")

    configs = [
        ("GCN", GCNLayer, {}),
        ("GAT", GATLayer, {"n_heads": 2}),
        ("GraphSAGE", SAGELayer, {}),
    ]

    results = {}
    node_scores = {}
    for name, layer_cls, kwargs in configs:
        model = NodeClassifier(layer_cls, in_dim=X.size(1), hidden_dim=hidden_dim, **kwargs)
        metrics, probs = train_one(model, X, y, edge_index, edge_weight, n_nodes,
                                    train_mask, test_mask, epochs=effective_epochs,
                                    model_name=name, verbose=verbose)
        results[name] = metrics
        node_scores[name] = probs.detach().numpy()

    if verbose:
        print(f"{'Model':<12}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}   TP/FP/FN/TN")
        for name, m in results.items():
            print(f"{name:<12}{m['accuracy']:>8.2f}{m['precision']:>8.2f}"
                  f"{m['recall']:>8.2f}{m['f1']:>8.2f}   {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")

        print("\nTop-5 highest-risk nodes per model:")
        for name in node_scores:
            order = np.argsort(-node_scores[name])[:5]
            ranked = [(nodes[i], round(float(node_scores[name][i]), 3)) for i in order]
            print(f"  {name:<10}: {ranked}")

    return results, node_scores, nodes


def ground_truth_compromised_set(events_df):
    attack_rows = events_df[events_df.label == "attack"]
    compromised = set(attack_rows.src_host) | set(attack_rows.dst_host)
    compromised |= {f"id:{u}" for u in attack_rows.src_user.unique()}
    return compromised


def run_comparison():
    """Synthetic-demo convenience wrapper around compare_models(). For
    real data, build your own CAPGBuilder + compromised set (see
    run_pipeline.py) and call compare_models() directly instead."""
    from data.generate_synthetic_logs import generate

    events_df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(events_df)
    compromised = ground_truth_compromised_set(events_df)

    results, node_scores, nodes = compare_models(builder, compromised)
    return builder, events_df, compromised, node_scores, nodes


def train_one(model, X, y, edge_index, edge_weight, n_nodes, train_mask, test_mask,
              epochs=200, lr=0.02, model_name="model", verbose=True, progress_every=25):
    # class imbalance is severe (a handful of compromised nodes vs. many
    # benign ones) -- weight the positive class so the model doesn't
    # trivially predict "benign" for everything.
    n_pos = y[train_mask].sum().clamp(min=1)
    n_neg = (train_mask.sum() - n_pos).clamp(min=1)
    pos_weight = (n_neg / n_pos)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start = time.time()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X, edge_index, edge_weight, n_nodes)
        loss = loss_fn(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        # Print periodic progress on larger graphs so a slow run is
        # visibly making progress instead of looking frozen -- this is
        # exactly the situation that looked like a hang before the
        # sparse-layer rewrite.
        if verbose and n_nodes > 500 and (epoch + 1) % progress_every == 0:
            elapsed = time.time() - start
            print(f"  [{model_name}] epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}  "
                  f"({elapsed:.1f}s elapsed)")

    model.eval()
    with torch.no_grad():
        logits = model(X, edge_index, edge_weight, n_nodes)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    return evaluate(preds[test_mask], y[test_mask]), probs


def evaluate(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def make_masks(n, y, test_frac=0.3):
    """Stratified-ish split: guarantee at least one positive in each split
    even when positives are rare relative to the graph size."""
    pos_idx = np.where(y.numpy() == 1)[0]
    neg_idx = np.where(y.numpy() == 0)[0]
    rng = np.random.default_rng(0)
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)

    n_pos_test = max(1, int(len(pos_idx) * test_frac))
    n_neg_test = max(1, int(len(neg_idx) * test_frac))

    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])

    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


if __name__ == "__main__":
    run_comparison()
