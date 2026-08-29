"""
Standalone accuracy report for the three GNN models (GCN, GAT, GraphSAGE).

Runs Stage 4's model comparison (models/train.py) on its own -- without
running the full 8-stage pipeline -- and writes a clean, readable report
into outputs/, plus prints it to the console. Useful when you just want
the numbers (e.g. for a report or a viva) without re-running everything
else.

Usage:
    python model_accuracy_report.py                       # synthetic demo data
    python model_accuracy_report.py --dataset lanl \\
        --path auth.txt.gz --path2 redteam.txt.gz \\
        --time-window 3600 --max-rows 200000                # real LANL data
"""
import argparse
import os
from datetime import datetime, timezone

from capg.graph import CAPGBuilder
from models.train import compare_models

# Same outputs/ folder every other part of the pipeline writes into
# (see pipeline_core.py's OUTPUT_DIR) -- keeps every generated artifact
# in one place instead of scattering .txt files into the project root.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def load_synthetic():
    from data.generate_synthetic_logs import generate
    from models.train import ground_truth_compromised_set
    events_df = generate()
    return events_df, ground_truth_compromised_set(events_df)


def load_lanl(args):
    from data import lanl_loader
    if args.path:
        lanl_loader.LANL_AUTH_PATH = args.path
    if args.path2:
        lanl_loader.LANL_REDTEAM_PATH = args.path2
    redteam_df = lanl_loader.load_redteam()
    lanl_loader.configure_from_redteam(redteam_df, high_value_top_n=5)
    events_df = lanl_loader.load_auth_window(
        time_window_seconds=args.time_window, start_time=args.start_time, max_rows=args.max_rows)
    events_df = lanl_loader.attach_labels(events_df, redteam_df)
    return events_df, lanl_loader.redteam_ground_truth_set(redteam_df)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "lanl"])
    p.add_argument("--path", default=None)
    p.add_argument("--path2", default=None)
    p.add_argument("--time-window", type=int, default=3600)
    p.add_argument("--start-time", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument("--output", default=None,
                   help="Output filename (not full path -- always saved inside outputs/). "
                        "Defaults to model_accuracy_report_<dataset>_<timestamp>.txt")
    return p.parse_args()


def format_report(dataset_name, n_nodes, n_ground_truth, results, node_scores, nodes):
    lines = []
    lines.append("=" * 78)
    lines.append("GNN MODEL ACCURACY REPORT")
    lines.append("=" * 78)
    lines.append(f"Dataset:              {dataset_name}")
    lines.append(f"Generated:            {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Graph size:           {n_nodes} nodes")
    lines.append(f"Ground-truth positives: {n_ground_truth}")
    lines.append("")
    lines.append(f"{'Model':<12}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}   TP/FP/FN/TN")
    lines.append("-" * 78)
    for name, m in results.items():
        lines.append(
            f"{name:<12}{m['accuracy']:>10.2%}{m['precision']:>11.2%}"
            f"{m['recall']:>9.2%}{m['f1']:>8.2%}   {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}"
        )
    lines.append("")
    lines.append("Top-5 highest-risk nodes per model (node, risk score 0-1):")
    for name in node_scores:
        import numpy as np
        order = np.argsort(-node_scores[name])[:5]
        ranked = [(nodes[i], round(float(node_scores[name][i]), 3)) for i in order]
        lines.append(f"  {name:<10}: {ranked}")
    lines.append("")
    lines.append("Note: with a small number of ground-truth positives, held-out")
    lines.append("precision/recall can be noisy (a single miss swings the number a lot).")
    lines.append("The top-5 risk ranking above is the more meaningful signal at small scale.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    args = parse_args()

    if args.dataset == "synthetic":
        events_df, ground_truth = load_synthetic()
    else:
        events_df, ground_truth = load_lanl(args)

    builder = CAPGBuilder()
    builder.ingest_events(events_df)
    ground_truth_in_graph = set(ground_truth) & set(builder.g.nodes())

    results, node_scores, nodes = compare_models(builder, ground_truth_in_graph, verbose=True)

    report = format_report(args.dataset, builder.g.number_of_nodes(),
                            len(ground_truth_in_graph), results, node_scores, nodes)
    print("\n" + report)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = args.output
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_accuracy_report_{args.dataset}_{timestamp}.txt"
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()