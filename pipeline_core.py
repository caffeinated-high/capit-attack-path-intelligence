"""
Shared HCAPO pipeline logic (Stages 1/3 through 8), factored out so
every entry point (synthetic demo, LANL, OpTC, DARPA TC, or any of the
flow-based datasets) calls the exact same code once its data is loaded
and normalized to the common event schema + config.py is populated.

This is what "the same pipeline runs on every dataset" concretely means
in this codebase: everything below this line only ever sees the
canonical columns (t, src_user, src_host, dst_host, auth_type, success,
label, technique) and config.HIGH_VALUE_HOSTS / VULNERABLE_HOSTS /
HOST_SUBNET_MAP -- never a dataset-specific column name.

Every run also now produces, in outputs/<run_id>/:
  - a plain-text .log file (utils/logging_setup.py)
  - a static PNG of the CAPG (visualization/graph_viz.py)
  - an interactive standalone HTML graph (visualization/graph_viz.py)
  - a single self-contained dashboard.html tying everything together
    (visualization/dashboard.py) -- this is the file to open if you
    just want to SEE what happened, without reading code or logs.
"""
import os

from capg.graph import CAPGBuilder
from cas.cas_builder import build_cas
from defense.defense_planner import execute_simulated, plan_defense
from knowledge_evolution import apply_feedback
from models.train import compare_models
from optimization.hierarchical_optimizer import greedy_hierarchical_select
from reasoning.path_reasoning import most_likely_source, top_k_paths
from utils.logging_setup import get_logger
from visualization.dashboard import generate_dashboard
from visualization.graph_viz import export_graph_html, export_graph_png

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def run_all_stages(events_df, ground_truth_compromised, dataset_name="run",
                    k_paths=5, budget=5.0, min_ground_truth_for_gnn=2,
                    max_cas_to_print=15, make_dashboard=True, make_graph_exports=True):
    """
    Runs Stages 1/3 through 8 on an already-loaded, already-labeled
    events DataFrame, with config.py already populated by the caller's
    dataset-specific loader.

    ground_truth_compromised : set of node ids (e.g. {"C586", "id:U66"})
        used ONLY for the Stage-4 GNN evaluation step and for
        highlighting in the graph exports -- never fed into the graph
        as an input feature.

    Returns (builder, record): the CAPGBuilder and the completed
    utils.logging_setup.RunRecord (has .log_file_path, and if
    make_dashboard=True, the dashboard path is printed and returned as
    record.dashboard_path).
    """
    logger, record = get_logger(dataset_name=dataset_name)
    record.dashboard_path = ""

    if events_df.empty:
        logger.error("No events to process -- widen your dataset window/filters and retry.")
        record.finish()
        return None, record

    logger.stage("STAGE 1 + 3: CAPG Construction")
    builder = CAPGBuilder()
    builder.ingest_events(events_df)
    stats = builder.stats()
    record.capg_stats = {**stats, "events_ingested": len(events_df)}
    logger.info(f"Ingested {len(events_df)} events.")
    logger.info(f"CAPG stats: {stats}")

    logger.stage("STAGE 2: Cognitive Attack State (CAS) Construction")
    states = build_cas(events_df, builder)
    non_benign = {k: v for k, v in states.items() if v.intent != "reconnaissance-or-benign"}
    logger.info(f"{len(states)} identities profiled, {len(non_benign)} flagged with non-benign intent.")
    for cas in list(non_benign.values())[:max_cas_to_print]:
        logger.info("  " + cas.summary())
        record.cas_summaries.append(cas.summary())

    logger.stage("STAGE 4: Attack Intent Inference (CAS heuristic + learned GNN)")
    attacker_cas = most_likely_source(states)
    if attacker_cas is None:
        logger.warning("No identities found in this data slice.")
        record.finish()
        _finalize_outputs(record, builder, dataset_name, set(), make_dashboard, make_graph_exports, logger)
        return builder, record

    logger.info("Highest-priority identity per CAS heuristic:")
    logger.info("  " + attacker_cas.summary())
    record.attacker_identity = attacker_cas.identity

    ground_truth_in_graph = set(ground_truth_compromised) & set(builder.g.nodes())
    record.ground_truth_count = len(ground_truth_in_graph)
    logger.info(f"{len(ground_truth_in_graph)} ground-truth compromised nodes fall inside this graph.")
    if len(ground_truth_in_graph) >= min_ground_truth_for_gnn:
        n_graph_nodes = builder.g.number_of_nodes()
        if n_graph_nodes > 500:
            logger.info(f"Graph has {n_graph_nodes} nodes -- GNN training will take a "
                        f"little while (progress is printed every 25 epochs so this "
                        f"doesn't look stuck).")
        results, node_scores, nodes = compare_models(builder, ground_truth_in_graph)
        record.gnn_metrics = results
        record.gnn_top_risk = {
            name: [(nodes[i], round(float(node_scores[name][i]), 3))
                   for i in sorted(range(len(nodes)), key=lambda i: -node_scores[name][i])[:5]]
            for name in node_scores
        }
    else:
        logger.warning("Too few ground-truth positives in this slice to train/evaluate "
                        "the GNNs meaningfully -- widen the window / load more rows.")

    logger.stage("STAGE 5: Attack Path Reasoning")
    paths = top_k_paths(builder, attacker_cas.identity, attacker_cas.objectives, k=k_paths)
    if not paths:
        logger.warning("No path found from the flagged identity to any high-value host.")
    for path, cost, edge_types in paths:
        logger.info(f"  cost={cost:6.2f}  path={' -> '.join(path)}  via={edge_types}")
        record.candidate_paths.append({"path": path, "cost": cost, "via": edge_types})

    logger.stage("STAGE 6: Hierarchical Cognitive Attack Path Optimization")
    chosen, covered, all_idxs = greedy_hierarchical_select(paths, budget=budget)
    logger.info(f"Covered {len(covered)}/{len(all_idxs)} candidate paths within budget={budget}")
    for c in chosen:
        logger.info(f"  [{c.level.upper():7s}] {c.action}")
        record.interventions.append({"level": c.level, "action": c.action})

    logger.stage("STAGE 7: Autonomous Defense Planning (simulated execution)")
    actions = plan_defense(chosen)
    action_log = execute_simulated(actions, verbose=False)
    for a, line in zip(actions, action_log):
        logger.info(line)
    record.defense_actions = [
        {"action_type": a.action_type, "target": a.target,
         "status": "PENDING_HUMAN_APPROVAL" if a.requires_human_approval else "AUTO-EXECUTED",
         "reason": a.reason}
        for a in actions
    ]

    logger.stage("STAGE 8: Cognitive Knowledge Evolution (feedback into CAPG)")
    n_updated = apply_feedback(builder, actions)
    record.edges_updated_by_feedback = n_updated
    logger.info(f"Updated {n_updated} CAPG edges.")

    logger.stage("PIPELINE COMPLETE")
    record.finish()

    highlight = ground_truth_in_graph | {attacker_cas.identity}
    _finalize_outputs(record, builder, dataset_name, highlight, make_dashboard, make_graph_exports, logger)

    return builder, record


def _finalize_outputs(record, builder, dataset_name, highlight_nodes,
                       make_dashboard, make_graph_exports, logger):
    """Writes the PNG/HTML graph exports and the dashboard, all into
    outputs/<dataset>_<timestamp>/ next to the .log file, and prints
    where to find them so a non-technical user has one clear place to
    look."""
    run_dir = os.path.dirname(record.log_file_path)
    run_id = os.path.splitext(os.path.basename(record.log_file_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    if make_graph_exports:
        png_path = os.path.join(out_dir, "capg_graph.png")
        html_path = os.path.join(out_dir, "capg_graph.html")
        export_graph_png(builder, png_path, highlight_nodes=highlight_nodes,
                          title=f"CAPG -- {dataset_name}")
        export_graph_html(builder, html_path, highlight_nodes=highlight_nodes,
                           title=f"CAPG -- {dataset_name}")
        logger.info(f"Graph image saved to: {png_path}")
        logger.info(f"Interactive graph saved to: {html_path}")

    if make_dashboard:
        dashboard_path = os.path.join(out_dir, "dashboard.html")
        generate_dashboard(record, builder, dashboard_path, highlight_nodes=highlight_nodes)
        record.dashboard_path = dashboard_path
        logger.info(f"Dashboard saved to: {dashboard_path}  <-- open this in a browser")

    record.to_json(os.path.join(out_dir, "run_record.json"))
