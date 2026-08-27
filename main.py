"""
End-to-end HCAPO pipeline demo.

Runs all 8 stages from the handbook in order, on the synthetic
enterprise log with its embedded attack campaign, and prints a report
at each stage. This is the Phase 1/2 prototype referenced in the
implementation roadmap.

    python3 main.py
"""
import sys

from capg.graph import CAPGBuilder
from cas.cas_builder import build_cas
from data.generate_synthetic_logs import generate
from defense.defense_planner import execute_simulated, plan_defense
from knowledge_evolution import apply_feedback
from models.train import ground_truth_compromised_set, run_comparison
from optimization.hierarchical_optimizer import greedy_hierarchical_select
from reasoning.path_reasoning import most_likely_source, top_k_paths


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    banner("STAGE 1 + 3: Enterprise Situation Understanding + CAPG Construction")
    events_df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(events_df)
    print(f"Ingested {len(events_df)} events.")
    print("CAPG stats:", builder.stats())

    banner("STAGE 2: Cognitive Attack State (CAS) Construction")
    states = build_cas(events_df, builder)
    non_benign = {k: v for k, v in states.items() if v.intent != "reconnaissance-or-benign"}
    print(f"{len(states)} identities profiled, {len(non_benign)} flagged with non-benign intent:")
    for cas in non_benign.values():
        print("  " + cas.summary())

    banner("STAGE 4: Attack Intent Inference (rule-based CAS + learned GNN corroboration)")
    attacker_cas = most_likely_source(states)
    print("Highest-priority identity per CAS heuristic:")
    print("  " + attacker_cas.summary())
    print("\nCross-checking against the learned GNN risk ranking "
          "(models/train.py, GCN/GAT/GraphSAGE comparison):")
    _, _, ground_truth, node_scores, nodes = run_comparison()
    gnn_agrees = attacker_cas.identity in [
        nodes[i] for i in sorted(range(len(nodes)), key=lambda i: -node_scores["GAT"][i])[:5]
    ]
    print(f"\n  GAT top-5 riskiest nodes agree with CAS-selected attacker? {gnn_agrees}")

    banner("STAGE 5: Attack Path Reasoning")
    paths = top_k_paths(builder, attacker_cas.identity, attacker_cas.objectives, k=5)
    for path, cost, edge_types in paths:
        print(f"  cost={cost:6.2f}  path={' -> '.join(path)}  via={edge_types}")

    banner("STAGE 6: Hierarchical Cognitive Attack Path Optimization")
    chosen, covered, all_idxs = greedy_hierarchical_select(paths, budget=5.0)
    print(f"Covered {len(covered)}/{len(all_idxs)} candidate paths within budget=5.0")
    for c in chosen:
        print(f"  [{c.level.upper():7s}] {c.action}")

    banner("STAGE 7: Autonomous Defense Planning (simulated execution)")
    actions = plan_defense(chosen)
    execute_simulated(actions)

    banner("STAGE 8: Cognitive Knowledge Evolution (feedback into CAPG)")
    n_updated = apply_feedback(builder, actions)
    print(f"Updated {n_updated} CAPG edges. Re-running Stage 5 to confirm disruption:")
    new_paths = top_k_paths(builder, attacker_cas.identity, attacker_cas.objectives, k=5)
    for path, cost, edge_types in new_paths:
        print(f"  cost={cost:6.2f}  path={' -> '.join(path)}  via={edge_types}")

    banner("PIPELINE COMPLETE")
    print("Ground-truth compromised nodes (for evaluation only):", sorted(ground_truth))


if __name__ == "__main__":
    sys.exit(main())
