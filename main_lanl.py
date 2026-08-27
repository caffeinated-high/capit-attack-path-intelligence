"""
End-to-end HCAPO pipeline running on the REAL LANL dataset instead of
the synthetic demo. Structurally identical to main.py -- same 8 stages,
same modules -- the only difference is where Stage 1's data comes from.

BEFORE RUNNING THIS:
  1. Download auth.txt.gz and redteam.txt.gz from
     https://csr.lanl.gov/data/cyber1/
  2. Open data/lanl_loader.py and set LANL_AUTH_PATH / LANL_REDTEAM_PATH
     to point at your local copies.
  3. (Optional) Tune TIME_WINDOW_SECONDS / MAX_ROWS below -- auth.txt.gz
     has 1B+ rows, so start small and widen once you've confirmed things
     work.

    python3 main_lanl.py
"""
import sys

import config
from capg.graph import CAPGBuilder
from cas.cas_builder import build_cas
from data import lanl_loader
from defense.defense_planner import execute_simulated, plan_defense
from knowledge_evolution import apply_feedback
from models.train import compare_models
from optimization.hierarchical_optimizer import greedy_hierarchical_select
from reasoning.path_reasoning import most_likely_source, top_k_paths

# ---------------------------------------------------------------------------
# Tune these once your paths are set in data/lanl_loader.py
# ---------------------------------------------------------------------------
TIME_WINDOW_SECONDS = 3600     # first simulated hour; widen once this works
START_TIME = 0
MAX_ROWS = 200_000             # hard safety cap regardless of window size


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    banner("STAGE 1: Load real LANL data + ground truth")
    redteam_df = lanl_loader.load_redteam()
    print(f"Loaded {len(redteam_df)} red-team ground-truth events.")

    # Populate config from an (offline, retrospective) read of the
    # red-team data -- see configure_from_redteam()'s docstring for why
    # this is a demo/benchmark convenience, not how you'd do it live.
    lanl_loader.configure_from_redteam(redteam_df, high_value_top_n=5)
    print("config.HIGH_VALUE_HOSTS set to:", config.HIGH_VALUE_HOSTS)

    events_df = lanl_loader.load_auth_window(
        time_window_seconds=TIME_WINDOW_SECONDS, start_time=START_TIME, max_rows=MAX_ROWS)
    events_df = lanl_loader.attach_labels(events_df, redteam_df)
    print(f"Loaded {len(events_df)} auth events in the window "
          f"[{START_TIME}, {START_TIME + TIME_WINDOW_SECONDS}).")
    print(f"  of which {int((events_df.label == 'attack').sum())} match red-team ground truth.")

    if events_df.empty:
        print("\nNo events in this window -- widen TIME_WINDOW_SECONDS/MAX_ROWS "
              "or change START_TIME and try again.")
        return 1

    banner("STAGE 1 + 3: CAPG Construction")
    builder = CAPGBuilder()
    builder.ingest_events(events_df)
    print("CAPG stats:", builder.stats())

    banner("STAGE 2: Cognitive Attack State (CAS) Construction")
    states = build_cas(events_df, builder)
    non_benign = {k: v for k, v in states.items() if v.intent != "reconnaissance-or-benign"}
    print(f"{len(states)} identities profiled, {len(non_benign)} flagged with non-benign intent.")
    for cas in list(non_benign.values())[:15]:
        print("  " + cas.summary())

    banner("STAGE 4: Attack Intent Inference (CAS heuristic + learned GNN)")
    attacker_cas = most_likely_source(states)
    if attacker_cas is None:
        print("No identities found in this window.")
        return 1
    print("Highest-priority identity per CAS heuristic:")
    print("  " + attacker_cas.summary())

    ground_truth = lanl_loader.redteam_ground_truth_set(redteam_df)
    # only keep ground-truth nodes that actually appear in this window's
    # graph -- the full redteam.txt spans all 58 days, our window doesn't
    ground_truth_in_window = ground_truth & set(builder.g.nodes())
    print(f"\n{len(ground_truth_in_window)} ground-truth compromised nodes fall "
          f"inside this window's CAPG.")
    if len(ground_truth_in_window) >= 2:
        compare_models(builder, ground_truth_in_window)
    else:
        print("Too few ground-truth positives in this window to train/evaluate "
              "the GNNs meaningfully -- widen the time window.")

    banner("STAGE 5: Attack Path Reasoning")
    paths = top_k_paths(builder, attacker_cas.identity, attacker_cas.objectives, k=5)
    if not paths:
        print("No path found from the flagged identity to any high-value host "
              "in this window/graph.")
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
    print(f"Updated {n_updated} CAPG edges.")

    banner("PIPELINE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
