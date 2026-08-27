"""
Cognitive Knowledge Evolution -- HCAPO Stage 8.

Closes the loop: after defense actions are (simulated as) executed,
feed the outcome back into the CAPG so the next reasoning/optimization
cycle sees an updated world. Here that means: edges targeted by a
REVOKE_CREDENTIAL_OR_PATCH action have their confidence collapsed to
~0 (the attacker's cheap path no longer exists), which will force
Stage 5 (path reasoning) to find a different, costlier path on the
next cycle if the attacker persists -- a lightweight, explainable
stand-in for the continual/online learning described in the handbook.
"""


def apply_feedback(capg_builder, executed_actions):
    g = capg_builder.g
    updated = 0
    for action in executed_actions:
        if action.action_type != "REVOKE_CREDENTIAL_OR_PATCH":
            continue
        if "->" not in action.target:
            continue
        u, v = action.target.split("->")
        if g.has_edge(u, v):
            for key in list(g[u][v]):
                g[u][v][key]["confidence"] = 0.01
                g[u][v][key]["weight"] = g[u][v][key]["weight"] * 50  # now very expensive to traverse
                updated += 1
    return updated


if __name__ == "__main__":
    from data.generate_synthetic_logs import generate
    from capg.graph import CAPGBuilder
    from cas.cas_builder import build_cas
    from reasoning.path_reasoning import top_k_paths, most_likely_source
    from optimization.hierarchical_optimizer import greedy_hierarchical_select
    from defense.defense_planner import plan_defense

    df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(df)
    states = build_cas(df, builder)
    attacker_cas = most_likely_source(states)

    print("=== Cycle 1 ===")
    paths = top_k_paths(builder, attacker_cas.identity, attacker_cas.objectives, k=5)
    for p, c, e in paths:
        print(f"  cost={c:.2f} path={' -> '.join(p)}")
    chosen, covered, _ = greedy_hierarchical_select(paths, budget=5.0)
    actions = plan_defense(chosen)
    n_updated = apply_feedback(builder, actions)
    print(f"\nStage 8: updated {n_updated} CAPG edges after defense actions.\n")

    print("=== Cycle 2 (post-feedback) ===")
    paths2 = top_k_paths(builder, attacker_cas.identity, attacker_cas.objectives, k=5)
    for p, c, e in paths2:
        print(f"  cost={c:.2f} path={' -> '.join(p)}")
