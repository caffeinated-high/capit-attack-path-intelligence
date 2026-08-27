"""
Hierarchical Cognitive Attack Path Optimization -- HCAPO Stage 6.

Given the ranked candidate paths from Stage 5, choose which
interventions to apply, at which level of the hierarchy, to break the
most (weighted-by-likelihood) attack paths per unit of "disruption
cost" (the operational cost of the intervention, e.g., isolating a
whole subnet is far more disruptive than revoking one credential).

This is implemented as a greedy weighted set-cover, which is the
standard, well-understood approximation approach for this class of
problem (provably within a log-factor of optimal) -- a reasonable,
explainable Phase-2/3 baseline before upgrading to a learned or
ILP/CVXPY-based solver as described in the handbook.

Hierarchy levels:
  HOST     - revoke a specific credential / isolate a specific host
  SUBNET   - isolate an entire subnet segment (workstations vs servers)
  MISSION  - add hardening/monitoring around a mission-critical asset
"""
from dataclasses import dataclass

import config


@dataclass
class Intervention:
    level: str          # host | subnet | mission
    action: str          # human-readable action
    target: str          # node / subnet / edge this acts on
    cost: float          # operational/business disruption cost
    paths_broken: set    # indices of candidate paths this would break


_DISRUPTION_COST = {"host": 1.0, "subnet": 4.0, "mission": 2.0}


def _subnet_of(node):
    """Reads real subnet/segment grouping from config.HOST_SUBNET_MAP
    (populated from your CMDB / asset inventory in a real deployment).
    Falls back to "unknown" if the host isn't in the map -- the greedy
    optimizer below still works correctly on "unknown", it just can't
    offer a targeted subnet-level intervention for that host."""
    return config.subnet_of(node)


def candidate_interventions(paths):
    """
    Enumerate one candidate intervention per (edge, level) that appears
    on at least one of the ranked attack paths, tagged with which paths
    it would break if applied.
    """
    candidates = {}

    for p_idx, (path, cost, edge_types) in enumerate(paths):
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            etype = edge_types[i]

            # HOST-level: revoke/patch this specific edge
            key = ("host", f"revoke_or_patch:{u}->{v}")
            candidates.setdefault(key, Intervention("host", f"Revoke credential / patch vuln on edge {u} -> {v} ({etype})",
                                                      f"{u}->{v}", _DISRUPTION_COST["host"], set()))
            candidates[key].paths_broken.add(p_idx)

            # SUBNET-level: isolate the subnet containing the destination
            subnet = _subnet_of(v)
            key = ("subnet", subnet)
            candidates.setdefault(key, Intervention("subnet", f"Isolate subnet '{subnet}'",
                                                      subnet, _DISRUPTION_COST["subnet"], set()))
            candidates[key].paths_broken.add(p_idx)

            # MISSION-level: harden the endpoint if it is one of the
            # configured high-value / mission-critical assets.
            if v in config.HIGH_VALUE_HOSTS:
                key = ("mission", v)
                candidates.setdefault(key, Intervention("mission", f"Add step-up auth / extra monitoring on '{v}'",
                                                          v, _DISRUPTION_COST["mission"], set()))
                candidates[key].paths_broken.add(p_idx)

    return list(candidates.values())


def greedy_hierarchical_select(paths, budget=5.0):
    """
    Greedy weighted set-cover: repeatedly pick the intervention with the
    best (newly-covered-paths / cost) ratio until the budget is spent or
    every candidate path is broken.
    """
    candidates = candidate_interventions(paths)
    all_path_idxs = set(range(len(paths)))
    covered = set()
    chosen = []
    remaining_budget = budget

    while covered != all_path_idxs and remaining_budget > 0:
        best, best_score = None, -1.0
        for c in candidates:
            new_paths = c.paths_broken - covered
            if not new_paths or c.cost > remaining_budget:
                continue
            score = len(new_paths) / c.cost
            if score > best_score:
                best, best_score = c, score
        if best is None:
            break
        chosen.append(best)
        covered |= best.paths_broken
        remaining_budget -= best.cost

    return chosen, covered, all_path_idxs


if __name__ == "__main__":
    from data.generate_synthetic_logs import generate
    from capg.graph import CAPGBuilder
    from cas.cas_builder import build_cas
    from reasoning.path_reasoning import top_k_paths, most_likely_source

    df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(df)
    states = build_cas(df, builder)
    attacker_cas = most_likely_source(states)
    paths = top_k_paths(builder, attacker_cas.identity, attacker_cas.objectives, k=5)

    chosen, covered, all_idxs = greedy_hierarchical_select(paths, budget=5.0)
    print(f"Covered {len(covered)}/{len(all_idxs)} candidate paths within budget.\n")
    for c in chosen:
        print(f"[{c.level.upper():7s}] {c.action}  (cost={c.cost}, breaks paths {sorted(c.paths_broken)})")
