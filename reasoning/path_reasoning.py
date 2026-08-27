"""
Attack Path Reasoning -- HCAPO Stage 5.

Given the CAPG, a starting point (the attacker's last known-good
foothold, from the CAS), and one or more inferred objectives, find and
rank the most likely attack paths using edge weight as an attacker
"cost" (lower weight = cheaper/likelier to traverse -- see the
_EDGE_TYPE_BASE_COST model in capg/graph.py). This is a direct,
explainable stand-in for the fuller MDP-rollout / probabilistic search
described in the handbook; it is exact and fast enough at this graph
scale, and gives HCAPO's optimization stage (Stage 6) concrete paths to
work with.
"""
import networkx as nx


def top_k_paths(capg_builder, source, targets, k=3, cutoff_nodes=12):
    """
    Returns up to k lowest-cost simple paths from `source` to any node in
    `targets`, each as (path, total_cost, edge_types_used).
    """
    g = capg_builder.g
    results = []

    # collapse the MultiDiGraph to a simple weighted DiGraph using the
    # minimum-cost edge between any two nodes (an attacker will always
    # prefer the cheapest available technique between two footholds).
    simple = nx.DiGraph()
    for u, v, data in g.edges(data=True):
        w = data["weight"]
        if not simple.has_edge(u, v) or simple[u][v]["weight"] > w:
            simple.add_edge(u, v, weight=w, edge_type=data["edge_type"])

    for target in targets:
        if source not in simple or target not in simple:
            continue
        try:
            paths_gen = nx.shortest_simple_paths(simple, source, target, weight="weight")
            for i, path in enumerate(paths_gen):
                if i >= k or len(path) > cutoff_nodes:
                    break
                cost = sum(simple[path[j]][path[j + 1]]["weight"] for j in range(len(path) - 1))
                edge_types = [simple[path[j]][path[j + 1]]["edge_type"] for j in range(len(path) - 1)]
                results.append((path, cost, edge_types))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # nx.shortest_simple_paths raises this lazily on the first
            # next() call, not when the generator is created -- so the
            # try/except has to wrap the iteration, not just the call.
            continue

    results.sort(key=lambda r: r[1])
    return results[:k]


_INTENT_PRIORITY = {
    "objective-acquisition": 3,
    "lateral-movement": 2,
    "privilege-escalation": 1,
    "reconnaissance-or-benign": 0,
    "unknown": 0,
}


def most_likely_source(cas_states):
    """Pick the identity CAS with the most advanced inferred intent
    (Stage 4's output), breaking ties by environmental_risk, as the
    attacker to reason about in Stage 5. Prioritizing intent stage over
    a raw risk score avoids picking a merely "busy" benign service
    account (see the false-positive discussion in the handbook's
    critical-analysis section)."""
    best = max(
        cas_states.values(),
        key=lambda c: (_INTENT_PRIORITY.get(c.intent, 0), c.environmental_risk),
        default=None,
    )
    return best


if __name__ == "__main__":
    from data.generate_synthetic_logs import generate
    from capg.graph import CAPGBuilder
    from cas.cas_builder import build_cas

    df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(df)
    states = build_cas(df, builder)

    attacker_cas = most_likely_source(states)
    print("Reasoning about:", attacker_cas.summary())

    # start from the identity node itself; objectives come straight from CAS
    paths = top_k_paths(builder, attacker_cas.identity, attacker_cas.objectives, k=3)
    for path, cost, edge_types in paths:
        print(f"cost={cost:.2f}  path={' -> '.join(path)}  via={edge_types}")
