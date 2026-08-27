"""
Cognitive Attack Path Graph (CAPG).

Implements HCAPO Stage 1 (Enterprise Situation Understanding, via the
log ingestion helpers) and Stage 3 (CAPG Construction).

The graph is a networkx.MultiDiGraph so multiple typed/timestamped
edges can exist between the same pair of nodes (e.g., many auth events
between the same user and host over time).
"""
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

import config


@dataclass
class CAPGNode:
    id: str
    type: str                    # host | identity | vuln | mission_target
    features: dict = field(default_factory=dict)
    last_seen: float = 0.0


@dataclass
class CAPGEdge:
    src: str
    dst: str
    edge_type: str                # auth | escalate | exploit | trust | pivot
    weight: float                 # "cost" to the attacker: lower = easier/likelier
    mitre_technique: str = ""
    timestamp: float = 0.0
    confidence: float = 1.0


# Rough cost model: how easy each edge type is for an attacker to traverse.
# Lower cost = more attractive path (used later for shortest-path reasoning).
_EDGE_TYPE_BASE_COST = {
    "auth": 3.0,
    "trust": 2.0,
    "exploit": 1.0,     # cheapest once a vuln is known -- highest priority to patch
    "escalate": 1.5,
    "pivot": 2.5,
}


class CAPGBuilder:
    """Incrementally builds/updates a CAPG from a stream of log events."""

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._seed_static_topology()

    def _seed_static_topology(self):
        """Stage 1: seed known static facts (asset inventory + vuln scan),
        read from config.py -- see that module's docstring for why this
        must come from an external inventory, not be learned from traffic."""
        for host, cve in config.VULNERABLE_HOSTS.items():
            self.add_node(host, "host", {"vulnerable": True, "cve": cve})
            vuln_id = f"vuln:{cve}"
            self.add_node(vuln_id, "vuln", {"cve": cve})
            self.add_edge(host, vuln_id, "exploit", cost_override=_EDGE_TYPE_BASE_COST["exploit"],
                          mitre_technique="T1068", timestamp=0)
        for host in config.HIGH_VALUE_HOSTS:
            self.add_node(host, "mission_target", {"criticality": "high"})

    # ---- node/edge primitives -------------------------------------------------
    def add_node(self, node_id, node_type, features=None, t=0.0):
        if node_id not in self.g:
            self.g.add_node(node_id, type=node_type, features=features or {}, last_seen=t)
        else:
            self.g.nodes[node_id]["last_seen"] = max(self.g.nodes[node_id]["last_seen"], t)
            if features:
                self.g.nodes[node_id]["features"].update(features)

    def add_edge(self, src, dst, edge_type, cost_override=None, mitre_technique="",
                 timestamp=0.0, confidence=1.0):
        weight = cost_override if cost_override is not None else _EDGE_TYPE_BASE_COST.get(edge_type, 2.0)
        self.g.add_edge(src, dst, key=f"{edge_type}:{timestamp}", edge_type=edge_type,
                         weight=weight, mitre_technique=mitre_technique,
                         timestamp=timestamp, confidence=confidence)

    # ---- Stage 1 + 3: ingest a batch of auth events ---------------------------
    def ingest_events(self, events_df):
        """
        Stage 1 (situation understanding) + Stage 3 (CAPG construction),
        fused here for simplicity: every event updates node freshness and
        adds/refreshes the corresponding typed edges.
        """
        for _, ev in events_df.iterrows():
            self.add_node(ev.src_host, "host", t=ev.t)
            self.add_node(ev.dst_host, "host", t=ev.t)
            identity_id = f"id:{ev.src_user}"
            self.add_node(identity_id, "identity", t=ev.t)

            # identity -> source host: "is logged into"
            self.add_edge(identity_id, ev.src_host, "auth", timestamp=ev.t,
                          mitre_technique=ev.technique)

            if ev.src_host != ev.dst_host:
                edge_type = "pivot" if ev.dst_host in config.HIGH_VALUE_HOSTS else "trust"
                self.add_edge(ev.src_host, ev.dst_host, edge_type, timestamp=ev.t,
                              mitre_technique=ev.technique, confidence=1.0 if ev.success else 0.4)

            # If this event is tagged with a known-technique privesc/
            # credential-theft label, add an explicit "escalate" edge.
            # The synthetic demo generator tags events this way; real
            # logs (e.g. raw LANL auth.txt) don't carry MITRE labels, so
            # `ev.technique` will just be "" and no escalate edges get
            # added from this rule -- the CAPG still works, it simply
            # has one less (very informative) edge type until you enrich
            # the pipeline with an actual technique classifier.
            if "PrivEsc" in ev.technique or "CredentialDumping" in ev.technique or "DomainAdminPivot" in ev.technique:
                self.add_edge(identity_id, ev.dst_host, "escalate", timestamp=ev.t,
                              mitre_technique=ev.technique)

    # ---- convenience -----------------------------------------------------------
    def stats(self):
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "node_types": _count_by(self.g.nodes(data=True), lambda d: d[1]["type"]),
            "edge_types": _count_by(self.g.edges(data=True), lambda d: d[2]["edge_type"]),
        }


def _count_by(iterable, key_fn):
    counts = {}
    for item in iterable:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts


if __name__ == "__main__":
    from data.generate_synthetic_logs import generate
    df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(df)
    print(builder.stats())
