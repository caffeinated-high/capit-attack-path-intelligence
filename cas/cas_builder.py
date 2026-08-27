"""
Cognitive Attack State (CAS) construction -- HCAPO Stage 2.

Turns the raw event stream + CAPG into the 7 CAS variables described in
the handbook (Part 3.2): intent, objectives, privileges, vulnerabilities,
trust, environmental risk, operational context.

This is intentionally rule-based / heuristic rather than a trained
model -- CAS construction in a real system is a hybrid of rules and
learned components, and a transparent rule-based version is the right
place to start (Phase 1-2 per the roadmap) before adding a learned
intent classifier (models/gnn.py handles the learned half, over CAPG).
"""
from dataclasses import dataclass, field
from collections import defaultdict

import config


@dataclass
class CognitiveAttackState:
    identity: str
    intent: str = "unknown"
    objectives: list = field(default_factory=list)          # ranked candidate targets
    privileges: list = field(default_factory=list)          # (host, level)
    vulnerabilities: list = field(default_factory=list)      # (host, cve)
    trust_edges: list = field(default_factory=list)          # (src, dst, type)
    environmental_risk: float = 0.0
    operational_context: dict = field(default_factory=dict)

    def summary(self):
        return (f"[{self.identity}] intent={self.intent} "
                f"objectives={self.objectives} risk={self.environmental_risk:.2f} "
                f"privileges={len(self.privileges)} vulns_in_reach={len(self.vulnerabilities)}")


def build_cas(events_df, capg_builder):
    """
    Builds one CAS per identity seen in the event log.

    Intent heuristic: an identity that has touched an increasing number
    of *distinct* hosts in a short time window, culminating in contact
    with a high-value asset, is scored as 'lateral-movement -> objective
    acquisition'. This mirrors Stage 4 (Attack Intent Inference) at a
    simple, explainable, rule-based level; models/gnn.py adds a learned
    alternative over the same graph.
    """
    g = capg_builder.g
    states = {}
    by_identity = events_df.groupby("src_user")

    for identity, rows in by_identity:
        rows = rows.sort_values("t")
        touched_hosts = list(dict.fromkeys(list(rows.src_host) + list(rows.dst_host)))
        distinct_host_count = len(set(touched_hosts))
        reached_high_value = config.HIGH_VALUE_HOSTS.intersection(touched_hosts)

        # --- session segmentation: a "session" is a run of events with
        # small gaps between them. Everyday behavior is spread across the
        # whole simulation; a real campaign shows up as one dense burst.
        # We measure burstiness *within a session*, not across the whole
        # history, otherwise an attacker's normal daily logins would
        # dilute the signal (and a lucky benign user's scattered activity
        # would never look bursty).
        session_gap_threshold = 5
        sessions, current = [], [rows.iloc[0]]
        for i in range(1, len(rows)):
            prev_t, cur_t = rows.iloc[i - 1].t, rows.iloc[i].t
            if cur_t - prev_t <= session_gap_threshold:
                current.append(rows.iloc[i])
            else:
                sessions.append(current)
                current = [rows.iloc[i]]
        sessions.append(current)

        cas = CognitiveAttackState(identity=f"id:{identity}")

        # --- privileges: any host this identity authenticated to or escalated on
        cas.privileges = [(h, "escalated") for h in touched_hosts if h in reached_high_value]
        cas.privileges += [(h, "authenticated") for h in touched_hosts if h not in reached_high_value]

        # --- vulnerabilities reachable from touched hosts
        cas.vulnerabilities = [(h, cve) for h, cve in config.VULNERABLE_HOSTS.items() if h in touched_hosts]

        # --- trust relationships actually traversed
        cas.trust_edges = [(s, d, "traversed") for s, d in zip(rows.src_host, rows.dst_host) if s != d]

        # --- environmental risk: fraction of touched hosts that are internet/high value
        cas.environmental_risk = min(1.0, (len(reached_high_value) * 0.5 + distinct_host_count / 10.0))

        # --- operational context
        cas.operational_context = {
            "n_events": len(rows),
            "time_span": (rows.t.min(), rows.t.max()),
            "distinct_hosts_touched": distinct_host_count,
        }

        # --- intent / objective inference (rule-based Stage 4 baseline)
        # Burstiness = distinct hosts touched per unit time. Spread-out,
        # everyday host usage has low burstiness; a lateral-movement
        # campaign hits many hosts in a short window and has high
        # burstiness. This single heuristic is deliberately simple (and
        # will produce false positives on real traffic -- see Part 10 of
        # the handbook) but is enough to separate the embedded campaign
        # from ordinary background activity here.
        # Only sessions of 3+ chained events count as candidate lateral
        # movement -- a single isolated auth event always touches 2 hosts
        # (src+dst) and would otherwise look "bursty" by accident.
        best_session_hosts, best_session_span, best_session_hv = set(), 1, set()
        best_score = -1.0
        for sess in sessions:
            if len(sess) < 3:
                continue
            sess_hosts = set()
            for r in sess:
                sess_hosts.add(r.src_host); sess_hosts.add(r.dst_host)
            span = max(1, sess[-1].t - sess[0].t)
            score = len(sess_hosts) / span
            if score > best_score:
                best_score = score
                best_session_hosts, best_session_span = sess_hosts, span
                best_session_hv = config.HIGH_VALUE_HOSTS.intersection(sess_hosts)

        burstiness = best_score if best_score >= 0 else 0.0

        if best_session_hv and burstiness > 0.3:
            cas.intent = "objective-acquisition"
            cas.objectives = sorted(best_session_hv)
        elif burstiness > 0.3 and len(best_session_hosts) >= 3:
            cas.intent = "lateral-movement"
            cas.objectives = list(config.HIGH_VALUE_HOSTS)   # hypothesize standard high-value targets
        elif any(h in config.VULNERABLE_HOSTS for h in touched_hosts) and burstiness > 0.3:
            cas.intent = "privilege-escalation"
            cas.objectives = [h for h in touched_hosts if h in config.VULNERABLE_HOSTS]
        else:
            cas.intent = "reconnaissance-or-benign"
            cas.objectives = []

        states[identity] = cas

    return states


if __name__ == "__main__":
    from data.generate_synthetic_logs import generate
    from capg.graph import CAPGBuilder

    df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(df)
    states = build_cas(df, builder)

    # print only the interesting (non-benign) identities
    for ident, cas in states.items():
        if cas.intent != "reconnaissance-or-benign":
            print(cas.summary())
