"""
Autonomous Defense Planning -- HCAPO Stage 7.

Translates the interventions chosen by Stage 6 into concrete,
executable actions. This module *simulates* execution (prints a SOAR-
style action log) rather than actually touching any real system --
per the handbook's Part 10 critique, real autonomous enforcement
should ship as human-approved decision support first and fully
autonomous only as a later, carefully-guarded stretch goal.
"""
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class DefenseAction:
    action_type: str      # ISOLATE_HOST | REVOKE_CREDENTIAL | PATCH | STEP_UP_AUTH | ISOLATE_SUBNET
    target: str
    reason: str
    requires_human_approval: bool = True


_ACTION_MAP = {
    "host": "REVOKE_CREDENTIAL_OR_PATCH",
    "subnet": "ISOLATE_SUBNET",
    "mission": "STEP_UP_AUTH_AND_MONITOR",
}

# Actions with a wide blast radius should always require a human in the
# loop; narrow, low-risk actions can be auto-approved once the system
# has a track record -- this mapping is itself a policy the human
# security team should be able to tune.
_AUTO_APPROVE_LEVELS = {"host"}


def plan_defense(chosen_interventions):
    actions = []
    for interv in chosen_interventions:
        action_type = _ACTION_MAP[interv.level]
        requires_approval = interv.level not in _AUTO_APPROVE_LEVELS
        actions.append(DefenseAction(
            action_type=action_type,
            target=interv.target,
            reason=interv.action,
            requires_human_approval=requires_approval,
        ))
    return actions


def execute_simulated(actions, verbose=True):
    """Prints what WOULD happen -- no real system is touched. A real
    deployment would swap this for SOAR-platform API calls guarded by
    the requires_human_approval flag. Set verbose=False when the caller
    (e.g. pipeline_core.py) is going to log the returned lines itself,
    to avoid printing everything twice."""
    log = []
    now = datetime.now(timezone.utc).isoformat()
    for a in actions:
        status = "PENDING_HUMAN_APPROVAL" if a.requires_human_approval else "AUTO-EXECUTED"
        entry = f"[{now}] {a.action_type:<26s} target={a.target:<20s} status={status}  reason={a.reason}"
        log.append(entry)
        if verbose:
            print(entry)
    return log


if __name__ == "__main__":
    from data.generate_synthetic_logs import generate
    from capg.graph import CAPGBuilder
    from cas.cas_builder import build_cas
    from reasoning.path_reasoning import top_k_paths, most_likely_source
    from optimization.hierarchical_optimizer import greedy_hierarchical_select

    df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(df)
    states = build_cas(df, builder)
    attacker_cas = most_likely_source(states)
    paths = top_k_paths(builder, attacker_cas.identity, attacker_cas.objectives, k=5)
    chosen, covered, all_idxs = greedy_hierarchical_select(paths, budget=5.0)

    actions = plan_defense(chosen)
    execute_simulated(actions)
