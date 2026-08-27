"""
Synthetic enterprise authentication-log generator.

Real LANL / OpTC datasets are multi-GB and not fetchable inside this
environment, so this module generates a LANL-schema-like log with the
same structure (time, src_user, src_host, dst_host, auth_type, success)
plus one embedded multi-stage red-team-style attack campaign, so the
rest of the CAPIT/HCAPO pipeline can be built and tested end to end.

Every event carries a ground-truth `label` (benign / attack) purely so
we can train and *evaluate* the GNN models later. The pipeline itself
never uses this label as an input feature -- that would be cheating.
"""
import random
from dataclasses import dataclass, field

import pandas as pd

import config

random.seed(7)

N_WORKSTATIONS = 30
N_SERVERS = 8
N_USERS = 25
SIM_STEPS = 400

WORKSTATIONS = [f"ws-{i:02d}" for i in range(N_WORKSTATIONS)]
SERVERS = [f"srv-{i:02d}" for i in range(N_SERVERS)] + ["dc-01", "finance-db"]
HOSTS = WORKSTATIONS + SERVERS
USERS = [f"user{i:02d}" for i in range(N_USERS)] + ["svc-backup", "svc-web"]

# a handful of hosts get a simulated known vulnerability
VULNERABLE_HOSTS = {
    "ws-03": "CVE-2023-1111",
    "ws-17": "CVE-2023-1234",
    "srv-02": "CVE-2022-9999",
}

MISSION_TARGET = "finance-db"


@dataclass
class Event:
    t: int
    src_user: str
    src_host: str
    dst_host: str
    auth_type: str          # network | interactive | service
    success: bool
    label: str = "benign"   # benign | attack  (ground truth, evaluation only)
    technique: str = ""     # MITRE ATT&CK-ish tag for attack events


def _benign_events(n):
    events = []
    for t in range(n):
        user = random.choice(USERS)
        src = random.choice(WORKSTATIONS)
        # benign traffic mostly workstation -> a server the user normally uses
        dst = random.choice(SERVERS[:-2])  # never touches dc-01/finance-db by default
        auth_type = random.choices(["network", "interactive", "service"], weights=[5, 3, 2])[0]
        success = random.random() > 0.03
        events.append(Event(t, user, src, dst, auth_type, success))
    return events


def _attack_campaign(start_t):
    """
    Hand-authored multi-stage campaign mirroring HCAPO Stages 1-8's
    target scenario: initial access -> local privesc -> lateral
    movement -> credential theft on the DC -> pivot to the mission
    target (finance-db).
    """
    attacker_user = "user07"          # a normal-looking, already-valid account
    beachhead = "ws-17"               # a vulnerable workstation
    events = []
    t = start_t

    # Stage: initial access (weak/guessed password)
    events.append(Event(t, attacker_user, beachhead, beachhead, "interactive",
                         True, "attack", "T1078-ValidAccounts")); t += 1

    # Stage: local privilege escalation via known CVE
    events.append(Event(t, attacker_user, beachhead, beachhead, "service",
                         True, "attack", f"T1068-PrivEsc({VULNERABLE_HOSTS[beachhead]})")); t += 1

    # Stage: lateral movement across two more workstations (recon + pivot)
    hop1, hop2 = "ws-09", "ws-22"
    events.append(Event(t, attacker_user, beachhead, hop1, "network",
                         True, "attack", "T1021-LateralMovement")); t += 2
    events.append(Event(t, attacker_user, hop1, hop2, "network",
                         True, "attack", "T1021-LateralMovement")); t += 2

    # Stage: credential dumping while reaching the domain controller
    events.append(Event(t, attacker_user, hop2, "dc-01", "network",
                         True, "attack", "T1003-CredentialDumping")); t += 3

    # Stage: domain-admin pivot to the mission-critical target
    events.append(Event(t, attacker_user, "dc-01", MISSION_TARGET, "network",
                         True, "attack", "T1078-DomainAdminPivot")); t += 1

    return events


def generate(sim_steps=SIM_STEPS, attack_start=None):
    """Also populates config.py with this demo's asset inventory
    (VULNERABLE_HOSTS / HIGH_VALUE_HOSTS / HOST_SUBNET_MAP) so the rest
    of the pipeline -- which reads exclusively from config, not from
    this module -- behaves identically whether you're running the
    synthetic demo or data/lanl_loader.py against real data."""
    config.VULNERABLE_HOSTS = dict(VULNERABLE_HOSTS)
    config.HIGH_VALUE_HOSTS = {"dc-01", MISSION_TARGET}
    config.HOST_SUBNET_MAP = {h: "workstations" for h in WORKSTATIONS}
    config.HOST_SUBNET_MAP.update({h: "servers" for h in SERVERS})
    config.HOST_SUBNET_MAP["dc-01"] = "domain-controllers"
    config.HOST_SUBNET_MAP[MISSION_TARGET] = "mission-critical"

    if attack_start is None:
        attack_start = int(sim_steps * 0.75)
    events = _benign_events(sim_steps)
    events += _attack_campaign(attack_start)
    events.sort(key=lambda e: e.t)
    df = pd.DataFrame([e.__dict__ for e in events])
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/capit_hcapo/data/synthetic_auth_logs.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} events -> {out_path}")
    print(df[df.label == "attack"])
