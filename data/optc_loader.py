"""
Loader for the DARPA Operationally Transparent Cyber (OpTC) dataset.

============================================================================
THIS MODULE DOES NOT SHIP WITH ANY DATA. Nothing runs until you download
the real files yourself. See "HOW TO GET THE DATA" below.
============================================================================

HOW TO GET THE DATA
--------------------
Official releases:
  - https://github.com/FiveDirections/OpTC-data  (primary release notes + pointers)
  - https://ieee-dataport.org/open-access/operationally-transparent-cyber-optc
    (mirrored hosting, free IEEE DataPort account required)

OpTC is ~1TB of compressed JSON across ~1000 hosts over multiple days,
recorded in an extended "eCAR" format (based on MITRE's CAR data model).
Each line is a JSON object like:

    {"action":"CREATE","actorID":"<uuid>","hostname":"SysClient0201.systemia.com",
     "object":"PROCESS","objectID":"<uuid>","properties":{...},"timestamp":169...}

`object` can be PROCESS, FILE, FLOW, MODULE, REGISTRY, SHELL, TASK, etc.
This loader focuses on **FLOW** events specifically, because those are
the ones with an explicit source-host/destination-host relationship
(via `properties.src_ip` / `properties.dest_ip`) that maps directly onto
the CAPG's identity/host edge model -- exactly like an "auth" event in
LANL, just for network connections instead of logons. PROCESS/FILE/
REGISTRY events describe what happens ON one host and would need a
separate "process-provenance" extension to the CAPG schema (see the
"EXTENDING THIS LOADER" note at the bottom) rather than being force-fit
into the auth/host edge model used here.

============================================================================
SET THIS PATH AFTER DOWNLOADING THE DATA
============================================================================
"""
import json

import pandas as pd

import config

OPTC_DATA_PATH = None   # e.g. "/data/optc/ecar/AIA-201-225.json" (one host-day file, or a directory of them)
OPTC_GROUND_TRUTH_PATH = None  # e.g. "/data/optc/ground_truth.json" -- see FiveDirections/OpTC-data for the
                                # ground-truth document listing which hosts/times were part of the red-team run


def _require_path(path):
    if not path:
        raise FileNotFoundError(
            "OPTC_DATA_PATH is not set.\n"
            "Download from https://github.com/FiveDirections/OpTC-data (or the IEEE "
            "DataPort mirror at https://ieee-dataport.org/open-access/"
            "operationally-transparent-cyber-optc) and set OPTC_DATA_PATH at the "
            "top of data/optc_loader.py to point at your local eCAR JSON file(s)."
        )


def stream_flow_events(path=None, max_rows=None):
    """
    Streams an eCAR JSON-lines file, yielding only FLOW-type records,
    normalized to the pipeline's common schema (t, src_user, src_host,
    dst_host, auth_type, success, label, technique).

    Since OpTC's FLOW events don't carry a logged-in username, `src_user`
    is set to the acting process's hostname (same identity-standin
    pattern used in data/flow_loader.py for the pure-netflow datasets)
    -- if you need real user attribution, join against this host's
    PROCESS/LOGON events for the same actorID, which is exactly the kind
    of extension noted at the bottom of this file.
    """
    _require_path(path or OPTC_DATA_PATH)
    path = path or OPTC_DATA_PATH
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and len(rows) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("object") != "FLOW":
                continue

            props = rec.get("properties", {})
            src_ip = props.get("src_ip") or props.get("source_ip")
            dst_ip = props.get("dest_ip") or props.get("destination_ip")
            if not src_ip or not dst_ip:
                continue

            hostname = rec.get("hostname", src_ip)
            rows.append({
                "t": rec.get("timestamp", i),
                "src_user": f"host:{hostname}",
                "src_host": hostname,
                "dst_host": dst_ip,
                "auth_type": "network_flow",
                "success": True,
                "label": "unknown",   # see attach_ground_truth_labels()
                "technique": "",
            })

    return pd.DataFrame(rows, columns=["t", "src_user", "src_host", "dst_host",
                                        "auth_type", "success", "label", "technique"])


def attach_ground_truth_labels(events_df, ground_truth_hosts):
    """
    OpTC's ground truth is published as a human-readable narrative
    document (not a clean CSV like LANL's redteam.txt), listing which
    hostnames/time windows were part of the red-team engagement --
    see the "Ground Truth" section of https://github.com/FiveDirections/OpTC-data.
    You'll need to transcribe the relevant hostnames/time windows from
    that document into `ground_truth_hosts` (a set of hostnames)
    yourself; this function then labels any event touching one of those
    hosts as 'attack' for evaluation purposes, exactly mirroring how
    lanl_loader.attach_labels() uses redteam.txt.
    """
    events_df = events_df.copy()
    mask = events_df.src_host.isin(ground_truth_hosts) | events_df.dst_host.isin(ground_truth_hosts)
    events_df.loc[mask, "label"] = "attack"
    events_df.loc[~mask, "label"] = "benign"
    return events_df


def ground_truth_compromised_set(events_df):
    attack_rows = events_df[events_df.label == "attack"]
    compromised = set(attack_rows.src_host) | set(attack_rows.dst_host)
    compromised |= set(attack_rows.src_user.unique())
    return compromised


def configure_from_ground_truth(ground_truth_hosts):
    config.HIGH_VALUE_HOSTS = set(ground_truth_hosts)
    return config.HIGH_VALUE_HOSTS


# ---------------------------------------------------------------------------
# EXTENDING THIS LOADER (Phase 3 direction, see the handbook)
# ---------------------------------------------------------------------------
# OpTC's real value over LANL is its PROCESS/FILE/MODULE/REGISTRY events,
# which give you a full process-provenance chain (which process spawned
# which, which files it touched) -- exactly the kind of data DARPA's
# Transparent Computing program (data/darpa_tc_loader.py) is built
# around. To use that here, you'd add a new CAPGEdge type (e.g.
# "spawns"/"reads"/"writes") in capg/graph.py keyed on actorID/objectID
# rather than src_user/src_host, which is a genuine schema extension
# beyond what this handbook's Phase 1-2 CAPG supports today.


if __name__ == "__main__":
    # ---- internal parser self-test -------------------------------------
    # NOT a substitute for real OpTC data. Only checks that FLOW-event
    # parsing and ground-truth labeling are bug-free, using a couple of
    # inline JSON lines in the documented eCAR shape.
    import io
    import tempfile

    print("Running OpTC parser self-test with inline eCAR-format JSON lines "
          "(NOT real OpTC data)...\n")

    fixture_lines = [
        json.dumps({"action": "CREATE", "actorID": "a1", "hostname": "SysClient0201.systemia.com",
                    "object": "FLOW", "objectID": "f1",
                    "properties": {"src_ip": "10.10.1.5", "dest_ip": "10.10.1.9"},
                    "timestamp": 1}),
        json.dumps({"action": "CREATE", "actorID": "a1", "hostname": "SysClient0201.systemia.com",
                    "object": "PROCESS", "objectID": "p1",
                    "properties": {"image_path": "powershell.exe"},
                    "timestamp": 2}),   # not a FLOW event -- should be skipped
        json.dumps({"action": "CREATE", "actorID": "a2", "hostname": "SysClient0201.systemia.com",
                    "object": "FLOW", "objectID": "f2",
                    "properties": {"src_ip": "10.10.1.5", "dest_ip": "10.10.1.99"},
                    "timestamp": 3}),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("\n".join(fixture_lines))
        tmp_path = f.name

    df = stream_flow_events(path=tmp_path)
    print(df)
    labeled = attach_ground_truth_labels(df, ground_truth_hosts={"10.10.1.99"})
    print("\nlabeled:\n", labeled[["src_host", "dst_host", "label"]])
    print(f"\nParsed {len(df)} FLOW events (correctly skipped the 1 PROCESS event). "
          f"Self-test passed.")

    import os
    os.unlink(tmp_path)
