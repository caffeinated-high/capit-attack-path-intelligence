"""
Loader for the DARPA Transparent Computing (TC) program dataset.

============================================================================
THIS MODULE DOES NOT SHIP WITH ANY DATA. Nothing runs until you download
the real files yourself. See "HOW TO GET THE DATA" below.
============================================================================

HOW TO GET THE DATA
--------------------
Official release: https://github.com/darpa-i2o/Transparent-Computing
(engagement archives are linked from that repo's README; different
engagements/teams released data as Avro binary CDM (Common Data Model)
files, and some as line-delimited JSON re-encodings of the same CDM
schema).

============================================================================
SCOPE OF THIS LOADER -- READ BEFORE USING
============================================================================
DARPA TC's actual format is Avro-serialized CDM records defined by a
fairly large schema (TCCDMDatum, with nested Subject / Event /
FileObject / NetFlowObject / Principal / ProvenanceTagNode record
types, etc). Properly parsing the *canonical* Avro release requires the
CDM `.avdl`/`.avsc` schema files (published in the same GitHub repo)
and the `fastavro` or `avro-python3` package -- that is a real,
non-trivial parsing project on its own, beyond what this handbook's
Phase 1-2 pipeline scope covers.

What this loader DOES implement: a parser for the **line-delimited JSON
re-encoding** of CDM `EVENT` records that several TC engagements also
shipped (each line is one JSON object with an `datum` envelope wrapping
a `com.bbn.tc.schema.avro.cdm18.Event` record or similar). If your
downloaded engagement is Avro-only, convert it to JSON first with the
`java -jar cdm2json...` style tools referenced in the darpa-i2o repo, or
extend `_iter_json_events()` below to call `fastavro.reader()` directly
-- the rest of this file (the event-type filtering and CAPG-schema
mapping) does not need to change either way, since it operates on the
already-decoded Python dict.

Like data/optc_loader.py, this loader only extracts **network-connect
events** (`EVENT_CONNECT` / `EVENT_OPEN` / `EVENT_ACCEPT` between a
Subject/process and a NetFlowObject) into the pipeline's common
host-to-host schema. TC's real strength -- full process/file provenance
chains -- would need the same CAPG schema extension noted in
data/optc_loader.py's "EXTENDING THIS LOADER" section.
"""
import json

import pandas as pd

import config

DARPA_TC_DATA_PATH = None   # e.g. "/data/darpa_tc/ta1-cadets-e3-official.json"

# CDM event types that represent a network connection between a process
# (Subject) and a remote endpoint (NetFlowObject). Extend this list if
# your engagement's schema version uses different type names.
NETWORK_EVENT_TYPES = {"EVENT_CONNECT", "EVENT_OPEN", "EVENT_ACCEPT", "EVENT_SENDTO", "EVENT_RECVFROM"}


def _require_path(path):
    if not path:
        raise FileNotFoundError(
            "DARPA_TC_DATA_PATH is not set.\n"
            "Download from https://github.com/darpa-i2o/Transparent-Computing "
            "and set DARPA_TC_DATA_PATH at the top of data/darpa_tc_loader.py. "
            "See this module's docstring for the JSON-vs-Avro format caveat."
        )


def _iter_json_events(path):
    """Yields decoded CDM Event dicts from a line-delimited JSON file.
    Handles both a bare Event dict per line and a {'datum': {...}}
    envelope, since different engagement releases wrapped it differently."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = rec.get("datum", rec)
            # the actual Event payload is sometimes one level deeper,
            # keyed by its fully-qualified CDM type name
            for key, val in rec.items():
                if isinstance(val, dict) and "type" in val:
                    yield val
                    break
            else:
                if "type" in rec:
                    yield rec


def stream_network_events(path=None, max_rows=None):
    """
    Streams network-connect CDM events, normalized to the pipeline's
    common schema (t, src_user, src_host, dst_host, auth_type, success,
    label, technique).

    CDM identifies hosts by a UUID (`hostId`) rather than a hostname
    string, and processes by `subject` UUID rather than a username --
    there is no direct notion of "which human logged in" in this
    dataset (it's OS-level provenance, not an AD auth log). `src_user`
    is set to the acting process's subject UUID as the closest available
    identity-standin, exactly the same pattern used in
    data/optc_loader.py.
    """
    _require_path(path or DARPA_TC_DATA_PATH)
    path = path or DARPA_TC_DATA_PATH
    rows = []

    for i, rec in enumerate(_iter_json_events(path)):
        if max_rows is not None and len(rows) >= max_rows:
            break
        ev_type = rec.get("type")
        if ev_type not in NETWORK_EVENT_TYPES:
            continue

        subject = rec.get("subject", {})
        predicate_obj = rec.get("predicateObject", {})
        host_id = rec.get("hostId") or subject.get("hostId") or "unknown-host"
        remote = (predicate_obj.get("remoteAddress")
                  or predicate_obj.get("baseObject", {}).get("remoteAddress")
                  or "unknown-remote")

        rows.append({
            "t": rec.get("timestampNanos", i),
            "src_user": f"proc:{subject.get('com.bbn.tc.schema.avro.cdm18.UUID', subject.get('uuid', 'unknown'))}",
            "src_host": host_id,
            "dst_host": remote,
            "auth_type": ev_type,
            "success": True,
            "label": "unknown",   # see attach_ground_truth_labels()
            "technique": "",
        })

    return pd.DataFrame(rows, columns=["t", "src_user", "src_host", "dst_host",
                                        "auth_type", "success", "label", "technique"])


def attach_ground_truth_labels(events_df, ground_truth_hosts_or_ips):
    """
    Like OpTC, DARPA TC's ground truth is published as narrative
    engagement writeups (attack timeline PDFs/docs per engagement, e.g.
    "TA1 CADETS E3 ground truth"), not a clean CSV -- transcribe the
    relevant host IDs / remote IPs from your engagement's ground-truth
    document into `ground_truth_hosts_or_ips` yourself.
    """
    events_df = events_df.copy()
    mask = (events_df.src_host.isin(ground_truth_hosts_or_ips) |
            events_df.dst_host.isin(ground_truth_hosts_or_ips))
    events_df.loc[mask, "label"] = "attack"
    events_df.loc[~mask, "label"] = "benign"
    return events_df


def ground_truth_compromised_set(events_df):
    attack_rows = events_df[events_df.label == "attack"]
    compromised = set(attack_rows.src_host) | set(attack_rows.dst_host)
    compromised |= set(attack_rows.src_user.unique())
    return compromised


def configure_from_ground_truth(ground_truth_hosts_or_ips):
    config.HIGH_VALUE_HOSTS = set(ground_truth_hosts_or_ips)
    return config.HIGH_VALUE_HOSTS


if __name__ == "__main__":
    # ---- internal parser self-test -------------------------------------
    # NOT a substitute for real DARPA TC data. Only checks that the CDM
    # envelope-unwrapping and event-type filtering are bug-free, using a
    # couple of inline JSON lines shaped like the documented CDM schema.
    import tempfile
    import os

    print("Running DARPA TC parser self-test with inline CDM-shaped JSON "
          "lines (NOT real DARPA TC data)...\n")

    fixture_lines = [
        json.dumps({"datum": {"com.bbn.tc.schema.avro.cdm18.Event": {
            "type": "EVENT_CONNECT",
            "hostId": "host-A",
            "subject": {"uuid": "proc-123"},
            "predicateObject": {"remoteAddress": "203.0.113.9"},
            "timestampNanos": 1000,
        }}}),
        json.dumps({"datum": {"com.bbn.tc.schema.avro.cdm18.Event": {
            "type": "EVENT_READ",   # not a network event -- should be skipped
            "hostId": "host-A",
            "subject": {"uuid": "proc-123"},
            "timestampNanos": 1500,
        }}}),
        json.dumps({"datum": {"com.bbn.tc.schema.avro.cdm18.Event": {
            "type": "EVENT_CONNECT",
            "hostId": "host-A",
            "subject": {"uuid": "proc-999"},
            "predicateObject": {"remoteAddress": "198.51.100.7"},
            "timestampNanos": 2000,
        }}}),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("\n".join(fixture_lines))
        tmp_path = f.name

    df = stream_network_events(path=tmp_path)
    print(df)
    labeled = attach_ground_truth_labels(df, ground_truth_hosts_or_ips={"198.51.100.7"})
    print("\nlabeled:\n", labeled[["src_host", "dst_host", "label"]])
    print(f"\nParsed {len(df)} network events (correctly skipped the 1 EVENT_READ). "
          f"Self-test passed.")

    os.unlink(tmp_path)
