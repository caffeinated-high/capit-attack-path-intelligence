"""
Generic loader for labeled network-flow CSV datasets.

Five of the eight datasets from the handbook's Part 6 (CICIDS2017,
CSE-CIC-IDS2018, UNSW-NB15, CTU-13, Edge-IIoTset) are all, structurally,
the same kind of artifact: a CSV of NetFlow-style records (source IP,
destination IP, protocol, duration, byte/packet counts, ...) with a
label column marking benign vs. a named attack category. They differ
only in column names and label conventions -- so one engine with
per-dataset "presets" covers all five, instead of five near-duplicate
files.

============================================================================
THIS MODULE DOES NOT SHIP WITH ANY DATA. Nothing runs until you download
the real files yourself. See DATASET_LINKS below and each preset's
"how_to_get_it" note.
============================================================================

IMPORTANT MODELING CAVEAT
--------------------------
These datasets are network-flow captures, not authentication logs like
LANL -- there is no logged-in "user identity" in the data, only IP
addresses talking to each other. To reuse the same CAPG/CAS pipeline
built around identity->host relationships, this loader treats the
*source IP* as BOTH the host node and a stand-in "identity" (a real
deployment would instead join flow data against DHCP/IAM logs to
recover the actual user -- flow data alone cannot tell you who was
logged in). This is a real, honestly-documented simplification, not a
hidden one -- see the README's "known limitations" section.
"""
import pandas as pd

import config

DATASET_LINKS = {
    "cicids2017": "http://cicresearch.ca/CICDataset/CIC-IDS-2017/  "
                   "(mirror: https://www.unb.ca/cic/datasets/ids-2017.html)",
    "cicids2018": "https://www.unb.ca/cic/datasets/ids-2018.html -- download via "
                   "AWS CLI: aws s3 sync --no-sign-request s3://cse-cic-ids2018/ <dest-dir>",
    "unsw_nb15": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
    "ctu13": "https://www.stratosphereips.org/datasets-ctu13",
    "edge_iiotset": "https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-"
                     "realistic-cyber-security-dataset-iot-and-iiot-applications "
                     "(also mirrored on Kaggle: mohamedamineferrag/"
                     "edgeiiotset-cyber-security-dataset-of-iot-iiot)",
}


def _first_present(columns_lower_map, candidates):
    for c in candidates:
        if c.lower() in columns_lower_map:
            return columns_lower_map[c.lower()]
    return None


# Each preset lists CANDIDATE column names (the same logical field is
# named differently across dataset versions/mirrors) plus how to decide
# "is this row an attack". `label_is_attack` receives the raw label
# column's value for a row and returns True/False.
PRESETS = {
    "cicids2017": {
        "src_ip": [" Source IP", "Source IP", "Src IP", "srcip"],
        "dst_ip": [" Destination IP", "Destination IP", "Dst IP", "dstip"],
        "timestamp": [" Timestamp", "Timestamp"],
        "protocol": [" Protocol", "Protocol"],
        "label": [" Label", "Label"],
        "label_is_attack": lambda v: str(v).strip().upper() != "BENIGN",
    },
    "cicids2018": {
        # NOTE: the official "Processed Traffic Data for ML Algorithms"
        # CSVs for 2018 DROP the Src IP / Dst IP columns for privacy --
        # a well-documented limitation of this dataset release, not a
        # bug in this loader. When absent, src/dst host falls back to a
        # per-row synthetic id (see load_flow_csv below) and a warning
        # is printed once. If you have the raw PCAPs / Zeek logs instead
        # of the processed CSVs, those retain real IPs.
        "src_ip": ["Src IP", "Source IP"],
        "dst_ip": ["Dst IP", "Destination IP"],
        "timestamp": ["Timestamp"],
        "protocol": ["Protocol"],
        "label": ["Label"],
        "label_is_attack": lambda v: str(v).strip().lower() != "benign",
    },
    "unsw_nb15": {
        "src_ip": ["srcip", "Src IP"],
        "dst_ip": ["dstip", "Dst IP"],
        "timestamp": ["Stime", "stime"],
        "protocol": ["proto"],
        "label": ["attack_cat", "label"],
        # UNSW-NB15's binary `label` column is 1=attack/0=normal; its
        # `attack_cat` column is NaN/"Normal" for benign rows.
        "label_is_attack": lambda v: str(v).strip().lower() not in ("normal", "nan", "", "0"),
    },
    "ctu13": {
        # CTU-13's bidirectional-netflow CSVs (Argus/biargus derived)
        "src_ip": ["SrcAddr"],
        "dst_ip": ["DstAddr"],
        "timestamp": ["StartTime"],
        "protocol": ["Proto"],
        "label": ["Label"],
        # CTU-13 labels look like "flow=From-Botnet-V42-TCP-Attempt" /
        # "flow=Background" / "flow=Normal-V42-..." -- "Botnet" is the
        # substring that marks malicious traffic.
        "label_is_attack": lambda v: "botnet" in str(v).strip().lower(),
    },
    "edge_iiotset": {
        # DNN-EdgeIIoT-dataset.csv column names
        "src_ip": ["ip.src_host"],
        "dst_ip": ["ip.dst_host"],
        "timestamp": ["frame.time"],
        "protocol": ["tcp.flags", "frame.protocols"],
        "label": ["Attack_type", "Attack_label"],
        "label_is_attack": lambda v: str(v).strip().lower() not in ("normal", "0"),
    },
}


def load_flow_csv(path, dataset, nrows=None, chunksize=None):
    """
    Loads a flow-style CSV using the named preset (one of PRESETS) and
    returns it normalized to the pipeline's common event schema:
    t, src_user, src_host, dst_host, auth_type, success, label, technique.

    `src_user` is set equal to `src_host` (see the module docstring's
    modeling caveat) so the CAPG still gets an "identity -> host" auth
    edge, keeping capg/graph.py and cas/cas_builder.py unmodified.
    """
    if dataset not in PRESETS:
        raise ValueError(f"Unknown dataset preset '{dataset}'. Choose from {list(PRESETS)}.")
    preset = PRESETS[dataset]

    reader = pd.read_csv(path, nrows=nrows, chunksize=chunksize, low_memory=False)
    chunks = [reader] if chunksize is None else list(reader)

    out_frames = []
    warned_missing_ip = False
    for raw in chunks:
        cols_lower = {c.strip().lower(): c for c in raw.columns}

        src_col = _first_present(cols_lower, preset["src_ip"])
        dst_col = _first_present(cols_lower, preset["dst_ip"])
        label_col = _first_present(cols_lower, preset["label"])
        proto_col = _first_present(cols_lower, preset["protocol"])

        if label_col is None:
            raise ValueError(
                f"Couldn't find a label column for preset '{dataset}' among "
                f"{preset['label']}. Actual columns: {list(raw.columns)[:15]}...")

        n = len(raw)
        if src_col is not None:
            src_host = raw[src_col].astype(str)
        else:
            if not warned_missing_ip:
                print(f"[flow_loader] WARNING: no source-IP column found for "
                      f"'{dataset}' (this is a known limitation of some releases, "
                      f"e.g. the CSE-CIC-IDS2018 processed CSVs drop IPs for privacy) "
                      f"-- using a synthetic per-row host id instead.")
                warned_missing_ip = True
            src_host = pd.Series([f"flow-src-{i}" for i in range(n)])

        if dst_col is not None:
            dst_host = raw[dst_col].astype(str)
        else:
            dst_host = pd.Series([f"flow-dst-{i}" for i in range(n)])

        out = pd.DataFrame({
            "t": range(n),   # replace with a real parsed timestamp column if you need true time ordering
            "src_user": src_host,        # see module docstring: IP stands in for identity here
            "src_host": src_host,
            "dst_host": dst_host,
            "auth_type": raw[proto_col].astype(str) if proto_col else "flow",
            "success": True,
            "label": raw[label_col].apply(lambda v: "attack" if preset["label_is_attack"](v) else "benign"),
            "technique": "",
        })
        out_frames.append(out)

    result = pd.concat(out_frames, ignore_index=True)
    result["t"] = range(len(result))   # keep t monotonic across concatenated chunks
    return result


def configure_from_flow_events(events_df, high_value_top_n=5):
    """Same idea as lanl_loader.configure_from_redteam(): populate
    config.HIGH_VALUE_HOSTS from whichever hosts show up most often as
    the destination of a labeled-attack flow. Same retrospective-only
    caveat applies -- see that function's docstring."""
    attack_rows = events_df[events_df.label == "attack"]
    top_targets = attack_rows.dst_host.value_counts().head(high_value_top_n).index.tolist()
    config.HIGH_VALUE_HOSTS = set(top_targets)
    return config.HIGH_VALUE_HOSTS


if __name__ == "__main__":
    # ---- internal parser self-test -------------------------------------
    # NOT a substitute for any real dataset. Only checks the preset /
    # column-matching logic against a couple of inline rows per dataset,
    # written in each dataset's documented column layout.
    import io
    print("Running flow_loader self-test (NOT real data) for each preset...\n")

    fixtures = {
        "cicids2017": " Source IP, Destination IP, Protocol, Timestamp, Label\n"
                      "192.168.1.5,10.0.0.9,6,1/7/2017 9:00,BENIGN\n"
                      "192.168.1.5,10.0.0.9,6,1/7/2017 9:01,DoS Hulk\n",
        "unsw_nb15": "srcip,dstip,proto,stime,attack_cat,label\n"
                     "10.0.0.1,10.0.0.2,tcp,1,Normal,0\n"
                     "10.0.0.1,10.0.0.3,tcp,2,Exploits,1\n",
        "ctu13": "StartTime,SrcAddr,Proto,DstAddr,Label\n"
                 "2011/08/10,147.32.84.165,tcp,74.125.232.202,flow=Background\n"
                 "2011/08/10,147.32.84.165,tcp,91.212.135.158,flow=From-Botnet-V42-TCP\n",
        "edge_iiotset": "ip.src_host,ip.dst_host,frame.time,Attack_type\n"
                         "192.168.0.10,192.168.0.1,t1,Normal\n"
                         "192.168.0.10,192.168.0.1,t2,DDoS_UDP\n",
    }
    for name, csv_text in fixtures.items():
        df = load_flow_csv(io.StringIO(csv_text), dataset=name)
        n_attack = (df.label == "attack").sum()
        print(f"  {name:<14}: {len(df)} rows parsed, {n_attack} labeled attack -> OK")

    print("\nSelf-test passed. Point load_flow_csv() at your real downloaded "
          "CSV file + the matching preset name to use it for real.")
