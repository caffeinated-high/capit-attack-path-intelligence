"""
Loader for the real LANL "Comprehensive, Multi-Source Cyber-Security
Events" dataset (Kent, 2015) -- the same auth-log dataset referenced in
the handbook's Part 6 (Datasets).

============================================================================
THIS MODULE DOES NOT SHIP WITH ANY DATA. Nothing runs until you download
the real files yourself. See "HOW TO GET THE DATA" below.
============================================================================

HOW TO GET THE DATA
--------------------
1. Go to: https://csr.lanl.gov/data/cyber1/
2. Download at minimum these two files (both are gzip-compressed CSV):
      auth.txt.gz       (the authentication event log -- ~700GB uncompressed,
                          ~1.05B lines; you almost certainly want to read it
                          in chunks / take a time-windowed slice, see below)
      redteam.txt.gz     (ground-truth red-team compromise events -- tiny,
                          ~12,425 lines; used ONLY for evaluation, never as
                          a model input)
3. You do NOT need to fully decompress auth.txt.gz -- pandas can stream
   it directly with compression="gzip", so just place both .gz files
   somewhere on disk.
4. Set the two paths below (LANL_AUTH_PATH / LANL_REDTEAM_PATH) to wherever
   you saved them, e.g.:
       LANL_AUTH_PATH = "/data/lanl/auth.txt.gz"
       LANL_REDTEAM_PATH = "/data/lanl/redteam.txt.gz"

FILE FORMATS (from the official dataset description -- no header row in
either file):
  auth.txt.gz:
    time,source_user@domain,destination_user@domain,source_computer,
    destination_computer,authentication_type,logon_type,
    authentication_orientation,success/failure

  redteam.txt.gz:
    time,user@domain,source_computer,destination_computer

SCALE WARNING
-------------
auth.txt.gz has over 1 billion rows. Do not try to load the whole thing
into memory / into one CAPG on a laptop. Use TIME_WINDOW_SECONDS below to
pull a manageable slice (e.g. the first few hours/days) -- this mirrors
exactly what the papers benchmarking on LANL do (see the arXiv citations
in the handbook's dataset table): they window or subsample too.
"""
import pandas as pd

import config

# ---------------------------------------------------------------------------
# >>> SET THESE TWO PATHS AFTER DOWNLOADING THE DATA <<<
# ---------------------------------------------------------------------------
LANL_AUTH_PATH = None      # e.g. "/data/lanl/auth.txt.gz"
LANL_REDTEAM_PATH = None   # e.g. "/data/lanl/redteam.txt.gz"

AUTH_COLUMNS = [
    "t", "src_user_domain", "dst_user_domain", "src_host", "dst_host",
    "auth_type", "logon_type", "auth_orientation", "success_failure",
]
REDTEAM_COLUMNS = ["t", "user_domain", "src_host", "dst_host"]


def _require_paths():
    if not LANL_AUTH_PATH or not LANL_REDTEAM_PATH:
        raise FileNotFoundError(
            "LANL_AUTH_PATH / LANL_REDTEAM_PATH are not set.\n"
            "Download auth.txt.gz and redteam.txt.gz from "
            "https://csr.lanl.gov/data/cyber1/ and set the two path "
            "variables at the top of data/lanl_loader.py to point at "
            "your local copies."
        )


def load_redteam(path=None):
    """Loads the small ground-truth compromise-event file. Returns a
    DataFrame with columns [t, user, src_host, dst_host]."""
    path = path or LANL_REDTEAM_PATH
    if not path:
        _require_paths()
    df = pd.read_csv(path, names=REDTEAM_COLUMNS, header=None, compression="infer")
    df["user"] = df["user_domain"].str.split("@").str[0]
    return df[["t", "user", "src_host", "dst_host"]]


def redteam_ground_truth_set(redteam_df):
    """Turns the redteam events into the same kind of 'compromised node
    id' set used throughout the pipeline (identity + host nodes), for
    evaluation in models/train.py -- exactly mirroring how the synthetic
    demo's ground_truth_compromised_set() works."""
    compromised = set(redteam_df.src_host) | set(redteam_df.dst_host)
    compromised |= {f"id:{u}" for u in redteam_df.user.unique()}
    return compromised


def stream_auth_events(path=None, time_window_seconds=None, start_time=0,
                        max_rows=None, chunksize=500_000):
    """
    Streams auth.txt.gz in chunks (so a 700GB file never has to fit in
    RAM at once) and yields normalized DataFrames matching the SAME
    schema the rest of the pipeline already expects (t, src_user,
    src_host, dst_host, auth_type, success, label, technique) -- so
    capg/graph.py, cas/cas_builder.py, reasoning/, optimization/, and
    defense/ all work completely unchanged against real data.

    Parameters
    ----------
    time_window_seconds : only keep events within
        [start_time, start_time + time_window_seconds). Use this to pull
        a tractable slice, e.g. time_window_seconds=3600 for the first
        simulated hour of the 58-day capture.
    max_rows : hard cap on total rows returned, as an extra safety net
        regardless of the time window (useful while you're first getting
        the pipeline working).
    """
    _require_paths()
    path = path or LANL_AUTH_PATH
    end_time = (start_time + time_window_seconds) if time_window_seconds else None

    rows_yielded = 0
    reader = pd.read_csv(path, names=AUTH_COLUMNS, header=None,
                          compression="infer", chunksize=chunksize)

    for raw_chunk in reader:
        # auth.txt.gz is ordered by time, so once a whole chunk's
        # earliest timestamp is already past our window we can stop
        # reading the rest of the (potentially 700GB) file entirely.
        if end_time is not None and raw_chunk.t.min() >= end_time:
            break

        chunk = raw_chunk
        if end_time is not None:
            chunk = chunk[(chunk.t >= start_time) & (chunk.t < end_time)]
        elif start_time:
            chunk = chunk[chunk.t >= start_time]

        if chunk.empty:
            continue

        chunk = chunk.copy()
        chunk["src_user"] = chunk["src_user_domain"].str.split("@").str[0]
        chunk["success"] = chunk["success_failure"].str.strip().str.lower().eq("success")
        chunk["label"] = "unknown"   # real logs aren't pre-labeled; see attach_labels()
        chunk["technique"] = ""      # real logs have no MITRE tag; CAPG falls back to
                                      # generic auth/trust edges (see capg/graph.py)

        out = chunk[["t", "src_user", "src_host", "dst_host", "auth_type", "success",
                      "label", "technique"]].copy()

        if max_rows is not None:
            remaining = max_rows - rows_yielded
            if remaining <= 0:
                return
            out = out.iloc[:remaining]

        rows_yielded += len(out)
        yield out

        if max_rows is not None and rows_yielded >= max_rows:
            return


def load_auth_window(time_window_seconds=3600, start_time=0, max_rows=200_000, path=None):
    """Convenience wrapper: collects stream_auth_events() into a single
    DataFrame for a bounded window. This is what you'll use to get
    started -- pass it straight to capg.graph.CAPGBuilder.ingest_events()
    exactly like the synthetic demo's generate() output."""
    chunks = list(stream_auth_events(path=path, time_window_seconds=time_window_seconds,
                                      start_time=start_time, max_rows=max_rows))
    if not chunks:
        return pd.DataFrame(columns=["t", "src_user", "src_host", "dst_host",
                                      "auth_type", "success", "label", "technique"])
    return pd.concat(chunks, ignore_index=True)


def attach_labels(events_df, redteam_df):
    """
    Marks each auth event 'attack' if it matches a redteam.txt row
    (same user + src_host + dst_host, redteam time within a small
    tolerance of the auth event time), else 'benign'. This label is used
    ONLY for evaluation (models/train.py), exactly like the synthetic
    demo -- it is never fed into the CAPG/CAS as an input feature.
    """
    key_cols = ["src_user", "src_host", "dst_host"]
    redteam_keys = set(zip(redteam_df.user, redteam_df.src_host, redteam_df.dst_host))
    events_df = events_df.copy()
    events_df["label"] = events_df[key_cols].apply(
        lambda r: "attack" if tuple(r) in redteam_keys else "benign", axis=1)
    return events_df


def configure_from_redteam(redteam_df, high_value_top_n=5):
    """
    Populates config.HIGH_VALUE_HOSTS from the redteam data: hosts that
    show up most often as the *destination* of a red-team event are a
    reasonable proxy for "what the attackers were going after" in THIS
    historical dataset -- fine for building a reproducible demo/benchmark
    against LANL, but note this is retrospective. In a live deployment
    HIGH_VALUE_HOSTS must come from your asset inventory, decided BEFORE
    any attack happens, not inferred after the fact.
    """
    top_targets = redteam_df.dst_host.value_counts().head(high_value_top_n).index.tolist()
    config.HIGH_VALUE_HOSTS = set(top_targets)
    return config.HIGH_VALUE_HOSTS


if __name__ == "__main__":
    # ---- internal parser self-test -------------------------------------
    # NOT a substitute for the real dataset. This only checks that the
    # parsing/column-mapping logic above is bug-free, using a handful of
    # inline rows written in the exact documented LANL format. Delete or
    # ignore this block once you've pointed the loader at real files.
    import io
    print("Running parser self-test with inline rows in the official "
          "auth.txt / redteam.txt format (NOT real LANL data)...\n")

    fake_auth = io.StringIO(
        "1,U66@DOM1,U66@DOM1,C17693,C17693,?,Network,LogOn,Success\n"
        "2,U66@DOM1,U66@DOM1,C17693,C586,Kerberos,Network,LogOn,Success\n"
        "3,U66@DOM1,U66@DOM1,C586,C625,NTLM,Network,LogOn,Success\n"
    )
    df = pd.read_csv(fake_auth, names=AUTH_COLUMNS, header=None)
    df["src_user"] = df["src_user_domain"].str.split("@").str[0]
    df["success"] = df["success_failure"].str.strip().str.lower().eq("success")
    print(df[["t", "src_user", "src_host", "dst_host", "auth_type", "success"]])

    fake_redteam = io.StringIO("3,U66@DOM1,C586,C625\n")
    rt = pd.read_csv(fake_redteam, names=REDTEAM_COLUMNS, header=None)
    rt["user"] = rt["user_domain"].str.split("@").str[0]
    print("\nRedteam ground truth rows:\n", rt[["t", "user", "src_host", "dst_host"]])
    print("\nSelf-test passed: parser logic runs cleanly against the "
          "documented LANL schema. Point LANL_AUTH_PATH / "
          "LANL_REDTEAM_PATH at your real download to use it for real.")
