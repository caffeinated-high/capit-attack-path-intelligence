"""
Unified entry point: runs the full 8-stage HCAPO pipeline against
ANY of the datasets from the handbook's Part 6, via one --dataset flag.
This is the single place that ties every data/*.py loader to the
shared pipeline_core.run_all_stages().

    python3 run_pipeline.py --dataset synthetic
    python3 run_pipeline.py --dataset lanl
    python3 run_pipeline.py --dataset optc
    python3 run_pipeline.py --dataset darpa_tc
    python3 run_pipeline.py --dataset cicids2017   --path /path/to/file.csv
    python3 run_pipeline.py --dataset cicids2018   --path /path/to/file.csv
    python3 run_pipeline.py --dataset unsw_nb15    --path /path/to/file.csv
    python3 run_pipeline.py --dataset ctu13        --path /path/to/file.csv
    python3 run_pipeline.py --dataset edge_iiotset --path /path/to/file.csv

For lanl/optc/darpa_tc, set the paths inside their respective loader
modules first (see each module's docstring), OR pass --path /
--path2 to override them for this run without editing the file.

Every dataset ultimately funnels into pipeline_core.run_all_stages(),
which is the ONLY place Stages 1-8 are implemented -- see that file
for the actual stage logic.
"""
import argparse
import sys

import config
from pipeline_core import run_all_stages

DATASET_LINKS = {
    "synthetic": "N/A -- generated locally, no download needed",
    "lanl": "https://csr.lanl.gov/data/cyber1/",
    "optc": "https://github.com/FiveDirections/OpTC-data",
    "darpa_tc": "https://github.com/darpa-i2o/Transparent-Computing",
    "cicids2017": "http://cicresearch.ca/CICDataset/CIC-IDS-2017/",
    "cicids2018": "https://www.unb.ca/cic/datasets/ids-2018.html "
                  "(aws s3 sync --no-sign-request s3://cse-cic-ids2018/ <dest-dir>)",
    "unsw_nb15": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
    "ctu13": "https://www.stratosphereips.org/datasets-ctu13",
    "edge_iiotset": "https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-"
                    "realistic-cyber-security-dataset-iot-and-iiot-applications",
}

FLOW_DATASETS = {"cicids2017", "cicids2018", "unsw_nb15", "ctu13", "edge_iiotset"}


def load_synthetic():
    from data.generate_synthetic_logs import generate
    from models.train import ground_truth_compromised_set
    events_df = generate()
    return events_df, ground_truth_compromised_set(events_df)


def load_lanl(args):
    from data import lanl_loader
    if args.path:
        lanl_loader.LANL_AUTH_PATH = args.path
    if args.path2:
        lanl_loader.LANL_REDTEAM_PATH = args.path2
    redteam_df = lanl_loader.load_redteam()
    lanl_loader.configure_from_redteam(redteam_df, high_value_top_n=5)
    events_df = lanl_loader.load_auth_window(
        time_window_seconds=args.time_window, start_time=args.start_time, max_rows=args.max_rows)
    events_df = lanl_loader.attach_labels(events_df, redteam_df)
    return events_df, lanl_loader.redteam_ground_truth_set(redteam_df)


def load_optc(args):
    from data import optc_loader
    if args.path:
        optc_loader.OPTC_DATA_PATH = args.path
    if not args.ground_truth_hosts:
        print("WARNING: no --ground-truth-hosts given for OpTC -- transcribe the "
              "compromised hostnames from the engagement's ground-truth document "
              "(see https://github.com/FiveDirections/OpTC-data) and pass them "
              "comma-separated via --ground-truth-hosts. Continuing with an empty "
              "ground-truth set (Stage 4's GNN comparison will be skipped).")
    gt_hosts = set(args.ground_truth_hosts.split(",")) if args.ground_truth_hosts else set()
    events_df = optc_loader.stream_flow_events(max_rows=args.max_rows)
    events_df = optc_loader.attach_ground_truth_labels(events_df, gt_hosts)
    optc_loader.configure_from_ground_truth(gt_hosts)
    return events_df, optc_loader.ground_truth_compromised_set(events_df)


def load_darpa_tc(args):
    from data import darpa_tc_loader
    if args.path:
        darpa_tc_loader.DARPA_TC_DATA_PATH = args.path
    if not args.ground_truth_hosts:
        print("WARNING: no --ground-truth-hosts given for DARPA TC -- transcribe "
              "the compromised host IDs / remote IPs from your engagement's "
              "ground-truth writeup and pass them comma-separated via "
              "--ground-truth-hosts. Continuing with an empty ground-truth set.")
    gt = set(args.ground_truth_hosts.split(",")) if args.ground_truth_hosts else set()
    events_df = darpa_tc_loader.stream_network_events(max_rows=args.max_rows)
    events_df = darpa_tc_loader.attach_ground_truth_labels(events_df, gt)
    darpa_tc_loader.configure_from_ground_truth(gt)
    return events_df, darpa_tc_loader.ground_truth_compromised_set(events_df)


def load_flow_dataset(args):
    from data import flow_loader
    if not args.path:
        raise SystemExit(f"--path is required for --dataset {args.dataset}. "
                          f"Download it from: {DATASET_LINKS[args.dataset]}")
    events_df = flow_loader.load_flow_csv(args.path, dataset=args.dataset, nrows=args.max_rows)
    flow_loader.configure_from_flow_events(events_df, high_value_top_n=5)
    from models.train import ground_truth_compromised_set
    return events_df, ground_truth_compromised_set(events_df)


LOADERS = {
    "synthetic": lambda args: load_synthetic(),
    "lanl": load_lanl,
    "optc": load_optc,
    "darpa_tc": load_darpa_tc,
}
for _flow_name in FLOW_DATASETS:
    LOADERS[_flow_name] = load_flow_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(LOADERS.keys()))
    p.add_argument("--path", default=None, help="Primary data file/path override (meaning depends on dataset)")
    p.add_argument("--path2", default=None, help="Secondary path override (e.g. LANL's redteam.txt)")
    p.add_argument("--ground-truth-hosts", default=None,
                   help="Comma-separated host/IP ids for datasets whose ground truth "
                        "isn't a clean CSV (optc, darpa_tc) -- see each loader's docstring")
    p.add_argument("--time-window", type=int, default=3600, help="LANL only: seconds of traffic to load")
    p.add_argument("--start-time", type=int, default=0, help="LANL only: window start time")
    p.add_argument("--max-rows", type=int, default=200_000, help="Row/event cap for any streaming loader")
    p.add_argument("--k-paths", type=int, default=5, help="Stage 5: how many candidate paths to rank")
    p.add_argument("--budget", type=float, default=5.0, help="Stage 6: optimizer's intervention budget")
    return p.parse_args()


def main():
    args = parse_args()
    config.reset()

    print(f"Dataset: {args.dataset}   (source: {DATASET_LINKS[args.dataset]})")
    events_df, ground_truth = LOADERS[args.dataset](args)
    print(f"Loaded {len(events_df)} events, {len(ground_truth)} ground-truth compromised nodes.\n")

    builder, record = run_all_stages(events_df, ground_truth, dataset_name=args.dataset,
                                      k_paths=args.k_paths, budget=args.budget)
    return 0 if builder is not None else 1


if __name__ == "__main__":
    sys.exit(main())
