"""
Central logging for the HCAPO pipeline.

Two things happen every run:
  1. A plain-text log file is written to logs/run_<timestamp>.log --
     this is the literal "log file" artifact you can hand to someone
     (a professor, a teammate) as evidence of what a run did.
  2. A RunRecord object accumulates the SAME information in structured
     form (stage name -> list of {level, message} entries, plus tables
     for CAPG stats / CAS summaries / GNN metrics / paths /
     interventions / actions) so visualization/dashboard.py can render
     it as a human-friendly HTML report afterward, instead of needing
     someone to read raw log text.

Usage:
    from utils.logging_setup import get_logger
    logger, record = get_logger(dataset_name="synthetic")
    logger.stage("STAGE 1: CAPG Construction")
    logger.info("Ingested 406 events")
    ...
    record.to_json("logs/run_....json")   # or just pass `record` to the dashboard generator
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


@dataclass
class RunRecord:
    """Structured record of one pipeline run, built up stage by stage.
    This is what visualization/dashboard.py actually renders -- it's
    the "data model" behind the human-friendly report."""
    dataset: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    stages: dict = field(default_factory=dict)          # stage name -> [ {level, message} ]
    capg_stats: dict = field(default_factory=dict)
    cas_summaries: list = field(default_factory=list)    # list of str
    attacker_identity: str = ""
    gnn_metrics: dict = field(default_factory=dict)       # model name -> metrics dict
    gnn_top_risk: dict = field(default_factory=dict)      # model name -> [(node, score), ...]
    ground_truth_count: int = 0
    candidate_paths: list = field(default_factory=list)   # [{"path": [...], "cost": .., "via": [...]}]
    interventions: list = field(default_factory=list)     # [{"level":..,"action":..}]
    defense_actions: list = field(default_factory=list)   # [{"action_type":..,"target":..,"status":..,"reason":..}]
    edges_updated_by_feedback: int = 0
    log_file_path: str = ""
    dashboard_path: str = ""

    def add(self, stage, level, message):
        self.stages.setdefault(stage, []).append({"level": level, "message": message})

    def finish(self):
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path


class StageLogger:
    """Thin wrapper around the stdlib logger that also mirrors every
    message into a RunRecord, and tags each message with "which stage"
    it belongs to (whatever the last stage(...) call set)."""

    def __init__(self, py_logger, record: RunRecord):
        self._log = py_logger
        self.record = record
        self._current_stage = "SETUP"

    def stage(self, title):
        self._current_stage = title
        self._log.info("\n" + "=" * 78)
        self._log.info(title)
        self._log.info("=" * 78)
        self.record.stages.setdefault(title, [])

    def info(self, message):
        self._log.info(message)
        self.record.add(self._current_stage, "info", message)

    def warning(self, message):
        self._log.warning(message)
        self.record.add(self._current_stage, "warning", message)

    def error(self, message):
        self._log.error(message)
        self.record.add(self._current_stage, "error", message)


def get_logger(dataset_name="synthetic", log_dir=LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{dataset_name}_{timestamp}.log")

    py_logger = logging.getLogger(f"hcapo.{dataset_name}.{timestamp}")
    py_logger.setLevel(logging.INFO)
    py_logger.handlers.clear()   # avoid duplicate handlers if called twice in one process
    py_logger.propagate = False

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    py_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    py_logger.addHandler(console_handler)

    record = RunRecord(dataset=dataset_name)
    record.log_file_path = log_path
    logger = StageLogger(py_logger, record)
    return logger, record


if __name__ == "__main__":
    logger, record = get_logger(dataset_name="selftest")
    logger.stage("STAGE 1: demo")
    logger.info("hello from the log file")
    logger.warning("this is a warning")
    record.finish()
    print(f"\nLog file written to: {record.log_file_path}")
    print("RunRecord stages:", list(record.stages.keys()))
