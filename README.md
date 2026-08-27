# CAPIT / HCAPO — Cognitive Attack Path Intelligence

Enterprise cyber defense research project using Graph Neural Networks
(GNNs) to model, predict, and disrupt multi-stage attacker behavior —
lateral movement, privilege escalation, and pivoting toward high-value
targets — across an enterprise network.

## Project journey

This repo started as a learning path into GNNs for security:
1. **`early-experiments/gnn_karate_club.py`** — first working GNN, on the
   classic Zachary's Karate Club toy graph.
2. **`early-experiments/gnn_custom_network.py`** — a hand-built toy
   network graph with a GNN run on it.
3. **Everything below** — the full result: a working, runnable 8-stage
   pipeline (CAPIT/HCAPO) that goes from raw enterprise logs, to a live
   typed attack graph, to GNN-based risk scoring, to an actual
   optimized defensive response — tested against both synthetic data
   and the real LANL cyber-security dataset.

## What this project does

In a compromised network, attackers move quietly between machines and
accounts rather than acting immediately — lateral movement. This
project represents the network as a graph (accounts/computers as
nodes, relationships as edges), builds a structured model of each
identity's inferred intent (the *Cognitive Attack State*), trains three
independent GNN architectures to score compromise risk, finds the
attacker's cheapest path to a high-value target, and computes the
minimal set of defensive actions that would stop it — then feeds the
outcome back into the model.

## Quick start

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
python run_pipeline.py --dataset synthetic
```

That's the only command you need for the demo. It runs all 8 stages on
generated synthetic data (no downloads required), and produces:
- a plain-text log file
- a static graph image
- an interactive graph
- a self-contained `dashboard.html` — open this in a browser for a
  plain-language summary of what happened, no code-reading required

See **[Running the pipeline](#running-the-pipeline)** below for real
datasets, and **[Project layout](#project-layout)** for what every file
does.

## Verified on real data

This isn't just synthetic. Running against a 1-hour window of the real
**LANL "Comprehensive, Multi-Source Cyber-Security Events"** dataset
(8,978 nodes):

| Metric | Result |
|---|---|
| Attacker identity flagged (Stage 4) | `ANONYMOUS LOGON` — a well-known suspicious unauthenticated-access artifact |
| Candidate attack paths found (Stage 5) | 5 |
| Paths covered by chosen defenses (Stage 6) | 5 of 5 (100%), within budget |
| Defense actions proposed (Stage 7) | 4, all minimal host-level credential revokes |
| CAPG edges updated by feedback (Stage 8) | 21 |

The system flagged this identity as the top-priority attacker entirely
on its own, from raw traffic, without being told what to look for.

## Running the pipeline

### On the synthetic demo (no download needed)
```bash
python run_pipeline.py --dataset synthetic
```

### On real datasets
No dataset is bundled. Each loader tells you exactly what to download
and where to set the path if you haven't yet.

| Dataset | Official download | Command |
|---|---|---|
| **LANL** auth logs | https://csr.lanl.gov/data/cyber1/ | `python run_pipeline.py --dataset lanl --path auth.txt.gz --path2 redteam.txt.gz` |
| **DARPA OpTC** | https://github.com/FiveDirections/OpTC-data | `python run_pipeline.py --dataset optc --path ecar_file.json --ground-truth-hosts "HOST1,HOST2"` |
| **DARPA Transparent Computing** | https://github.com/darpa-i2o/Transparent-Computing | `python run_pipeline.py --dataset darpa_tc --path events.json --ground-truth-hosts "host-id-1"` |
| **CICIDS2017** | http://cicresearch.ca/CICDataset/CIC-IDS-2017/ | `python run_pipeline.py --dataset cicids2017 --path file.csv` |
| **CSE-CIC-IDS2018** | https://www.unb.ca/cic/datasets/ids-2018.html | `python run_pipeline.py --dataset cicids2018 --path file.csv` |
| **UNSW-NB15** | https://research.unsw.edu.au/projects/unsw-nb15-dataset | `python run_pipeline.py --dataset unsw_nb15 --path file.csv` |
| **CTU-13** | https://www.stratosphereips.org/datasets-ctu13 | `python run_pipeline.py --dataset ctu13 --path file.csv` |
| **Edge-IIoTset** | IEEE DataPort / Kaggle `mohamedamineferrag/edgeiiotset-...` | `python run_pipeline.py --dataset edge_iiotset --path file.csv` |

Useful flags: `--max-rows N` (cap how much loads — important for
LANL/OpTC), `--time-window SECONDS --start-time SECONDS` (LANL only),
`--k-paths N`, `--budget N`.

## The 8 HCAPO stages

Every dataset — synthetic or real — is normalized to one common event
schema and run through the exact same pipeline code:

| Stage | What it does |
|---|---|
| 1. Enterprise Situation Understanding | Ingest raw logs, normalize into a common schema |
| 2. Cognitive Attack State (CAS) Construction | Per-identity profile: inferred intent, objectives, privileges, risk |
| 3. CAPG Construction | Build the live typed graph of hosts, identities, vulnerabilities |
| 4. Attack Intent Inference | Flag the likely attacker; cross-check with GCN/GAT/GraphSAGE |
| 5. Attack Path Reasoning | Find the attacker's cheapest paths to their inferred objective |
| 6. Hierarchical Optimization | Pick the minimal set of interventions that breaks every path |
| 7. Autonomous Defense Planning | Turn interventions into (simulated) defense actions |
| 8. Cognitive Knowledge Evolution | Feed outcomes back into the graph, raising the attacker's cost |

## Project layout

```
early-experiments/            The GNN learning path this project grew out of
  gnn_karate_club.py
  gnn_custom_network.py

config.py                     Central "which hosts matter" config, shared by every stage
pipeline_core.py              The 8-stage pipeline logic -- every entry point calls this
run_pipeline.py                Unified CLI -- THE command to run
main.py, main_lanl.py         Older single-dataset entry points, kept for reference only

utils/logging_setup.py        Log-file writer + structured RunRecord (feeds the dashboard)
visualization/graph_viz.py    CAPG -> static PNG + interactive HTML (auto-focused on large graphs)
visualization/dashboard.py    Combines everything into one self-contained dashboard.html
visualization/vendor/         Vendored vis-network.js -- dashboard works fully offline, no CDN

data/generate_synthetic_logs.py   Synthetic demo data source
data/lanl_loader.py                LANL auth-log loader
data/optc_loader.py                DARPA OpTC loader
data/darpa_tc_loader.py            DARPA Transparent Computing loader
data/flow_loader.py                One engine, 5 presets: CICIDS2017/2018, UNSW-NB15, CTU-13, Edge-IIoTset

capg/graph.py                  Stage 1+3: the CAPG graph builder
cas/cas_builder.py             Stage 2: Cognitive Attack State construction
models/gnn.py                  GCN / GAT / GraphSAGE -- hand-written in PyTorch, sparse (edge-list) computation
models/train.py                Stage 4: trains + compares all 3 GNNs
reasoning/path_reasoning.py    Stage 5: cheapest-path attack reasoning
optimization/hierarchical_optimizer.py   Stage 6: greedy intervention selection
defense/defense_planner.py     Stage 7: simulated defense action planning
knowledge_evolution.py         Stage 8: feedback loop into the CAPG

logs/, outputs/                (generated at runtime, gitignored)
```

## Engineering notes

A few real problems came up moving from the small synthetic demo to
the real ~9,000-node LANL graph, each fixed along the way:
- **GNN training scalability** — the original dense-matrix GAT layer
  became a memory/compute bottleneck at real scale; rewrote all three
  GNN layers to use sparse, edge-list-based computation.
- **Unreadable large-graph rendering** — auto-switches to a focused
  subgraph view (flagged nodes + direct connections) once a graph
  exceeds 300 nodes.
- **Offline dashboard** — the graph-rendering library is vendored and
  embedded directly in the HTML, so it works without any CDN/internet
  dependency.
- **Cross-platform file encoding** — every file write explicitly uses
  UTF-8 to avoid Windows-only encoding crashes.

## What this does and does not prove

**It does show:**
- A working, inspectable implementation of every HCAPO stage that runs
  in seconds on synthetic data and scales to real multi-thousand-node
  graphs.
- That a rule-based heuristic *and* three independent GNN architectures
  converge on the correct attacker identity without being told the
  answer, on both synthetic and real data.
- That the optimizer correctly prefers cheap, narrow interventions over
  disruptive ones, and that the feedback loop measurably raises the
  attacker's cost on the next cycle.

**It does not prove:**
- Full generalization to production enterprise traffic — the real-data
  result above is one identity, one time window, assessed
  qualitatively, not yet checked against official ground truth for
  that specific slice.
- That the rule-based CAS heuristic is production-ready on its own —
  it still produces false positives, which is exactly why the learned
  GNN branch exists as a corroborating second signal.
- Full parity with each dataset's richest signal (e.g. OpTC/DARPA TC's
  full process-provenance graphs) — these are documented extension
  points, not silently-dropped functionality.

## Extending toward Phase 3

- Replace the rule-based CAS heuristic with a trained sequence/attention
  intent classifier.
- Add temporal decay to the GAT layer so the CAPG's evolution over time
  is modeled directly, not just a static snapshot.
- Replace the greedy Stage-6 optimizer with a CVXPY-based bi-level
  program for true multi-objective trade-offs.
- Extend the CAPG schema with process-provenance edges to unlock
  OpTC's/DARPA TC's full event streams.
- Build honest multi-campaign train/test splits (train on LANL,
  evaluate on OpTC) instead of single-dataset evaluation.
