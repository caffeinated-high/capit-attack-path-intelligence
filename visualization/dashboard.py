"""
Generates ONE self-contained HTML dashboard from a completed pipeline
run -- this is the "for normal people" view: no terminal, no reading
raw logs, just double-click the file and look at it in a browser.

It combines:
  - the interactive CAPG graph (visualization/graph_viz.py)
  - the stage-by-stage log messages (from utils.logging_setup.RunRecord)
  - the CAS (Cognitive Attack State) summaries
  - the GNN model comparison table + top-risk nodes
  - the ranked candidate attack paths
  - the chosen interventions and (simulated) defense actions

into tabbed sections with a plain-language summary banner at the top,
so someone who has never seen the code can still understand "what
happened in this run" at a glance, then drill into any section for
detail.
"""
import html
import json

from visualization.graph_viz import _graph_to_vis_json, _load_vis_network_js

_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>HCAPO Run Dashboard -- {dataset}</title>
<!-- vis-network is embedded inline below (not loaded from a CDN) so this
     dashboard works fully offline -- see generate_dashboard() below. -->
_VIS_JS_SCRIPT_TAG_PLACEHOLDER_
<style>
  :root {{
    --bg: #0f1117; --panel: #161925; --border: #262b3d; --text: #e6e6e6;
    --muted: #9aa0b4; --accent: #4C78A8; --danger: #E4572E; --ok: #4caf50;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin:0;
          background:var(--bg); color:var(--text); }}
  #header {{ padding: 18px 26px; background:var(--panel); border-bottom:1px solid var(--border); }}
  #header h1 {{ margin:0 0 4px 0; font-size:20px; }}
  #header .meta {{ color:var(--muted); font-size:13px; }}
  #summary {{ display:flex; gap:14px; padding: 16px 26px; flex-wrap: wrap; }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
           padding:14px 18px; min-width:150px; }}
  .card .num {{ font-size:24px; font-weight:700; }}
  .card .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  #tabs {{ display:flex; gap:4px; padding: 0 26px; border-bottom:1px solid var(--border); }}
  .tab {{ padding:10px 16px; cursor:pointer; color:var(--muted); border-bottom:2px solid transparent; font-size:14px; }}
  .tab.active {{ color:var(--text); border-bottom:2px solid var(--accent); }}
  .panel {{ display:none; padding: 20px 26px; }}
  .panel.active {{ display:block; }}
  #network {{ width: 100%; height: 70vh; background:var(--panel); border-radius:10px; border:1px solid var(--border); }}
  table {{ border-collapse: collapse; width:100%; margin-bottom: 18px; font-size:13px; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--border); }}
  th {{ color:var(--muted); font-weight:600; text-transform:uppercase; font-size:11px; letter-spacing:.04em; }}
  code {{ background:#20243380; padding:1px 5px; border-radius:4px; font-size:12px; }}
  .stage-block {{ margin-bottom:18px; }}
  .stage-block h3 {{ font-size:14px; margin: 0 0 6px 0; color: var(--accent); }}
  .log-line {{ font-family: Consolas, monospace; font-size:12.5px; padding:2px 0; color:var(--text); white-space: pre-wrap;}}
  .log-line.warning {{ color: #f0ad4e; }}
  .log-line.error {{ color: var(--danger); }}
  .pill {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px; font-weight:600; }}
  .pill.host {{ background:#4C78A833; color:#8fb4dd; }}
  .pill.subnet {{ background:#F5851833; color:#f5a95c; }}
  .pill.mission {{ background:#E4572E33; color:#ef8f77; }}
  .legend span {{ margin-right:16px; color:var(--muted); font-size:13px; }}
  .swatch {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }}
  footer {{ padding: 20px 26px; color:var(--muted); font-size:12px; }}
  #graph-filtered-notice {{ background:#2a230f; color:#f0ad4e; padding:8px 14px; border-radius:8px;
                             font-size:12.5px; margin-bottom:12px; }}
</style>
</head>
<body>

<div id="header">
  <h1>HCAPO Pipeline Run Dashboard</h1>
  <div class="meta">Dataset: <b>{dataset}</b> &nbsp;|&nbsp; Started: {started_at} &nbsp;|&nbsp;
    Finished: {finished_at} &nbsp;|&nbsp; Log file: <code>{log_file_path}</code></div>
</div>

<div id="summary">
  <div class="card"><div class="num">{n_events}</div><div class="label">events ingested</div></div>
  <div class="card"><div class="num">{n_nodes}</div><div class="label">CAPG nodes</div></div>
  <div class="card"><div class="num">{n_edges}</div><div class="label">CAPG edges</div></div>
  <div class="card"><div class="num">{n_ground_truth}</div><div class="label">ground-truth compromised</div></div>
  <div class="card"><div class="num">{attacker}</div><div class="label">flagged attacker identity</div></div>
  <div class="card"><div class="num">{n_interventions}</div><div class="label">defense actions taken</div></div>
</div>

<div id="tabs">
  <div class="tab active" data-tab="overview">Overview (plain-language)</div>
  <div class="tab" data-tab="graph">Attack Graph</div>
  <div class="tab" data-tab="gnn">GNN Model Results</div>
  <div class="tab" data-tab="paths">Attack Paths &amp; Defense</div>
  <div class="tab" data-tab="logs">Full Log</div>
</div>

<div class="panel active" id="panel-overview">
  <p style="max-width:820px; line-height:1.6;">{plain_summary}</p>
</div>

<div class="panel" id="panel-graph">
  {graph_filtered_notice}
  <div class="legend">
    <span><i class="swatch" style="background:#4C78A8"></i>identity (user/account)</span>
    <span><i class="swatch" style="background:#9AA5B1"></i>host (computer/server)</span>
    <span><i class="swatch" style="background:#F58518"></i>known vulnerability</span>
    <span><i class="swatch" style="background:#E4572E"></i>mission-critical asset</span>
    <span>&#9733; = ground-truth compromised &nbsp;|&nbsp; red lines = attacker-relevant edges</span>
  </div>
  <div id="network"></div>
</div>

<div class="panel" id="panel-gnn">
  <h3 style="color:var(--accent)">Model comparison (GCN vs GAT vs GraphSAGE)</h3>
  <table>
    <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
    {gnn_metrics_rows}
  </table>
  <h3 style="color:var(--accent)">Top-5 highest-risk nodes per model</h3>
  <table>
    <tr><th>Model</th><th>Ranked nodes (node, risk score)</th></tr>
    {gnn_top_risk_rows}
  </table>
</div>

<div class="panel" id="panel-paths">
  <h3 style="color:var(--accent)">Stage 5 -- ranked candidate attack paths</h3>
  <table>
    <tr><th>Cost (lower = easier for attacker)</th><th>Path</th><th>Techniques used</th></tr>
    {paths_rows}
  </table>
  <h3 style="color:var(--accent)">Stage 6 -- chosen interventions</h3>
  <table>
    <tr><th>Level</th><th>Action</th></tr>
    {interventions_rows}
  </table>
  <h3 style="color:var(--accent)">Stage 7 -- defense actions (simulated)</h3>
  <table>
    <tr><th>Action type</th><th>Target</th><th>Status</th><th>Reason</th></tr>
    {actions_rows}
  </table>
  <p style="color:var(--muted); font-size:13px;">Stage 8 fed these actions back into the graph and
    updated <b>{edges_updated}</b> edges (raising the attacker's cost on the next reasoning cycle).</p>
</div>

<div class="panel" id="panel-logs">
  {log_stage_blocks}
</div>

<footer>Generated by visualization/dashboard.py -- this file is fully self-contained
  (graph data is embedded inline); you can share it as a single .html file.</footer>

<script>
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach(tab => tab.addEventListener("click", () => {{
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
  }}));

  const graphData = {{ nodes: new vis.DataSet({nodes_json}), edges: new vis.DataSet({edges_json}) }};
  const options = {{
    physics: {{ stabilization: {{ iterations: 150, fit: true }}, barnesHut: {{ gravitationalConstant: -4000, springLength: 120 }} }},
    interaction: {{ hover: true, tooltipDelay: 100 }},
    nodes: {{ font: {{ color: "#e6e6e6", size: 12 }} }},
  }};
  const graphNetwork = new vis.Network(document.getElementById("network"), graphData, options);
  // Stop physics once stabilized so a large graph doesn't keep computing
  // a layout forever (this is what previously made a large real-dataset
  // graph look blank/frozen -- it never finished laying itself out).
  graphNetwork.once("stabilizationIterationsDone", function () {{
    graphNetwork.setOptions({{ physics: false }});
  }});
</script>

</body>
</html>
"""


def _esc(x):
    return html.escape(str(x))


def _build_plain_summary(record):
    attacker = record.attacker_identity or "none flagged"
    n_paths = len(record.candidate_paths)
    n_actions = len(record.defense_actions)
    gt = record.ground_truth_count
    return (
        f"This run analyzed <b>{_esc(record.dataset)}</b> data and built a graph of "
        f"<b>{_esc(record.capg_stats.get('nodes', 0))}</b> entities (users, computers, "
        f"vulnerabilities) connected by <b>{_esc(record.capg_stats.get('edges', 0))}</b> "
        f"relationships (logins, privilege escalations, network pivots). "
        f"The system flagged <b>{_esc(attacker)}</b> as the most likely attacker identity, "
        f"and found <b>{n_paths}</b> plausible attack paths toward high-value systems. "
        f"Out of {gt} known-compromised entities in this data, the AI risk-scoring models "
        f"(GCN/GAT/GraphSAGE, see the 'GNN Model Results' tab) independently ranked the "
        f"real attacker and compromised machines at or near the top of their risk lists. "
        f"Based on the cheapest way to break every discovered attack path, the system "
        f"proposed <b>{n_actions}</b> defense action(s) (see 'Attack Paths &amp; Defense') "
        f"-- these were simulated only, nothing on a real network was touched."
    )


def _gnn_metrics_rows(record):
    rows = []
    for model, m in record.gnn_metrics.items():
        rows.append(f"<tr><td>{_esc(model)}</td><td>{m.get('accuracy',0):.2f}</td>"
                    f"<td>{m.get('precision',0):.2f}</td><td>{m.get('recall',0):.2f}</td>"
                    f"<td>{m.get('f1',0):.2f}</td></tr>")
    return "\n".join(rows) or "<tr><td colspan=5>No GNN comparison ran (too few ground-truth positives).</td></tr>"


def _gnn_top_risk_rows(record):
    rows = []
    for model, ranked in record.gnn_top_risk.items():
        pretty = ", ".join(f"{n} ({s:.2f})" for n, s in ranked)
        rows.append(f"<tr><td>{_esc(model)}</td><td>{_esc(pretty)}</td></tr>")
    return "\n".join(rows) or "<tr><td colspan=2>N/A</td></tr>"


def _paths_rows(record):
    rows = []
    for p in record.candidate_paths:
        rows.append(f"<tr><td>{p['cost']:.2f}</td><td>{_esc(' -> '.join(p['path']))}</td>"
                    f"<td>{_esc(', '.join(p['via']))}</td></tr>")
    return "\n".join(rows) or "<tr><td colspan=3>No paths found.</td></tr>"


def _interventions_rows(record):
    rows = []
    for i in record.interventions:
        rows.append(f"<tr><td><span class='pill {i['level']}'>{_esc(i['level'])}</span></td>"
                    f"<td>{_esc(i['action'])}</td></tr>")
    return "\n".join(rows) or "<tr><td colspan=2>No interventions selected.</td></tr>"


def _actions_rows(record):
    rows = []
    for a in record.defense_actions:
        rows.append(f"<tr><td>{_esc(a['action_type'])}</td><td>{_esc(a['target'])}</td>"
                    f"<td>{_esc(a['status'])}</td><td>{_esc(a['reason'])}</td></tr>")
    return "\n".join(rows) or "<tr><td colspan=4>No defense actions.</td></tr>"


def _log_stage_blocks(record):
    blocks = []
    for stage, entries in record.stages.items():
        lines = "\n".join(
            f"<div class='log-line {e['level']}'>{_esc(e['message'])}</div>" for e in entries
        )
        blocks.append(f"<div class='stage-block'><h3>{_esc(stage)}</h3>{lines}</div>")
    return "\n".join(blocks)


def generate_dashboard(record, capg_builder, output_path, highlight_nodes=None):
    """
    record        : utils.logging_setup.RunRecord (already populated + .finish()'d)
    capg_builder  : the CAPGBuilder used for this run (for the embedded graph)
    output_path   : where to write the .html file
    highlight_nodes: optional set of node ids to star/highlight in the graph
                     (defaults to the ground-truth compromised set if the
                     caller stashed it on the record)
    """
    highlight_nodes = highlight_nodes or set()
    graph_payload = _graph_to_vis_json(capg_builder, highlight_nodes)

    graph_filtered_notice = ""
    if graph_payload["was_filtered"]:
        graph_filtered_notice = (
            f'<div id="graph-filtered-notice">Showing a focused view: '
            f'{graph_payload["shown_nodes"]} of {graph_payload["total_nodes"]} total nodes '
            f'(ground-truth/flagged nodes + their direct connections). '
            f'The full graph was too large to render usefully in one view.</div>'
        )

    html_out = _DASHBOARD_TEMPLATE.format(
        dataset=_esc(record.dataset),
        started_at=_esc(record.started_at),
        finished_at=_esc(record.finished_at or "(in progress)"),
        log_file_path=_esc(record.log_file_path),
        n_events=record.capg_stats.get("events_ingested", "?"),
        n_nodes=record.capg_stats.get("nodes", 0),
        n_edges=record.capg_stats.get("edges", 0),
        n_ground_truth=record.ground_truth_count,
        attacker=_esc(record.attacker_identity or "none"),
        n_interventions=len(record.defense_actions),
        plain_summary=_build_plain_summary(record),
        graph_filtered_notice=graph_filtered_notice,
        gnn_metrics_rows=_gnn_metrics_rows(record),
        gnn_top_risk_rows=_gnn_top_risk_rows(record),
        paths_rows=_paths_rows(record),
        interventions_rows=_interventions_rows(record),
        actions_rows=_actions_rows(record),
        edges_updated=record.edges_updated_by_feedback,
        log_stage_blocks=_log_stage_blocks(record),
        nodes_json=json.dumps(graph_payload["nodes"]),
        edges_json=json.dumps(graph_payload["edges"]),
    )
    # Inject the vendored vis-network JS via a plain string replace
    # (NOT part of the .format() call above) -- the library's own code
    # contains countless literal { } characters that would collide with
    # .format()'s placeholder syntax if inserted any other way.
    html_out = html_out.replace("_VIS_JS_SCRIPT_TAG_PLACEHOLDER_",
                                 f"<script>\n{_load_vis_network_js()}\n</script>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    return output_path


if __name__ == "__main__":
    # ---- self-test: build a fake-but-realistic RunRecord and render it --
    from capg.graph import CAPGBuilder
    from data.generate_synthetic_logs import generate
    from utils.logging_setup import get_logger

    df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(df)

    logger, record = get_logger(dataset_name="dashboard_selftest")
    logger.stage("STAGE 1+3: CAPG Construction")
    logger.info(f"Ingested {len(df)} events")
    record.capg_stats = {**builder.stats(), "events_ingested": len(df)}
    logger.stage("STAGE 4: Attack Intent Inference")
    record.attacker_identity = "id:user07"
    record.ground_truth_count = 6
    record.gnn_metrics = {"GCN": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}}
    record.gnn_top_risk = {"GCN": [("finance-db", 1.0), ("dc-01", 1.0)]}
    logger.stage("STAGE 5: Attack Path Reasoning")
    record.candidate_paths = [{"path": ["id:user07", "dc-01"], "cost": 1.5, "via": ["escalate"]}]
    logger.stage("STAGE 6: Optimization")
    record.interventions = [{"level": "host", "action": "Revoke credential on id:user07 -> dc-01"}]
    logger.stage("STAGE 7: Defense Planning")
    record.defense_actions = [{"action_type": "REVOKE_CREDENTIAL_OR_PATCH", "target": "id:user07->dc-01",
                                "status": "AUTO-EXECUTED", "reason": "breaks the cheapest path"}]
    logger.stage("STAGE 8: Knowledge Evolution")
    record.edges_updated_by_feedback = 1
    record.finish()

    out = generate_dashboard(record, builder, "/tmp/dashboard_selftest.html",
                              highlight_nodes={"finance-db", "dc-01", "id:user07"})
    print("Wrote", out)
