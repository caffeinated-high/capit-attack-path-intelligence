"""
Turns a CAPGBuilder's graph into something a human can actually look
at, two ways:

  1. export_graph_png()  -- a static image (matplotlib), good for
     pasting into a report/slide/paper, or handing to someone who just
     wants "the graph" as a picture.
  2. export_graph_html() -- a single self-contained, zero-server HTML
     file (uses the vis-network JS library from a CDN) that you can
     double-click and open in any browser: pan, zoom, drag nodes,
     hover for details.

============================================================================
LARGE-GRAPH HANDLING (read this if your graph looked like a black blob,
or the interactive HTML appeared blank/frozen)
============================================================================
On a small graph (the ~70-node synthetic demo) rendering every node is
fine. On a real dataset (e.g. a LANL window with ~9,000 nodes and tens
of thousands of edges), drawing ALL of them:
  - makes the PNG an unreadable solid black mass of overlapping edges
  - makes the browser's physics-simulation layout in the interactive
    HTML take a very long time (or effectively never finish), which
    looks exactly like "the graph isn't there" / a blank/frozen page

The fix: once a graph exceeds MAX_RENDER_NODES, both export functions
automatically switch to rendering a FOCUSED subgraph instead of the
whole thing -- every ground-truth/flagged (`highlight_nodes`) node,
plus their direct (1-hop) neighbors, padded out with a few extra
high-degree nodes for context if there's budget left. This is the
actually-useful view anyway: nobody can read meaning out of 9,000
overlapping dots, but "the attacker's node and everything directly
connected to it" is exactly what you want to look at.

Small graphs are rendered in full, unchanged -- nothing about the
70-node synthetic demo's behavior changes.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")   # no display needed -- we only ever save to a file
import matplotlib.pyplot as plt
import networkx as nx

NODE_COLORS = {
    "identity": "#4C78A8",
    "host": "#9AA5B1",
    "vuln": "#F58518",
    "mission_target": "#E4572E",
}
ATTACKER_EDGE_TYPES = {"escalate", "exploit", "pivot"}

# The vis-network JS library is vendored locally (visualization/vendor/
# vis-network.min.js) and embedded INLINE into every generated HTML
# file, rather than loaded from a CDN <script src="..."> tag. Some
# networks (corporate/campus firewalls, some antivirus web filters)
# block CDN domains like cdnjs.cloudflare.com; when that happens a
# CDN-based graph silently never draws (the page looks blank except
# for the header/legend, because the `vis` object the code needs never
# gets defined). Embedding the library's actual code removes that
# external dependency entirely -- the generated HTML works completely
# offline, on any network, forever.
_VENDOR_JS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "vis-network.min.js")


def _load_vis_network_js():
    with open(_VENDOR_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()

# Once the graph has more nodes than this, switch to rendering a
# focused subgraph instead of everything. 300 is small enough to stay
# readable as a static PNG and fast in the browser's physics layout.
MAX_RENDER_NODES = 300


def _node_color(node_type):
    return NODE_COLORS.get(node_type, "#9AA5B1")


def _collapse_to_simple(g):
    """Collapses the CAPG's MultiDiGraph down to a simple DiGraph,
    keeping the minimum-cost edge between any two nodes (an attacker
    always prefers the cheapest available technique between two
    footholds) -- shared by both export functions below."""
    g_simple = nx.DiGraph()
    for u, v, data in g.edges(data=True):
        w = data["weight"]
        if not g_simple.has_edge(u, v) or g_simple[u][v]["weight"] > w:
            g_simple.add_edge(u, v, **data)
    g_simple.add_nodes_from(g.nodes(data=True))
    return g_simple


def _build_focus_view(g_simple, highlight_nodes, max_nodes=MAX_RENDER_NODES):
    """Returns (view_graph, was_filtered). See the module docstring's
    "LARGE-GRAPH HANDLING" section for why this exists."""
    if g_simple.number_of_nodes() <= max_nodes:
        return g_simple, False

    highlight_nodes = set(highlight_nodes) & set(g_simple.nodes())
    undirected = g_simple.to_undirected()

    keep = set(highlight_nodes)
    for n in highlight_nodes:
        keep.update(undirected.neighbors(n))

    if len(keep) > max_nodes:
        # Even at 1-hop there's too much to show -- keep every
        # highlighted node (never drop those) and fill the remaining
        # budget with their neighbors.
        keep = set(highlight_nodes)
        for n in highlight_nodes:
            for nb in undirected.neighbors(n):
                if len(keep) >= max_nodes:
                    break
                keep.add(nb)
    else:
        # Room to spare -- pad with the highest-degree remaining nodes
        # purely for extra visual context.
        remaining_budget = max_nodes - len(keep)
        others = sorted((n for n in g_simple.nodes() if n not in keep),
                         key=lambda n: g_simple.degree(n), reverse=True)
        keep.update(others[:remaining_budget])

    return g_simple.subgraph(keep).copy(), True


def export_graph_png(capg_builder, path, highlight_nodes=None, title="Cognitive Attack Path Graph",
                      max_nodes=MAX_RENDER_NODES):
    """Static PNG render via matplotlib + networkx's spring layout."""
    highlight_nodes = highlight_nodes or set()
    g_simple = _collapse_to_simple(capg_builder.g)
    total_nodes = g_simple.number_of_nodes()
    view, was_filtered = _build_focus_view(g_simple, highlight_nodes, max_nodes)

    pos = nx.spring_layout(view, seed=7, k=0.8)

    node_colors, node_sizes, edge_colors = [], [], []
    for n, data in view.nodes(data=True):
        node_colors.append(_node_color(data.get("type", "host")))
        node_sizes.append(650 if n in highlight_nodes else 220)
    for u, v, data in view.edges(data=True):
        edge_colors.append("#D62728" if data.get("edge_type") in ATTACKER_EDGE_TYPES else "#C7C7C7")

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(view, pos, edge_color=edge_colors, arrows=True,
                            arrowsize=10, width=1.0, alpha=0.6, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(view, pos, node_color=node_colors, node_size=node_sizes,
                            edgecolors=["#B30000" if n in highlight_nodes else "none" for n in view.nodes()],
                            linewidths=2.0)
    # Only label highlighted nodes (plus, on small views, higher-degree
    # ones too) -- labeling everything on a several-hundred-node focus
    # view still turns into unreadable clutter otherwise.
    if was_filtered:
        labels = {n: n for n in view.nodes() if n in highlight_nodes}
    else:
        labels = {n: n for n in view.nodes() if n in highlight_nodes or view.degree(n) > 2}
    nx.draw_networkx_labels(view, pos, labels=labels, font_size=8)

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=t,
                                  markerfacecolor=c, markersize=10) for t, c in NODE_COLORS.items()]
    plt.legend(handles=legend_handles, loc="upper left", frameon=False)

    full_title = title
    if was_filtered:
        full_title += (f"\n(showing a focused view: {view.number_of_nodes()} of "
                        f"{total_nodes} total nodes -- ground-truth/flagged nodes "
                        f"+ their direct connections)")
    plt.title(full_title, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _graph_to_vis_json(capg_builder, highlight_nodes=None, max_nodes=MAX_RENDER_NODES):
    """Builds the {nodes:[...], edges:[...]} structure vis-network
    expects, plus a `filtered` flag/counts so the caller (dashboard.py
    or export_graph_html below) can show a "focused view" notice.
    Shared by export_graph_html() and the dashboard so the graph only
    has to be serialized once."""
    highlight_nodes = highlight_nodes or set()
    g_simple = _collapse_to_simple(capg_builder.g)
    total_nodes = g_simple.number_of_nodes()
    view, was_filtered = _build_focus_view(g_simple, highlight_nodes, max_nodes)

    vis_nodes = []
    for n, data in view.nodes(data=True):
        vis_nodes.append({
            "id": n,
            "label": n,
            "color": _node_color(data.get("type", "host")),
            "shape": "star" if n in highlight_nodes else "dot",
            "size": 26 if n in highlight_nodes else 12,
            "borderWidth": 3 if n in highlight_nodes else 1,
            "title": f"type: {data.get('type')}\\nid: {n}" + (
                "\\n\u26a0 ground-truth compromised" if n in highlight_nodes else ""),
        })

    vis_edges = []
    for u, v, data in view.edges(data=True):
        is_attacker = data.get("edge_type") in ATTACKER_EDGE_TYPES
        vis_edges.append({
            "from": u, "to": v,
            "arrows": "to",
            "color": {"color": "#D62728" if is_attacker else "#C7C7C7"},
            "width": 2.5 if is_attacker else 1,
            "title": f"{data.get('edge_type')} (weight={data.get('weight', 0):.2f})",
        })

    return {
        "nodes": vis_nodes,
        "edges": vis_edges,
        "was_filtered": was_filtered,
        "shown_nodes": view.number_of_nodes(),
        "total_nodes": total_nodes,
    }


_VIS_NETWORK_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<!-- vis-network is embedded inline below (not loaded from a CDN) so
     this file works fully offline and isn't affected by networks that
     block CDN domains -- see export_graph_html() and the module
     docstring for why. -->
_VIS_JS_SCRIPT_TAG_PLACEHOLDER_
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#0f1117; color:#e6e6e6; }}
  #header {{ padding: 14px 20px; background:#161925; border-bottom:1px solid #2a2f45; }}
  #header h1 {{ margin:0; font-size:18px; font-weight:600; }}
  #filtered-notice {{ padding: 6px 20px; font-size:12.5px; color:#f0ad4e; background:#2a230f; }}
  #legend {{ padding: 8px 20px; font-size:13px; color:#9aa0b4; }}
  #legend span {{ margin-right: 16px; }}
  .swatch {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }}
  #network {{ width: 100%; height: 85vh; }}
  #loading {{ position:fixed; top:50%; left:50%; transform:translate(-50%,-50%); color:#9aa0b4; font-size:14px; }}
</style>
</head>
<body>
  <div id="header"><h1>{title}</h1></div>
  {filtered_notice_html}
  <div id="legend">
    <span><i class="swatch" style="background:#4C78A8"></i>identity</span>
    <span><i class="swatch" style="background:#9AA5B1"></i>host</span>
    <span><i class="swatch" style="background:#F58518"></i>vuln</span>
    <span><i class="swatch" style="background:#E4572E"></i>mission_target</span>
    <span>&#9733; = ground-truth compromised &nbsp;|&nbsp; red edges = escalate/exploit/pivot</span>
  </div>
  <div id="loading">Loading graph...</div>
  <div id="network"></div>
  <script>
    const data = {{ nodes: new vis.DataSet({nodes_json}), edges: new vis.DataSet({edges_json}) }};
    const options = {{
      physics: {{
        stabilization: {{ iterations: 150, fit: true }},
        barnesHut: {{ gravitationalConstant: -4000, springLength: 120 }}
      }},
      interaction: {{ hover: true, tooltipDelay: 100 }},
      nodes: {{ font: {{ color: "#e6e6e6", size: 12 }} }},
    }};
    const network = new vis.Network(document.getElementById("network"), data, options);
    // Stop the physics simulation once it settles so a large graph
    // doesn't keep the browser tab busy indefinitely (this is what
    // previously made a large graph look "blank"/frozen -- it was
    // still computing a layout for thousands of nodes with no end).
    network.once("stabilizationIterationsDone", function () {{
      network.setOptions({{ physics: false }});
      document.getElementById("loading").style.display = "none";
    }});
  </script>
</body>
</html>
"""


def export_graph_html(capg_builder, path, highlight_nodes=None, title="Cognitive Attack Path Graph",
                       max_nodes=MAX_RENDER_NODES):
    """Self-contained, fully offline interactive HTML (vis-network's JS
    is embedded inline -- no CDN, no server, no build step needed)."""
    payload = _graph_to_vis_json(capg_builder, highlight_nodes, max_nodes)

    filtered_notice_html = ""
    if payload["was_filtered"]:
        filtered_notice_html = (
            f'<div id="filtered-notice">Showing a focused view: '
            f'{payload["shown_nodes"]} of {payload["total_nodes"]} total nodes '
            f'(ground-truth/flagged nodes + their direct connections). '
            f'The full graph was too large to render usefully in one view.</div>'
        )

    html = _VIS_NETWORK_HTML_TEMPLATE.format(
        title=title,
        filtered_notice_html=filtered_notice_html,
        nodes_json=json.dumps(payload["nodes"]),
        edges_json=json.dumps(payload["edges"]),
    )
    # Inject the vendored vis-network JS via a plain string replace
    # (NOT part of the .format() call above) -- the library's own code
    # contains countless literal { } characters that would collide with
    # .format()'s placeholder syntax if inserted any other way.
    html = html.replace("_VIS_JS_SCRIPT_TAG_PLACEHOLDER_",
                         f"<script>\n{_load_vis_network_js()}\n</script>")

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


if __name__ == "__main__":
    # ---- self-test on both a small graph and a large stress-test graph --
    import random
    from capg.graph import CAPGBuilder
    from data.generate_synthetic_logs import generate

    print("Small graph (synthetic demo, ~70 nodes) -- should render in full:")
    df = generate()
    builder = CAPGBuilder()
    builder.ingest_events(df)
    highlight = {"finance-db", "dc-01", "id:user07", "ws-17", "ws-09", "ws-22"}
    export_graph_png(builder, "/tmp/small_graph.png", highlight_nodes=highlight)
    payload = _graph_to_vis_json(builder, highlight_nodes=highlight)
    print(f"  total_nodes={payload['total_nodes']} shown_nodes={payload['shown_nodes']} "
          f"was_filtered={payload['was_filtered']}")

    print("\nLarge stress-test graph (~9000 nodes, NOT real data) -- should auto-filter:")
    random.seed(1)
    big_builder = CAPGBuilder()
    hosts = [f"C{i}" for i in range(9000)]
    for h in hosts:
        big_builder.add_node(h, "host")
    for h in hosts:
        for _ in range(5):
            big_builder.add_edge(h, random.choice(hosts), "trust", timestamp=0)
    big_highlight = set(random.sample(hosts, 20))
    export_graph_png(big_builder, "/tmp/large_graph.png", highlight_nodes=big_highlight,
                      title="CAPG -- large stress test")
    export_graph_html(big_builder, "/tmp/large_graph.html", highlight_nodes=big_highlight,
                       title="CAPG -- large stress test")
    big_payload = _graph_to_vis_json(big_builder, highlight_nodes=big_highlight)
    print(f"  total_nodes={big_payload['total_nodes']} shown_nodes={big_payload['shown_nodes']} "
          f"was_filtered={big_payload['was_filtered']}")
    print("\nWrote /tmp/small_graph.png, /tmp/large_graph.png, /tmp/large_graph.html")
