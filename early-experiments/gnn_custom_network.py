import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv

# 1. Build a small "enterprise network" graph by hand
G = nx.Graph()

# Nodes: a mix of user accounts, hosts, and one server
nodes = [
    "user_arjun", "user_admin", "user_guest",
    "host_laptop1", "host_laptop2", "host_desktop1",
    "host_desktop2", "host_fileserver", "host_dbserver",
    "host_webserver", "host_backup", "host_printer",
]
G.add_nodes_from(nodes)

# Edges: who/what connects to what (login/connection events)
edges = [
    ("user_arjun", "host_laptop1"),
    ("user_admin", "host_desktop1"),
    ("user_guest", "host_laptop2"),
    ("host_laptop1", "host_fileserver"),
    ("host_desktop1", "host_fileserver"),
    ("host_desktop1", "host_dbserver"),   # admin reaching the database — high value
    ("host_fileserver", "host_backup"),
    ("host_desktop2", "host_webserver"),
    ("host_webserver", "host_dbserver"),  # webserver reaching database — classic attack path
    ("host_laptop2", "host_printer"),
    ("host_desktop2", "host_fileserver"),
]
G.add_edges_from(edges)

# 2. Label each node: 0 = normal, 1 = suspicious (marked by hand for this toy example)
# In a real system this would come from actual threat intel / red-team labels.
suspicious_nodes = {"host_desktop1", "host_webserver", "host_dbserver"}
labels = {node: (1 if node in suspicious_nodes else 0) for node in nodes}
nx.set_node_attributes(G, labels, "y")

# 3. Give every node a simple starting feature (just an identity vector for now)
for i, node in enumerate(nodes):
    G.nodes[node]["x"] = [1.0 if j == i else 0.0 for j in range(len(nodes))]

# 4. Visualize it
plt.figure(figsize=(8, 6))
colors = ["red" if labels[n] == 1 else "lightblue" for n in G.nodes]
nx.draw(G, with_labels=True, node_color=colors, node_size=1500, font_size=7)
plt.title("Toy Enterprise Network Graph (red = suspicious)")
plt.savefig("custom_network_graph.png", dpi=150, bbox_inches="tight")
print("Saved visualization to custom_network_graph.png")

# 5. Convert to a PyTorch Geometric graph
data = from_networkx(G, group_node_attrs=["x"])
data.y = torch.tensor([labels[n] for n in nodes], dtype=torch.long)

# Simple train mask: pretend we know the true label for about half the nodes
data.train_mask = torch.tensor(
    [True, True, False, True, False, True, False, True, True, False, False, True]
)

print("\nGraph summary:")
print(data)

# 6. Same GCN architecture as before, adapted to this graph
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 8)
        self.conv2 = GCNConv(8, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = GCN(num_features=data.num_features, num_classes=2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
print("\nPredictions per node:")
for node, p, true_label in zip(nodes, pred.tolist(), data.y.tolist()):
    flag = "SUSPICIOUS" if p == 1 else "normal"
    match = "✓" if p == true_label else "✗"
    print(f"  {node:16s} -> predicted: {flag:10s} (true: {true_label}) {match}")