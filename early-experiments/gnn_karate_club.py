import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv

# 1. Load the dataset — a tiny real graph, built into PyG
dataset = KarateClub()
data = dataset[0]

print("Graph summary:")
print(data)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of classes:", dataset.num_classes)

# 2. Define a simple 2-layer Graph Neural Network
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 3. Move everything to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = GCN().to(device)
data = data.to(device)

# 4. Set up training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. Train for 100 steps
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# 6. Check accuracy on the whole graph
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
correct = (pred == data.y).sum()
acc = int(correct) / int(data.num_nodes)
print(f"\nFinal accuracy on full graph: {acc:.2%}")