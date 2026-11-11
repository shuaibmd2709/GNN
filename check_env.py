import torch, torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

print("torch:", torch.__version__)
print("pyg:", torch_geometric.__version__)
print("cuda:", torch.cuda.is_available())

x = torch.randn(5, 4)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
out = GCNConv(4, 8)(x, edge_index)
print("GCNConv out:", out.shape)  # expect torch.Size([5, 8])