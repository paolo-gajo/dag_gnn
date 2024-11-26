# %%
import os.path as osp

import torch
torch.set_printoptions(threshold=100000)
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9, Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(
        num_val=0,
        num_test=0,
        is_undirected=True,
        add_negative_train_samples=False
        ),
])

path = '/home/pgajo/pyg/src/gnn/tutorial/data/'
dataset = Planetoid(path, name = 'Cora', transform=transform)
# dataset = QM9(path)

# %%
print(dataset)

# %%
print(dataset[0])

# %%
print(dataset[0].edge_label_index.shape)

# %%
print(dataset[0][0].edge_index.shape)

# %%
max(dataset[0][0].edge_label)

# %%



