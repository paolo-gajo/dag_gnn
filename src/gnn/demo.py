import torch
import json
import numpy as np
import sys
sys.path.append('/home/pgajo/pyg/src')
from gnn.utils import (
    get_numpy_graph,
    numpy_to_df_graph,
    gd2df,
    get_edge_index,
    get_possible_triples,
    to_step_graph,
    get_reps,
    )
np.set_printoptions(linewidth=10000)

from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

json_path = '/home/pgajo/pyg/data/yamakata/efrc_ud.json'
with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

data = data[:20]

# get all node and edge labels
pos_tags = sorted({tag for line in data for tag in line['pos_tags']})
head_tags = sorted({tag for line in data for tag in line['head_tags']})

# Generate all possible triples
triples = get_possible_triples(pos_tags, head_tags)

dataset = []

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
embedder_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

for sample in tqdm(data, total=len(data)):
    npgraph = get_numpy_graph(sample,
                            return_dict=True,
                            reduce=0,
                            )
    df_graph = gd2df(npgraph)
    step_graph = to_step_graph(df_graph)
    if step_graph['edge_index'].shape[0] > 0:
        graph = Data()
        graph.x = get_reps(step_graph['sents'], tokenizer, embedder_model)
        graph.edge_index = torch.tensor(step_graph['edge_index'])
        dataset.append(graph)

train_data, intermediate = train_test_split(dataset, test_size=0.2)
val_data, test_data = train_test_split(intermediate, test_size=0.5)

# test_dataset = dataset[:10000]
# val_dataset = dataset[10000:20000]
# train_dataset = dataset[20000:]

# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

dataset = T.RandomLinkSplit(loader)

from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
device = torch.device('cuda')

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


model = Net(embedder_model.config.hidden_size, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    for train_sample in tqdm(train_data, total=len(train_data), desc='Training...'):
        x = train_sample.x.to(device)
        edges = train_sample.edge_index.to(device)
        z = model.encode(x, edges)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_sample.edge_index, num_nodes=train_sample.num_nodes,
            num_neg_samples=train_sample.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_sample.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_sample.edge_label,
            train_sample.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        total_loss += loss
        loss.backward()
        optimizer.step()
    return total_loss/len(train_data)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


best_val_auc = final_test_auc = 0
for epoch in range(1, 101):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print(f'Final Test: {final_test_auc:.4f}')

z = model.encode(test_data.x, test_data.edge_index)
final_edge_index = model.decode_all(z)