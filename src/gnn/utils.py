import torch
from typing import List
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData, Data
pd.set_option('display.max_rows', None)
from transformers import AutoTokenizer, AutoModel

def get_numpy_graph(sample, return_dict=True, reduce=False):
    """
    Constructs a numpy-based graph representation of a sample containing syntactic or dependency parsing information.

    Args:
    sample (dict): A dictionary containing the following keys:
        - 'head_indices': List of integers representing the head index of each word in the sentence.
        - 'pos_tags': List of strings representing part-of-speech tags for each word.
        - 'head_tags': List of strings representing dependency labels for each word.
        - 'step_indices': List of integers representing the sequential step indices for each word.
        - 'words': List of strings representing the words in the sentence.
    return_dict (bool): If True, returns the graph as a dictionary with column names as keys. 
                        If False, returns the graph as a numpy array.
    reduce (bool): If True, applies filtering to remove unused nodes, keeping only those that have a head or are referenced as heads by others.

    Returns:
    dict or numpy.ndarray: A structured representation of the graph, either as a dictionary (if `return_dict=True`) 
                           or a numpy array (if `return_dict=False`). The graph includes the following columns:
        - 'old_word_id': Original word indices before filtering.
        - 'word': Words in the sentence, including a root token.
        - 'head': The word's syntactic head.
        - 'node_type': POS tag or node type of the word.
        - 'edge_type': Dependency label or edge type of the word.
        - 'old_head_id': Original head indices before filtering.
        - 'step_id': Step indices corresponding to each word.
        - 'word_id': Updated word indices after filtering.
        - 'head_id': Updated head indices after filtering.
    """

    head_id = np.array(sample['head_indices'])
    node_type = np.array(sample['pos_tags'])
    edge_type = np.array(sample['head_tags'])
    step_id = np.array(sample['step_indices'])

    head_id = np.concatenate([[0], head_id])
    node_type = np.concatenate([['O'], node_type])
    edge_type = np.concatenate([['root'], edge_type])
    step_id = np.concatenate([[0], step_id])

    words = np.array(['root'] + sample['words'])
    heads = words[head_id]
    word_id = np.arange(words.shape[0])

    graph = np.stack([word_id, words, heads, node_type, edge_type, head_id, step_id], axis=1)

    if reduce:
        # make and apply masks
        has_in_edge = np.isin(word_id, head_id).astype(int)
        head_id_mask = head_id.astype(int) != 0
        has_in_edge_mask = has_in_edge.astype(int) == 1
        valid_mask = np.logical_or(head_id_mask, has_in_edge_mask)
        graph = graph[valid_mask]
    
    # reset word ids and remap head ids
    reset_word_id = np.arange(graph.shape[0]).reshape(-1, 1)
    id_map = {old_id: new_id for new_id, old_id in enumerate(graph[:, 0].astype(int))}
    new_word_ids = np.array([id_map.get(int(hid), -1) for hid in graph[:, 5]])

    # new_word_ids = np.array(new_word_ids)

    graph = np.hstack([graph, reset_word_id, new_word_ids.reshape(-1, 1)])

    if return_dict:
        gd = {}
        cols = [
            'old_word_id',
            'word',
            'head',
            'node_type',
            'edge_type',
            'old_head_id',
            'step_id',
            'word_id',
            'head_id',
        ]
        for i, col in enumerate(cols):
            gd[col] = graph[:, i]
        return gd
    else:
        return graph

def numpy_to_df_graph(numpy_graph):
    gd = {}

    gd['old_word_id'] = numpy_graph[:, 0]
    gd['has_in_edge'] = numpy_graph[:, 1]
    gd['word'] = numpy_graph[:, 2]
    gd['head'] = numpy_graph[:, 3]
    gd['node_type'] = numpy_graph[:, 4]
    gd['edge_type'] = numpy_graph[:, 5]
    gd['old_head_id'] = [int(el) for el in numpy_graph[:, 6]]
    gd['step_id'] = numpy_graph[:, 7]
    gd['word_id'] = numpy_graph[:, 8]
    gd['head_id'] = numpy_graph[:, 9]
    df_graph = pd.DataFrame(gd,
                            columns=[
                                'word_id',
                                # 'has_in_edge',
                                'word',
                                'node_type',
                                'edge_type',
                                # 'old_head_id',
                                'head',
                                # 'node_uuid',
                                # 'head_uuid',
                                'head_id',
                                'step_id',
                                ])
    return df_graph

def gd2df(graph_dict):
    df = pd.DataFrame(graph_dict, columns=graph_dict.keys())
    return df

# def get_edge_index(df_graph, edge_triple):
#     edge_index = np.stack([
#         df_graph['word_id'].astype(int),
#         df_graph['head_id'].astype(int),
#     ])
#     return edge_index

def get_edge_index(df_graph, edge_triple):
    """
    Get edge indices for a given edge triple from the DataFrame graph.

    Args:
        df_graph (pd.DataFrame): DataFrame representation of the graph.
        edge_triple (tuple): A triple in the form (source_node_type, edge_type, target_node_type).

    Returns:
        np.ndarray: Edge index in PyTorch Geometric format (2, num_edges).
    """
    src_type, edge_type, tgt_type = edge_triple
    
    # Filter the graph for matching source, edge, and target types
    filtered_df = df_graph[
        (df_graph['node_type'] == src_type) &
        (df_graph['edge_type'] == edge_type) &
        (df_graph['head_id'].isin(df_graph[df_graph['node_type'] == tgt_type]['word_id']))
    ]
    
    # Extract source and target node IDs
    source_ids = filtered_df['word_id'].to_numpy()
    target_ids = filtered_df['head_id'].to_numpy()
    
    # Create edge index array
    edge_index = np.stack([source_ids, target_ids], axis=0)
    return edge_index


def to_step_graph(graph: pd.DataFrame):
    step_ids = np.array(graph['step_id'].astype('int'))
    num_steps = max(step_ids)
    edges_pos = []
    all_edges = []
    sents = []
    # iterate over nodes to check
    for i in range(1, num_steps + 1):
        # iterate over other nodes
        df_src = graph[step_ids == i]
        sents.append(' '.join(list(df_src['word'])))
        nodes_src_word_id = np.array(df_src['word_id'].astype('int'))
        nodes_src_head_id = np.array(df_src['head_id'].astype('int'))
        for j in range(1, num_steps + 1):
            if i == j:
                continue
            df_tgt = graph[step_ids == j]
            nodes_tgt_word_id = np.array(df_tgt['word_id'].astype('int'))
            nodes_tgt_head_id = np.array(df_tgt['head_id'].astype('int'))
            tgt_points_src = np.any(np.isin(nodes_src_word_id, nodes_tgt_head_id))
            src_points_tgt = np.any(np.isin(nodes_tgt_word_id, nodes_src_head_id))
            if tgt_points_src or src_points_tgt:
                edges_pos.append((i, j))
            all_edges.append((i, j))
    edges_pos_set = list(set(tuple(sorted(t)) for t in edges_pos))
    
    # all_edges_set = set(tuple(sorted(t)) for t in all_edges)
    # edges_neg_set = all_edges_set.difference(edges_pos_set)
    if edges_pos_set:
        edge_index = np.array(edges_pos_set).T - 1
    else:
        edge_index = np.array(edges_pos_set)

    return {
        'sents': sents,
        'edge_index': edge_index,
        # 'edges_neg': edges_neg_set,s
    }

def get_possible_triples(pos_tags, head_tags):  
    """
    Generate all possible (source_node, edge_type, target_node) triples.

    Args:
        pos_tags (list): List of node types.
        head_tags (list): List of edge types.

    Returns:
        list: List of all possible triples.
    """
    triples = [(src, edge, tgt) for src in pos_tags for edge in head_tags for tgt in pos_tags]
    return triples

def to_homodata(graph):
    """
    Using this for step graphs, since they have no labels.
    """
    data = Data()
    return

def to_heterodata(graph, pos_tags, triples):
    """
    I could use this for full recipe graphs if needed.
    """
    data = HeteroData()
    # Add nodes to HeteroData
    for node_type in pos_tags:
        nodes = np.array(graph[graph['node_type'] == node_type]['node_type'])
        if len(nodes) > 0:
            data[node_type].node_id = nodes

    # Add edges to HeteroData for each triple
    for triple in triples:
        edge_index = get_edge_index(graph, triple)
        if edge_index.size > 0:  # Add edge only if there are valid connections
            src_type, edge_type, tgt_type = triple
            data[(src_type, edge_type, tgt_type)].edge_index = torch.from_numpy(edge_index.astype('int'))

    # Print final HeteroData structure
    print("Final HeteroData object:", data)

    print(data[(src_type, edge_type, tgt_type)])
    return

def get_reps(sequences, tokenizer, model):
    inputs = tokenizer(sequences, return_tensors = 'pt', padding = True)
    # only use the CLS token rep as pooling
    reps = model(**inputs).last_hidden_state[:, 0]
    return reps

def main():
    return

if __name__ == "__main__":
    main