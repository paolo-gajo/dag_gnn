import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np

# Create a directed tree graph
G = nx.DiGraph()
edges = [
    (1, 2), (2, 9), (3, 4), (4, 5), (5, 9), (9, 11), (6, 10), (7, 8), (8, 10), (10, 11),
]  # example edges for a tree structure
edges = sorted(edges, key=lambda x: x[-1])
G.add_edges_from(edges)

def get_subbranches(graph, return_dict = False):
    sub_branches = []
    leaf_list = [n for (n, d) in graph.in_degree if d == 0]
    fork_list = [n for (n, d) in graph.in_degree if d > 1]
    for leaf in leaf_list:
        for fork in fork_list:
            sub_branches.extend(list(nx.algorithms.all_simple_paths(graph, leaf, fork)))
    
    if return_dict:
        branch_dict = {i: [] for i in range(1, len(graph.nodes())+1)}
        for branch in sub_branches:
            branch_dict[branch[-1]].append(branch)
        return branch_dict
    else:
        return sub_branches

sub_branches = get_subbranches(G)
branch_dict = get_subbranches(G, return_dict=1)
print(sub_branches)
print(branch_dict)