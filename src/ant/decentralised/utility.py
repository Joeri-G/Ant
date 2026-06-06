from ..market import Market, BaseAgent
import cvxpy as cp
import numpy as np
import networkx as nx
from typing import List, Tuple, Set

def k_highest_indices(lst, k):
    # Sort (index, value) pairs by value descending, take top k indices
    return [i for i, v in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)[:k]]

def get_k_hop_community(graph: nx.Graph, start_node: int, k: int) -> Set[int]:
    """Returns a numpy array of all nodes reachable from start_node within k hops."""
    if k < 0:
        return set()
    lengths = nx.single_source_shortest_path_length(graph, start_node, cutoff=k)
    community_list = list(lengths.keys())
    community_list.sort()
    return np.array(community_list)

def create_grid_graph(n):
    """
    Create a grid graph with nodes arranged in (n/4) by 4 grid.
    Nodes are labeled with integers 0 to n-1.
    
    Args:
        n: Total number of nodes (must be divisible by 4)
    
    Returns:
        G: graph with integer node IDs
    """
    if n & 0b11 != 0:
        raise ValueError(f"n must be divisible by 4, got {n}")
    
    rows = n // 4
    cols = 4
    
    G_temp = nx.grid_2d_graph(rows, cols)
    
    mapping = {(i, j): i * cols + j for i in range(rows) for j in range(cols)}
    G = nx.relabel_nodes(G_temp, mapping)
    
    return G
