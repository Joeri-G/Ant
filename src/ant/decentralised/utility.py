from ..market import Market, BaseAgent
import cvxpy as cp
import numpy as np
import networkx as nx
from typing import List, Tuple, Set

SOLVER_EPSILON = np.finfo(float).eps * 2

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

def COAP(X, endowments, resource_values, i, community_indices, neighbours) -> np.ndarray:
    """
    Calculates the optimal allocation vector x_i for agent i that maximizes 
    the Community Utility U^k_i, given a fixed set of community members.
    
    This function assumes a sequential market model where all other agents' 
    allocations remain static (x_j(t+1) = x_j(t) for j != i).

    Parameters:
    -----------
    X : np.ndarray
        Current market allocation matrix (n x n). 
        X[a, b] = amount allocated FROM agent a TO agent b.
    endowments : np.ndarray
        Vector of length n. endowments[i] is the max distributable resources for agent i.
    resource_values : np.ndarray
        Vector of length n. resource_values[j] is the value of agent j's resource type.
    i : int
        The index of the center agent whose allocation we are optimizing.
    community_indices : list or np.array
        List of integers representing the agents in the community C^k_i.
        Must include 'i' and any neighbors of 'i'.
    neighbours : list or np.array
        List of integers representing the agents in the neighbourhood N_i.

    Returns:
    --------
    x_new : np.ndarray
        The new optimal allocation vector for agent i (length n).
        Non-zero values will only appear at indices present in community_indices 
        AND reachable from i (logically enforced by the problem structure).
    """
    n = len(endowments)
    # Initializes problem size and creates a non-negative optimization variable vector for the agent's allocations.
    x_i = cp.Variable(n, nonneg=True)
    # Calculates the weighted resources already received by all agents from everyone except the center agent 'i'.
    static = X.T @ resource_values - X[i] * resource_values[i]

    # Constructs the objective function: sum of (endowment * value * log(total received)) for all agents in the community.
    # Epsilon is added to prevent log of (near) zero
    static_received = cp.sum([(endowments[j] * resource_values[j]) * cp.log(static[j] + x_i[j] * resource_values[i] + SOLVER_EPSILON) for j in community_indices])

    constraints = [
        cp.sum(x_i) <= endowments[i], # x_i must be within endowments
        x_i >= 0, # nonnegative allocation (might be redundant since x_1: nonneg=True)
        
    ] + [x_i[j] <= SOLVER_EPSILON for j in range(n) if j not in neighbours] # Allocations to agents not in the neighbourhood of i must be zero
    # Formulates the convex optimization problem to maximize the objective under endowment and non-negativity constraints.
    problem = cp.Problem(cp.Maximize(static_received), constraints)

    # Solves the problem using the SCS solver without printing output logs.
    # problem.solve(solver=cp.SCS, verbose=False)
    # problem.solve(solver=cp.SCS, eps=1e-7, max_iters=5000, verbose=False, normalize=True)
    try:
        problem.solve(solver=cp.CLARABEL,
           tol_gap_abs=1e-8,      # Absolute duality gap tolerance
           tol_gap_rel=1e-8,      # Relative duality gap tolerance  
           tol_feas=1e-8,         # Feasibility tolerance
           max_iter=2000,         # Maximum iterations
           verbose=False,         # Suppress output
           time_limit=5.0)        # Time limit in seconds
    except Exception:
        problem.solve(solver=cp.SCS, eps=1e-7, max_iters=5000, verbose=False)

    return (x_i.value if problem.status in ['optimal', 'optimal_inaccurate'] else None)
