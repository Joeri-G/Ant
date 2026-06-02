from ..market import Market, BaseAgent
import cvxpy as cp
import numpy as np
import networkx as nx
from typing import List, Tuple, Set

SOLVER_EPSILON = np.finfo(float).eps

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


    def COAP(
        market: Market, 
        community_set: Set(int),
        community_center: int,
        current_allocation: np.ndarray,
        time: int
    ) -> np.ndarray:
        """
        Who needs docs anyways?
        """
        agents: List[BaseAgent] = market.agents
        n = len(agents)

        # D_j(t+1) = D_j(t)
        # For agent i: D_i(t+1) is their actually distributable resources
        # distributable_resources = np.array([agent.production_timeline[time-1] for agent in agents])
        # distributable_resources[community_center] = agents[community_center].production_timeline[t]
        distributable_resources = np.array([agent.endowment for agent in agents])
        resource_values = np.array([agent.resource_value for agent in agents])

        alphas = resource_values * distributable_resources

        adj_mask = nx.to_numpy_array(market.graph, nodelist=range(n), dtype=int)

        # allocation_matrix[i][j] represents allocation from agent i to agent j
        allocation_matrix = cp.vstack([
        cp.Variable(n) if i == community_center else row for i, row in enumerate(current_allocation)
        ])

        constraints = []

        # Allocations to unconnected agents must be zero
        zero_entries = cp.multiply(1 - adj_mask, allocation_matrix)
        constraints += [zero_entries <= SOLVER_EPSILON]
        # Sum of allocations for each agent <= endowment

        constraints += [cp.sum(allocation_matrix, axis=1) <= distributable_resources]
        # All allocations must be non-negative
        constraints += [allocation_matrix >= 0]

        # The weighted sum of received resources for agent i
        weighted_sums = allocation_matrix.T[community_set, :] @ resource_values
        log_terms = cp.log(weighted_sums)
        objective_expr = cp.sum(alphas[community_set] @ log_terms)

        prob = cp.Problem(cp.Maximize(objective_expr), constraints)

        # result = prob.solve()

        result = prob.solve()

        return allocation_matrix[community_center].value
