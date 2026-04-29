"""
This module contains a number of centralised strategies
"""

from .market import Market, BaseAgent
import cvxpy as cp
import numpy as np
import networkx as nx
from typing import List

def P4(market: Market) -> float:
    """
    Calculates the market equilibrium based on the method described by Hellinga in 2025.
    Returns the optimal objective value.
    """
    agents: List[BaseAgent] = market.agents
    n = len(agents)
    
    # Pre-calculate constant vectors for efficiency
    alphas = np.array([agent.resource_value() * agent.long_term_resource_endowment() for agent in agents])
    resource_values = np.array([agent.resource_value() for agent in agents])
    
    # Build adjacency mask for the graph
    # adj_mask[i][j] = 1 if connected, 0 otherwise
    adj_mask = nx.to_numpy_array(market.graph, nodelist=range(n), dtype=int)
    
    # allocation_matrix[i][j] represents allocation from agent i to agent j
    allocation_matrix = cp.Variable((n, n))
    
    constraints = []
    
    # Allocations to unconnected agents must be zero
    zero_entries = cp.multiply(1 - adj_mask, allocation_matrix)
    constraints += [zero_entries == 0]
    # Sum of allocations for each agent <= long_term_resource_endowment
    endowments = np.array([agent.long_term_resource_endowment() for agent in agents])
    constraints += [cp.sum(allocation_matrix, axis=1) <= endowments]
    # All allocations must be non-negative
    constraints += [allocation_matrix >= 0]
    
    # Weighted sum for each agent i
    weighted_sums = allocation_matrix @ resource_values
    log_terms = cp.log(weighted_sums)
    objective_expr = cp.sum(alphas * log_terms)
    
    prob = cp.Problem(cp.Maximize(objective_expr), constraints)

    result = prob.solve()
    return result
