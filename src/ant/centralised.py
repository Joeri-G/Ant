"""
This module contains a number of centralised strategies
"""

from .market import Market, BaseAgent
import cvxpy as cp
import numpy as np
import networkx as nx
from typing import List, Tuple


def P4(market: Market) -> Tuple(np.ndarray, float):
    """
    Calculates the market equilibrium based on the method described by Hellinga in 2025.
    Returns the optimal allocation matrix and utility vector
    """
    agents: List[BaseAgent] = market.agents
    n = len(agents)

    EPSILON = np.finfo(float).eps

    endowments = np.array([agent.endowment for agent in agents])
    resource_values = np.array([agent.resource_value for agent in agents])

    alphas = resource_values * endowments

    adj_mask = nx.to_numpy_array(market.graph, nodelist=range(n), dtype=int)

    # allocation_matrix[i][j] represents allocation from agent i to agent j
    allocation_matrix = cp.Variable((n, n))

    constraints = []

    # Allocations to unconnected agents must be zero
    zero_entries = cp.multiply(1 - adj_mask, allocation_matrix)
    constraints += [zero_entries <= EPSILON]
    # Sum of allocations for each agent <= endowment

    constraints += [cp.sum(allocation_matrix, axis=1) <= endowments]
    # All allocations must be non-negative
    constraints += [allocation_matrix >= 0]

    # The weighted sum of received resources for agent i
    weighted_sums = allocation_matrix.T @ resource_values
    log_terms = cp.log(weighted_sums)
    objective_expr = cp.sum(alphas @ log_terms)

    prob = cp.Problem(cp.Maximize(objective_expr), constraints)

    result = prob.solve()

    utility_vector = allocation_matrix.value.T @ resource_values

    return np.array(allocation_matrix.value), utility_vector

