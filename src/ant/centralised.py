"""
This module contains a number of centralised strategies
"""

from .market import Market
import cvxpy as cp

def P4(market: Market) -> List[List[int]]:
    """
    This function calculates the market equilibrium based on the method described by Hellinga in 2025.
    """
    def market_score(allocation_matrix):
        score = 0
        for agent in market.agents:
            alpha = agent.resource_value() * agent.long_term_resource_endowment()
            log_sum = 0
            for other in agent.neighbours():
                log_sum += other.resource_value() * allocation_matrix[agent.id][other.id]
            score += alpha * np.log(log_sum)
        return score

    allocation_matrix = cp.Variable((n, n))
    constraints = []
    indexes = list(range(n))
    n = len(market.agents)
    # build constraints
    for agent in market.agents:
        neighbours = nx.neighbors(market.graph, agent.id)
        unconnected = list(filter(lambda i: i not in neighbours, indexes))
        # the allocaion from i to an unconnected agent must be zero
        constraints += [allocation_matrix[agent.id][j] == 0 for j in unconnected]
        # the sum of the allocations must be less than an agents long_term_resource
        constraints += [cp.sum(allocation_matrix[agent.id]) <= agent.long_term_resource_endowment()]
        # all allocations must be positive
        constraints += [allocation_matrix[agent.id][j] >= 0 for j in neighbours]
    
    # transform the market_score into an objective
    # TRANSFORM THIS
    obj = cp.Maximize(market_score(allocation_matrix))
    
    # solve
    prob = cp.Problem(obj, constraints)
    return prob.solve()
    