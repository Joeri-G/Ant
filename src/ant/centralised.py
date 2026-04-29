"""
This module contains a number of centralised strategies
"""

from .market import Market

"""
This function calculates the market equilibrium using the method described by Hellinga in 2025.
"""
def P4(market: Market) -> float:
    allocation_vector = [
        ...
    ]
    score = 0
    for agent in market.agents:
        alpha = agent.resource_value() * agent.long_term_resource_endownment()
        log_sum = 0
        for other in agent.neighbours():
            log_sum += other.resource_value() * allocation_vector[agent.id][other.id]
        score += alpha * np.log(log_sum)
    return score
