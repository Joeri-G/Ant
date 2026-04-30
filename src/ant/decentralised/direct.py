import numpy as np
from ant.decentralised.utility import k_highest_indices
from ant.market import BaseAgent
from typing import List

"""
This module contains decentralised strategies that use information about the agents own node.
"""


class ProportionalAgent(BaseAgent):
    def __init__(
        self,
        id: int,
        market: Optional[Market] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(id, market=market, seed=seed)

    def allocate(self, time: int) -> np.ndarray:
        """
        Divides the surplus resources amongst the neighbours based on the resources received in the last round.
        """
        num_neighbors = len(self.received)
        values = np.array([other.resource_value() for other in self.neighbours()])
        total_received = np.sum(self.received @ values)
        
        if total_received == 0:  # default -> spread across neighbours
            return np.ones(num_neighbors) / num_neighbors * self.resource_count
        
        fractions = (self.received * values / total_received) * self.resource_count

        return fractions

class EqualDivisionAgent(BaseAgent):
    def __init__(
        self,
        id: int,
        market: Optional[Market] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(id, market=market, seed=seed)

    def allocate(self, time: int) -> np.ndarray:
        """
        Divides the surplus resources equally amongst neighbours.
        """
        num_neighbors = len(self.received)
        total_received = np.sum(self.received)
        return np.ones(num_neighbors) / num_neighbors * self.resource_count
