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
        resource_value: float = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(id, market=market, resource_value=resource_value, seed=seed)

    def allocate(self, time: int) -> np.ndarray:
        num_neighbors = len(self.received)
        total_received = np.sum(self.received)
        if total_received == 0: # default -> spread across neighbours
            return np.ones(num_neighbors) / num_neighbors * self.resource_count
        fractions = np.array(
            [
                (v / total_received) * self.resource_count
                for i, v in enumerate(self.received)
            ]
        )

        return fractions
