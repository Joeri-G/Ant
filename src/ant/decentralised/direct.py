import numpy as np
import networkx as nx
from ant.decentralised.utility import k_highest_indices
from ant.agent import BaseAgent
from typing import List

"""
This module contains decentralised strategies that use information about the agents own node.
"""


class ProportionalAgent(BaseAgent):
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
    def allocate(self, time: int) -> np.ndarray:
        """
        Divides the surplus resources equally amongst neighbours.
        """
        num_neighbors = len(self.received)
        total_received = np.sum(self.received)
        return np.ones(num_neighbors) / num_neighbors * self.resource_count

class OptimalAgent(BaseAgent):
    _optimal_allocation_vector = None

    def set_allocation_matrix(self, optimal_allocation_matrix: np.ndarray):
        row = optimal_allocation_matrix[self.id]
        neighbours = list(nx.neighbors(self.market.graph, self.id))
        self._optimal_allocation_vector = row[neighbours]

    def allocate(self, time: int) -> np.ndarray:
        if self._optimal_allocation_vector is None:
            raise ValueError("The optimal allocation matrix has to be set")
        return self._optimal_allocation_vector
