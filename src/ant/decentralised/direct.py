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
        num_neighbors = len(self.edges())

        total_received = np.sum(self.received @ self.market.resource_values)

        if total_received == 0:  # default -> spread across neighbours
            fractions = np.ones(num_neighbors) * self.resource_count / num_neighbors
            allocation_vector = np.zeros(len(self.market))
            allocation_vector[self.edges()] = fractions
            return allocation_vector
        return (
            self.received * self.market.resource_values / total_received
        ) * self.resource_count


class EqualDivisionAgent(BaseAgent):
    def allocate(self, time: int) -> np.ndarray:
        """
        Divides the surplus resources equally amongst neighbours.
        """
        num_neighbors = len(self.edges())
        total_received = np.sum(self.received)
        allocation_vector = np.zeros(len(self.market))
        allocation_vector[self.edges()] = 1
        return allocation_vector / num_neighbors * self.resource_count


class OptimalAgent(BaseAgent):
    _optimal_allocation_ratio = None

    def set_allocation_matrix(self, optimal_market_matrix: np.ndarray):
        allocation_row = optimal_market_matrix[self.id]
        self._optimal_allocation_ratio = allocation_row / np.sum(allocation_row)
        # masked_allocation_row = np.zeros(len(allocation_row))
        # masked_allocation_row[self.edges()] = allocation_row[self.edges()]
        # self._optimal_allocation_ratio = masked_allocation_row / np.sum(masked_allocation_row)

    def allocate(self, time: int) -> np.ndarray:
        if self._optimal_allocation_ratio is None:
            raise ValueError("The optimal allocation matrix has to be set")
        allocation_vector = self._optimal_allocation_ratio * self.resource_count
        if np.any(np.isnan(allocation_vector)):
            print("NaN encountered in Optimal Agent Allocation Vector")
            return np.zeros(len(allocation_vector))
        return allocation_vector


class GreedyAgent(BaseAgent):
    def allocate(self, time: int) -> np.ndarray:
        allocation_vector = np.zeros(len(self.market))

        sharing_ratios = self.market.sharing_ratio_calculation(time)
        edges = self.edges()
        min_sharing_ratio_edge = np.argmin(sharing_ratios[edges])
        neighbour = edges[min_sharing_ratio_edge]
        allocation_vector[neighbour] = self.resource_count
        return allocation_vector
