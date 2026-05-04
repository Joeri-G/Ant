"""
This module contains decentralised strategies that use information about the agents neighbours.
"""

import numpy as np
import networkx as nx
from ant.decentralised.utility import k_highest_indices
from ant.agent import BaseAgent
from typing import List


class RecirpocateAgent(BaseAgent):
    def allocate(self, time: int) -> np.ndarray:
        """
        Divides the surplus resources amongst the neighbours based on the resources received in the last round.
        """
        num_neighbors = len(self.edges())
        total_received = np.sum(self.received)

        if total_received == 0 or time == 0:  # default -> spread across neighbours
            fractions = (
                np.ones(num_neighbors) * self.production_timeline[time] / num_neighbors
            )
            allocation_vector = np.zeros(len(self.market))
            allocation_vector[self.edges()] = fractions
            return allocation_vector

        production_vector = self.market.production_vector_calculation(time - 1)

        weighted_proportional_received = (
            self.received / production_vector
        ) * self.market.resource_values

        fractions = weighted_proportional_received / np.sum(
            weighted_proportional_received
        )

        return fractions * self.production_timeline[time]


class MaxFinder(BaseAgent):
    def allocate(self, time: int) -> np.ndarray:
        """
        Allocates everything to the neighbour with the highest resource value
        """
        weighted_production_vector = (
            self.market.production_vector_calculation(time - 1)
            * self.market.resource_values
        )
        best_neighbour = self.edges()[
            np.argmax(weighted_production_vector[self.edges()])
        ]
        allocation_vector = np.zeros(len(self.market))
        allocation_vector[best_neighbour] = self.production_timeline[time]
        return allocation_vector
