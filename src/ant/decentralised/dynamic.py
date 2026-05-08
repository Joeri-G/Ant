"""
This module contains decentralised strategies that use information about all agents within k hops.
"""

import numpy as np
import networkx as nx
import cvxpy as cp
from ant.centralised import SOLVER_EPSILON
from ant.decentralised.direct import ProportionalAgent
from ant.agent import BaseAgent
from ant.decentralised.submarket import VariableSubMarket
from ant.centralised import P4


class OptimizerAgent(ProportionalAgent):
    def __init__(
        self,
        id: int,
        market: Optional[Market] = None,
        seed: Optional[int] = None,
        k=1,
        report_crashes: bool = False,
        **kwargs,
    ):
        super().__init__(id, market=market, seed=seed, **kwargs)
        self.k = k
        self.submarket = VariableSubMarket(self.market, self.id, **kwargs)
        self.has_crashed = False
        self.report_crashes = report_crashes

    def allocate(self, time: int) -> np.ndarray:
        """
        At each time step this agent will try to optimize their allocation for
        the small portion of the graph they can see and they have an influence on.
        """
        if self.has_crashed:
            return super().allocate(time)

        fractions = np.zeros(len(self.market))

        n = len(self.submarket)
        calculated_allocation_matrix = cp.Variable((n, n))
        # everything that is not in the sub-market is set to zero

        # constants
        endowments = self.submarket.endowments
        resource_values = self.submarket.resource_values

        alphas = resource_values * endowments

        constraints = []

        # Allocations to unconnected agents must be zero
        zero_entries = cp.multiply(
            1 - self.submarket.adjacency_mask, calculated_allocation_matrix
        )
        constraints += [zero_entries <= SOLVER_EPSILON]
        # Sum of allocations for each agent <= endowment

        constraints += [cp.sum(calculated_allocation_matrix, axis=1) <= endowments]
        # All allocations must be non-negative
        constraints += [calculated_allocation_matrix >= 0]

        # The weighted sum of received resources for agent i
        weighted_sums = calculated_allocation_matrix.T @ resource_values
        log_terms = cp.log(weighted_sums)
        objective_expr = cp.sum(alphas @ log_terms)

        # All allocations in the neighbours mask should have the value of the most recent allocation matrix
        fixed_values = self.submarket.submarket_allocation_matrix[
            self.submarket.adjacency_mask_without_center
        ]
        if fixed_values.shape != (0,):
            masked_allocations = calculated_allocation_matrix[
                self.submarket.adjacency_mask_without_center
            ]
            constraints += [masked_allocations == fixed_values]

        prob = cp.Problem(cp.Maximize(objective_expr), constraints)

        try:
            result = prob.solve(verbose=False)
        except Exception as e:
            if self.report_crashes:
                print(f"Solver crashed. id={self.id} t={time}, k={self.k}")
            result = -1
            self.has_crashed = True

        if result < 0:
            return super().allocate(time)

        result_matrix = np.array(calculated_allocation_matrix.value)

        submarket_optimal_allocation_vector = result_matrix[
            self.submarket.subgraph_center
        ]

        fractions[self.submarket.ids] = submarket_optimal_allocation_vector

        fractions /= np.sum(fractions)  # normalize

        return fractions * self.production_timeline[time]
