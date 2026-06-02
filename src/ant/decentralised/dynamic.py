"""
This module contains decentralised strategies that use information about all agents within k hops.
"""

import numpy as np
import networkx as nx
import cvxpy as cp
from ant.centralised import SOLVER_EPSILON
from ant.decentralised.direct import ProportionalAgent
from ant.agent import BaseAgent
from ant.centralised import P4
from ant.decentralised.utility import COAP, get_k_hop_community

class COAPAgent(ProportionalAgent):
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
        self.has_crashed = False
        self.report_crashes = report_crashes
        self.community_indices = []
    
    def post_market_initialization_hook(self):
        """
        Build the problem structure and solve it
        """
        self.community_indices = get_k_hop_community(self.market.graph, self.id, self.k)
    
    def allocate(self, time: int) -> np.ndarray:
        if time == 0:
            return super().allocate(time)
            # allocation_vector = np.zeros(len(self.market))
            # allocation_vector[self.edges()] = self.production_timeline[0] / len(self.edges())
            # return allocation_vector


        if self.has_crashed:
            print("USING PROP")
            return super().allocate(time)
        try:
            best_allocation_vector, _community_utility = COAP(self.market, self.community_indices, self.id, self.market.allocation_matrix)
        except Exception as _e:
            return super().allocate(time)
            self.has_crashed = True

        return best_allocation_vector / self.endowment * self.production_timeline[time]
