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
from ant.decentralised.utility import get_k_hop_community
from ant.decentralised.COAP import single_shot_COAP, make_fixed_agent_coap_solver

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

        self.COAP = None
    
    def post_market_initialization_hook(self):
        """
        Build the problem structure and solve it
        """
        self.community_indices = get_k_hop_community(self.market.graph, self.id, self.k)
        self.COAP = make_fixed_agent_coap_solver(len(self.market), self.id, self.community_indices, self.edges(), self.market.endowments, self.market.resource_values)

    
    def allocate(self, time: int) -> np.ndarray:
        if time == 0 or self.has_crashed:
            # print(f"{self.id}@t={time}: USING PROP")
            return super().allocate(time)
        
        # best_allocation_vector = single_shot_COAP(self.market.allocation_matrix, self.market.endowments, self.market.resource_values, self.id, self.community_indices, self.edges())

        best_allocation_vector = self.COAP(self.market.allocation_matrix)

        if best_allocation_vector is None:
            self.has_crashed = True
            return super().allocate(time)

        return best_allocation_vector / self.endowment * self.production_timeline[time]

        # try:
        #     best_allocation_vector = COAP(self.market, self.community_indices, self.id, self.market.allocation_matrix, time)
        # except Exception as e:
        #     self.has_crashed = True
        #     print(f"{self.id}@t={time} REVERTING TO PROP")
        #     print(e)
        #     return super().allocate(time)

        # return best_allocation_vector
