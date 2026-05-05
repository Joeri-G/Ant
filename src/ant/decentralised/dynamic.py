"""
This module contains decentralised strategies that use information about all agents within k hops.
"""

from ant.decentralised.neighbours import OptimizerAgent
from ant.decentralised.direct import ProportionalAgent
from ant.decentralised.submarket import VariableSubMarket


class OptimizerAgentK2(OptimizerAgent):
    def __init__(
        self, id: int, market: Optional[Market] = None, seed: Optional[int] = None, k=2
    ):
        super().__init__(id, market, seed, k)


class OptimizerAgentK3(OptimizerAgent):
    def __init__(
        self, id: int, market: Optional[Market] = None, seed: Optional[int] = None, k=3
    ):
        super().__init__(id, market, seed, k)


class OptimizerAgentK4(OptimizerAgent):
    def __init__(
        self, id: int, market: Optional[Market] = None, seed: Optional[int] = None, k=4
    ):
        super().__init__(id, market, seed, k)


class OptimizerAgentKm(OptimizerAgent):
    def __init__(
        self, id: int, market: Optional[Market] = None, seed: Optional[int] = None, k=6
    ):
        super().__init__(id, market, seed, k)
