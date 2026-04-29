"""
The data structures necessary for modeling a market

Nomenclature
N:      the set of all agents
n:      number of agents
i, j:   indices for agents
E:      set of connections between agents
G:      network structure
N_i:    Neighbourhood of agent i

t:      Timestamp index
e_i:    Long-term resource average endownment of agent i
D_i:    Distributable resources by agent i
x_{ij}: Allocation from agent i to agent j
x_i:    Allocation vector of agent i
X:      Allocation matrix

r_i:    Resources received by agent i
R_i:    Total resources received by agent i
v_i:    Value of agent i's resource
U_i:    Total utility obtained by agent i
u_{ij}: U Vector of all received utilities
u_i(.): Utility function for agent i
p_i:    Sharing ratio of agent i
w_i:    Strategy weight of agent i
g_i:    Memory decay of agent i
c:      Convergence error tolerance
S^M:    Incentive Ratio of market M
"""

import numpy as np
import networkx as nx
from typing import List


class BaseAgent:
    id: int
    received: np.ndarray
    resource_count: int
    graph: nx.Graph

    def __init__(self, id, graph=None):
        self.id = id
        self.resource_count = 0
        self.received = np.empty((0, 0), dtype=int)
        self.graph = graph

    def utility(self):
        return 0

    def receive(self, incomming: List[int]) -> None:
        self.received = incomming

    def produce(self, time: int) -> int:
        produced = 10
        self.resource_count += produced
        return produced

    def consume(self, time: int) -> int:
        consumed = 0
        self.resource_count -= consumed
        return consumed

    def allocate(self, time: int) -> List[int]:
        allocation_vector = np.fromiter(
            [0 for _ in range(len(self.received))], dtype=int
        )
        if len(allocation_vector) > 0:
            allocation_vector[0] = self.resource_count
        self.resource_count = 0
        return allocation_vector


"""
A market instance contains:
  - A graph denoting the connections between agents
  - A list of agents
"""


class Market:
    agents: List[BaseAgent]
    graph: nx.Graph
    market_time: int

    def __init__(
        self,
        n: int,
        graph: nx.Graph = None,
        agents: List[BaseAgent] = None,
        agent_type: type = BaseAgent,
    ):
        self.market_time = 0
        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.fast_gnp_random_graph(
                n, 0.35
            )  # random graph with 35% edge probability
        self.agents = np.fromiter(
            (agent_type(i, self.graph) for i in range(n)), dtype=object
        )

    def step(self, time: int) -> None:
        pass

    def simulate(self, duration: int) -> None:
        for i in range(duration):
            self.step(i)

    def __repr__(self):
        return f"Market({self.graph}, {self.agents})"
