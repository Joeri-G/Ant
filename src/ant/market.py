"""
Market simulation framework for modeling agent interactions and resource allocation.

This module provides data structures for simulating economic markets where agents
interact through a network graph, producing, consuming, and allocating resources.

Nomenclature:
    N: Set of all agents
    n: Number of agents
    i, j: Indices for agents
    E: Set of connections between agents
    G: Network structure
    N_i: Neighbourhood of agent i
    t: Timestamp index
    e_i: Long-term resource average endowment of agent i
    D_i: Distributable resources by agent i
    x_{ij}: Allocation from agent i to agent j
    x_i: Allocation vector of agent i
    X: Allocation matrix
    r_i: Resources received by agent i
    R_i: Total resources received by agent i
    v_i: Value of agent i's resource
    U_i: Total utility obtained by agent i
    u_{ij}: Vector of all received utilities
    u_i(.): Utility function for agent i
    p_i: Sharing ratio of agent i
    w_i: Strategy weight of agent i
    g_i: Memory decay of agent i
    c: Convergence error tolerance
    S^M: Incentive Ratio of market M
"""

from typing import TYPE_CHECKING
import numpy as np
import networkx as nx
from typing import List, Optional, Iterator, Any, Union

# if TYPE_CHECKING:
from .agent import BaseAgent


class Market:
    """
    Represents a market simulation environment with interconnected agents.

    The market manages a network of agents that interact according to their
    connections in the graph, simulating resource production, distribution,
    and consumption over time.

    Attributes:
        agents (np.ndarray): Array of agent instances in the market
        graph (nx.Graph): Network structure defining agent connections
        equilibrium_utility (float): The equilibrium utility, computed with P4 (Hellinga, 2025)
        equilibrium_allocation (np.ndarray): The equilibrium allocation
    """

    def __init__(
        self,
        n: int,
        graph: Optional[nx.Graph] = None,
        agents: Optional[List[BaseAgent]] = None,
        agent_type: type = BaseAgent,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new market simulation.

        Args:
            n: Number of agents to create (if agents not provided)
            graph: Pre-existing network graph (optional)
            agents: Pre-created list of agents (optional)
            agent_type: Class type for creating agents (default: BaseAgent)
            seed: The seed used for graph generation (optional)

        Raises:
            ValueError: If both graph and n are provided without agents
            TypeError: If agent_type is not a subclass of BaseAgent
        """
        if not issubclass(agent_type, BaseAgent):
            raise TypeError("agent_type must be a subclass of BaseAgent")

        if graph is not None:
            self.graph = graph
        else:
            self.graph: nx.Graph = nx.fast_gnp_random_graph(n, 0.45, seed=seed)

        if agents is not None:
            self.agents: Iterator[BaseAgent] = np.array(agents, dtype=BaseAgent)
        else:
            self.agents = np.empty(n, dtype=BaseAgent)
            for i in range(n):
                self.agents[i] = agent_type(i, market=self, seed=seed + i)

        if seed is not None:
            self.seed = seed

        self.allocation_matrix = np.zeros((n, n), dtype=float)
        self.receive_matrix = np.zeros((n, n), dtype=float)

        self.resource_values = np.array([agent.resource_value for agent in self.agents])

        self.sharing_ratios = np.zeros(n)
        self.sharing_ratios_time = -1
        self.production_vector = np.zeros(n)
        self.production_time = -1

        self.equilibrium_utility = np.zeros(n)
        self.equilibrium_allocation = np.zeros((0, 0))
        self.equilibrium_length = 1

    def step(self, time) -> None:
        """
        Execute one timestep of the market simulation.
        """

        allocation_matrix = np.zeros((len(self), len(self)), dtype=float)

        self.sharing_ratios = self.sharing_ratio_calculation(time)

        for agent in self.agents:
            if len(agent.edges()) > 0:
                allocation_matrix[agent.id] = agent.allocate(time)

        receive_matrix = allocation_matrix.T

        for agent in self.agents:
            agent.receive(receive_matrix[agent.id], time)

        self.allocation_matrix = allocation_matrix
        self.receive_matrix = receive_matrix

    def sharing_ratio_calculation(self, time):
        if time == self.sharing_ratios_time:
            return self.sharing_ratios
        self.sharing_ratios = np.array(
            [agent.weighted_sharing_ratio(time) for agent in self.agents]
        )
        self.sharing_ratios_time = time
        return self.sharing_ratios

    def production_vector_calculation(self, time):
        if time == self.production_time:
            return self.production_vector
        self.production_vector = np.array(
            [agent.production_timeline[time] for agent in self.agents]
        )
        return self.production_vector

    def market_utility(self) -> np.ndarray:
        return np.array([agent.utility() for agent in self.agents])

    def market_loss(self, time) -> float:
        """
        Calculate the loss of the currect market state.

        Returns:
            The loss of the current market state, compared to the equilibrium
        """

        average_utility_list = np.array(
            [agent.utility_over_time(time) for agent in self.agents]
        )

        utility_difference = average_utility_list - self.equilibrium_utility
        # utility_size = np.sum(equilibrium_utility_list)

        return np.sqrt(np.sum(np.square(utility_difference))) / self.equilibrium_length

    def set_market_equilibrium(self, eq_allocation, eq_utility):
        self.equilibrium_utility = eq_utility
        self.equilibrium_allocation = eq_allocation
        self.equilibrium_length = np.linalg.norm(self.equilibrium_utility)

    def simulate(self, duration: int) -> List[float]:
        """
        Run the market simulation for a specified number of timesteps.

        Args:
            duration: Number of timesteps to simulate

        Raises:
            ValueError: If duration is negative
        """
        if duration < 0:
            raise ValueError("Duration must be non-negative")

        if self.equilibrium_allocation is None or self.equilibrium_utility is None:
            raise ValueError(
                "An equilibrium allocation must be provided in order to compute market loss. Use Market.set_market_equilibrium"
            )

        results = np.zeros(duration)
        for time in range(duration):
            self.step(time)
            val = self.market_loss(time)
            # val = np.sum(np.array([agent.utility_over_time() for agent in self.agents]))
            results[time] = val

        for agent in self.agents:
            agent.reset()

        return results

    def neighbours(self, agent: Union[BaseAgent, int]) -> np.ndarray:
        """
        Run the market simulation for a specified number of timesteps.

        Args:
            agent: The agent whose neighbours should be returned

        Returns:
            Iterator[BaseAgent]: Iterator over neighbours
        """
        edges = self.edges(agent)
        return np.array(list(map(lambda id: self.agents[id], edges)))

    def edges(self, agent: Union[BaseAgent, int]) -> np.ndarray:
        if type(agent) == int:
            id = agent
        elif issubclass(agent_type, BaseAgent):
            id = agent.id
        else:
            raise TypeError("either provide an id or an agent")
        return np.fromiter(nx.neighbors(self.graph, id), int)

    def __repr__(self) -> str:
        """Return a string representation of the market."""
        return f"Market(graph={self.graph}, agents={len(self.agents)} agents)"

    def __len__(self) -> int:
        """Return the number of agents in the market."""
        return len(self.agents)
