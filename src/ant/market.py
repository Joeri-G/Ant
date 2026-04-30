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
        market_time (int): Current simulation timestep
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

        self.market_time = 0

        if graph is not None:
            self.graph = graph
        else:
            self.graph: nx.Graph = nx.fast_gnp_random_graph(n, 0.45, seed=seed)

        if agents is not None:
            self.agents: Iterator[BaseAgent] = np.array(agents, dtype=BaseAgent)
        else:
            self.agents: Iterator[BaseAgent] = np.array(
                [agent_type(id, market=self, seed=seed + id) for id in range(n)],
                dtype=BaseAgent,
            )

        if seed is not None:
            self.seed = seed

        self.equilibrium_utility = None
        self.equilibrium_allocation = None

    def step(self) -> None:
        """
        Execute one timestep of the market simulation.
        """
        time = self.market_time
        # batch the allocation vectors for each agent
        adj_mask = nx.to_numpy_array(self.graph, nodelist=range(len(self)), dtype=int)
        X = np.zeros((len(self), len(self)), dtype=float)

        for agent in self.agents:
            agent.produce(time)
            agent.consume(time)
            allocation = agent.allocate(time)
            neighbor_indices = np.where(adj_mask[agent.id] > 0)[0]
            if len(neighbor_indices) > 0:
                X[agent.id, neighbor_indices] = allocation

        # apply the allocation vectors
        for agent in self.agents:
            neighbor_indices = np.where(adj_mask[agent.id] > 0)[0]
            if len(neighbor_indices) == 0:
                continue
            r_i = X[neighbor_indices, agent.id]
            x_i = X[agent.id, neighbor_indices]
            agent.send(x_i)
            agent.receive(r_i)
        self.market_time = time + 1

    def market_loss(self) -> float:
        """
        Calculate the loss of the currect market state.

        Returns:
            The loss of the current market state, compared to the equilibrium
        """

        def eq_utility(agent: BaseAgent):
            neighbor_indices = np.where(adj_mask[agent.id] > 0)[0]
            received = self.equilibrium_allocation[neighbor_indices, agent.id]
            return agent.utility(received)

        adj_mask = nx.to_numpy_array(self.graph, nodelist=range(len(self)), dtype=int)

        average_utility = np.array([agent.utility_over_time() for agent in self.agents])
        eq_utility = np.array([eq_utility(agent) for agent in self.agents])

        return np.sqrt(
            np.sum(np.square(average_utility - eq_utility))
        ) / np.linalg.norm(eq_utility)

    def set_market_equilibrium(self, eq_allocation, eq_utility):
        self.equilibrium_utility = eq_utility
        self.equilibrium_allocation = eq_allocation

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
        self.market_time = 0
        for t in range(duration):
            self.step()
            val = self.market_loss()
            results[t] = val

        return results

    def neighbours(self, agent: Union[BaseAgent, int]) -> Iterator[BaseAgent]:
        """
        Run the market simulation for a specified number of timesteps.

        Args:
            agent: The agent whose neighbours should be returned

        Returns:
            Iterator[BaseAgent]: Iterator over neighbours
        """
        if type(agent) == int:
            id = agent
        elif issubclass(agent_type, BaseAgent):
            id = agent.id
        else:
            raise TypeError("either provide an id or an agent")
        edges = nx.neighbors(self.graph, id)
        return map(lambda id: self.agents[id], edges)

    def __repr__(self) -> str:
        """Return a string representation of the market."""
        return f"Market(graph={self.graph}, agents={len(self.agents)} agents)"

    def __len__(self) -> int:
        """Return the number of agents in the market."""
        return len(self.agents)
