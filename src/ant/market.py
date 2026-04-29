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

import numpy as np
import networkx as nx
import random
from typing import List, Optional, Iterator, Any, Union


class BaseAgent:
    """
    Base class representing an agent in the market simulation.

    Agents can produce, consume, and allocate resources based on their
    position in the network graph and their individual strategies.

    Attributes:
        id (int): Unique identifier for the agent
        received (np.ndarray): Array tracking incoming allocations from neighbors
        resource_count (float): Current number of resources held by the agent
        graph (nx.Graph): Reference to the market network structure
        resource_value (float): Value multiplier for this agent's resources
    """

    def __init__(
        self,
        id: int,
        market: Optional[Market] = None,
        resource_value: float = 1,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new agent.

        Args:
            id: Unique identifier for this agent
            graph: Network graph defining agent connections (optional)
            resource_value: Multiplier for resource valuation (default: 1)

        Raises:
            ValueError: If id is negative
        """
        if id < 0:
            raise ValueError("Agent ID must be non-negative")

        self.id = id
        self.resource_count = 0
        self.received: Iterator[float] = np.zeros(
            len(list(nx.neighbors(market.graph, id))) if market is not None else 0,
            dtype=float,
        )
        self.market = market
        self._resource_value = resource_value
        self._cached_endowment: Optional[int] = None

        self.random = random.Random()
        self.random.seed(seed)

        self.utility_over_time = (0, 0)  # value, weight

    def resource_value(self) -> float:
        """Get the resource value multiplier for this agent."""
        return self._resource_value

    def utility(self) -> float:
        """
        Calculate the utility derived from current resources.

        Returns:
            float: Utility value (base implementation returns 0)

        Note:
            This method should be overridden by subclasses to implement
            specific utility functions.
        """
        neighbor_values = np.array(
            [other.resource_value() for other in self.neighbours()]
        )  # maybe this should be cached?
        current_utility = np.sum(self.received @ neighbor_values)
        time = self.utility_over_time[1]
        utility_over_time = (self.utility_over_time[0] * time + current_utility) / (
            time + 1
        )
        self.utility_over_time = (utility_over_time, time + 1)
        return utility_over_time

    def receive(self, incoming: List[float]) -> None:
        """
        Process incoming resource allocations from neighboring agents.

        Args:
            incoming: List of resource amounts received from each neighbor

        Note:
            The order of incoming resources corresponds to the order of
            neighbors in the graph.
        """
        self.received = np.array(incoming, dtype=float)

    def produce(self, time: int) -> int:
        """
        Generate new resources for the agent.

        Args:
            time: Current simulation timestep (unused in base implementation)

        Returns:
            int: Number of resources produced

        Note:
            This method should be overridden for more sophisticated production
            models that consider time or other factors.
        """
        produced = 10
        self.resource_count += produced
        return produced

    def long_term_resource_endowment(self) -> float:
        """
        Calculate the long-term average resource production rate.

        Returns:
            float: Average resources produced per timestep

        Note:
            Uses caching to avoid recalculating the same simulation repeatedly.
            The simulation runs 1000 timesteps to establish a stable average.
        """
        if self._cached_endowment is not None:
            return self._cached_endowment

        endowment_timeline = 1000
        production_history = [self.produce(i) for i in range(endowment_timeline)]
        self._cached_endowment = int(np.mean(production_history))

        # Reset resource count after simulation
        self.resource_count = 0
        return self._cached_endowment

    def consume(self, time: int) -> int:
        """
        Consume resources from the agent's inventory.

        Args:
            time: Current simulation timestep (unused in base implementation)

        Returns:
            int: Number of resources consumed (base implementation returns 0)

        Note:
            This method should be overridden to implement consumption logic.
        """
        consumed = 0
        self.resource_count -= consumed
        return consumed

    def allocate(self, time: int) -> np.ndarray:
        """
        Distribute resources to neighboring agents.

        Args:
            time: Current simulation timestep (unused in base implementation)

        Returns:
            np.ndarray: Allocation vector showing resources sent to each neighbor

        Note:
            Current implementation sends all resources to the first neighbor.
            Override for more sophisticated allocation strategies.
        """
        num_neighbors = len(self.received)
        allocation_vector = np.zeros(num_neighbors, dtype=float)

        favorite_neighbour = self.random.randint(0, num_neighbors - 1)
        second_favorite_neighbour = self.random.randint(0, num_neighbors - 1)
        ratio = self.random.random()
        allocation_vector[favorite_neighbour] = self.resource_count * ratio
        allocation_vector[second_favorite_neighbour] = self.resource_count * (1 - ratio)
        return allocation_vector

    def send(self, allocation_vector: List[float]) -> None:
        self.resource_count -= np.sum(allocation_vector)

    def neighbours(self) -> Iterator[BaseAgent]:
        """
        Get the neighboring agents in the network.

        Returns:
            Iterator[BaseAgent]: Generator yielding neighbors

        Raises:
            ValueError: If graph is not set
        """
        return self.market.neighbours(self.id)


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

    # Broken
    # TODO: adhere to Hellinga, 2025, 4.4
    def market_loss(self, optimal) -> float:
        optimal_utility = np.array([np.sum(optimal[:, agent.id]) for agent in self.agents])
        market_utility = np.array([agent.utility() for agent in self.agents])
        return np.sqrt(np.sum(market_utility - optimal_utility)) / np.linalg.norm(
            optimal
        )

    def simulate(self, duration: int, optimal: np.ndarray) -> List[float]:
        """
        Run the market simulation for a specified number of timesteps.

        Args:
            duration: Number of timesteps to simulate

        Raises:
            ValueError: If duration is negative
        """
        if duration < 0:
            raise ValueError("Duration must be non-negative")

        results = np.zeros(duration)
        self.market_time = 0
        for t in range(duration):
            self.step()
            val = self.market_loss(optimal)
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
