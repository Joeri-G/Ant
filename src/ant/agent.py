import networkx as nx
import numpy as np
from random import Random
from typing import List, Optional, Iterator, Any, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .market import Market

BASE_ENDOWMENT_RANGE = (1, 5)
BASE_VALUE_RANGE = (1, 5)
BASE_DISTRIBUTABLE_VARIANCE = 0.1
BASE_UTILITY_TIMELINE = 10000


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
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize a new agent.

        Args:
            id: Unique identifier for this agent
            graph: Network graph defining agent connections (optional)

        Raises:
            ValueError: If id is negative
        """
        if id < 0:
            raise ValueError("Agent ID must be non-negative")

        self.id = id
        self.market = market
        self.random = Random()

        if seed is not None:
            self.random.seed(seed)

        self.resource_value = self.random.uniform(
            BASE_VALUE_RANGE[0], BASE_VALUE_RANGE[1]
        )
        self.endowment = self.random.uniform(
            BASE_ENDOWMENT_RANGE[0], BASE_ENDOWMENT_RANGE[1]
        )

        self.received: List[float] = np.zeros(
            len(self.market) if market is not None else 0,
            dtype=float,
        )

        self.received_history: List[float] = np.zeros(
            (BASE_UTILITY_TIMELINE, len(self.market)) if market is not None else (0, 0),
            dtype=float,
        )

        self.production_timeline = np.zeros(BASE_UTILITY_TIMELINE)

        self.reset()

    def utility(self, received=None) -> float:
        """
        Calculate the utility derived from mopst recent received vector.

        Returns:
            float: Utility value (base implementation returns 0)

        Note:
            This method should be overridden by subclasses to implement
            specific utility functions.
        """
        received_vector = self.received if received is None else received
        return np.sum(received_vector @ self.market.resource_values)

    def utility_over_time(self, time) -> float:
        """
        Keep track of the utility in previous states
        """
        self._utility_over_time[time] = self.utility()
        assert time >= 0  # np.mean([]) -> NaN
        return np.mean(self._utility_over_time[: time + 1])

    def reset(self) -> None:
        self._utility_over_time = np.zeros(BASE_UTILITY_TIMELINE)
        self.production_timeline = np.array(
            [
                self.random.gauss(self.endowment, BASE_DISTRIBUTABLE_VARIANCE)
                for _ in range(BASE_UTILITY_TIMELINE)
            ]
        )

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
        return self.production_timeline[time]

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
        num_neighbors = len(self.edges())
        neighbour_vector = np.zeros(num_neighbors, dtype=float)
        allocation_vector = np.zeros(len(self.market), dtype=float)
        favorite_neighbour = self.random.randint(0, num_neighbors - 1)
        second_favorite_neighbour = self.random.randint(0, num_neighbors - 1)
        ratio = self.random.random()
        neighbour_vector[favorite_neighbour] = self.production_timeline[time] * ratio
        neighbour_vector[second_favorite_neighbour] = self.production_timeline[time] * (
            1 - ratio
        )
        allocation_vector[self.edges()] = neighbour_vector
        return allocation_vector

    def receive(self, incoming: List[float], time=0) -> None:
        """
        Process incoming resource allocations from neighboring agents.

        Args:
            incoming: List of resource amounts received from each neighbor

        Note:
            The order of incoming resources corresponds to the order of
            neighbors in the graph.
        """
        entry = np.array(incoming, dtype=float)
        self.received = entry
        self.received_history[time] = entry

    def average_received_resources(self, time):
        recv_slice = self.received_history[: time + 1]
        return np.mean(recv_slice, axis=0)

    def weighted_sharing_ratio(self, time: int) -> float:
        avg_recv_vector = self.average_received_resources(time)
        sharing_ratio = np.sum(avg_recv_vector @ self.market.resource_values) / (
            self.resource_value * self.endowment
        )
        return sharing_ratio

    def neighbours(self) -> List[BaseAgent]:
        """
        Get the neighboring agents in the network.

        Returns:
            List[BaseAgent]: List with neighbors

        Raises:
            ValueError: If graph is not set
        """
        return self.market.neighbours(self.id)

    def edges(self) -> List[int]:
        return self.market.edges(self.id)

    def __eq__(self, other):
        assert isinstance(other, BaseAgent)
        return self.id == other.id

    def __lt__(self, other):
        assert isinstance(other, BaseAgent)
        return self.id < other.id
