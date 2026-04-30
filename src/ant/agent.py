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
        self.resource_count = 0
        self.received: Iterator[float] = np.zeros(
            len(list(nx.neighbors(market.graph, id))) if market is not None else 0,
            dtype=float,
        )
        self.random = Random()
        self.random.seed(seed)

        self.market = market
        self._resource_value = self.random.uniform(
            BASE_VALUE_RANGE[0], BASE_VALUE_RANGE[1]
        )
        self._cached_endowment: Optional[int] = None

        self.endowment = self.random.uniform(
            BASE_ENDOWMENT_RANGE[0], BASE_ENDOWMENT_RANGE[1]
        )

        self._utility_over_time = {"value": 0, "weight": 0}  # value, weight

    def resource_value(self) -> float:
        """Get the resource value multiplier for this agent."""
        return self._resource_value

    def utility(self, received=None) -> float:
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

        received = self.received if received is None else received
        return np.sum(received @ neighbor_values)

    def utility_over_time(self) -> float:
        """
        Keep track of the utility in previous states
        """
        current_utility = self.utility()
        time = self._utility_over_time["weight"]
        new_utility = (self._utility_over_time["value"] * time + current_utility) / (
            time + 1
        )
        self._utility_over_time = {"value": new_utility, "weight": time + 1}
        return new_utility

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
        produced = self.random.gauss(self.endowment, BASE_DISTRIBUTABLE_VARIANCE)
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
