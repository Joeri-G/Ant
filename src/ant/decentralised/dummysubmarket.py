import networkx as nx
import numpy as np

from ant.market import Market

class DummySubmarket:
    def __init__(
        self, market: Market, center, warn_subgraph_ratio: bool = False, **kwargs
    ):
        self.parent_market = market
        self.parent_graph = self.parent_market.graph
        
        self.submask = self.neighbourhood_mask(center, 1)
        

        self.true_ids = np.array(list(range(len(self.parent_graph))))[self.submask]
        self.true_agents = np.array([self.parent_market.agents[idx] for idx in self.true_ids])
        self.dummy_id = np.max(self.true_ids) + 1
        self.all_ids = np.array([idx for idx in self.true_ids] + [self.dummy_id])

        self.graph = nx.subgraph(self.parent_graph, self.true_ids).copy()
        self.subgraph_center = list(self.true_ids).index(center)
        self.graph.add_node(self.dummy_id)
        
        # keep on using the same block of memory for this
        self._submarket_allocation_matrix = np.zeros((len(self.all_ids), len(self.all_ids)))

        self._resource_values = np.zeros(len(self.parent_graph))
        self._endowments = np.zeros(len(self.parent_market))
        self._post_market_construction_init_flag = False

    @property
    def most_recent_allocation_matrix(self):
        return self.parent_market.allocation_matrix

    @property
    def submarket_allocation_matrix(self):
        dummy_col_idx = len(self.all_ids) - 1
        self._submarket_allocation_matrix[:, self.submask][self.submask, :] = self.most_recent_allocation_matrix[:, self.submask][self.submask, :]
        print(np.sum(self.most_recent_allocation_matrix[self.submask, :], dim=0))
        

        

    @property
    def resource_values(self):
        if not self._post_market_construction_init_flag:
            self._post_market_construction_init()
        return self._resource_values

    @property
    def endowments(self):
        if not self._post_market_construction_init_flag:
            self._post_market_construction_init()
        return self._endowments

    def _post_market_construction_init(self):
        self._post_market_construction_init_flag = True
        self._resource_values = self.parent_market.resource_values[self.submask]
        self._endowments = self.parent_market.endowments[self.submask]

        # Add an adge to the dummy node for all nodes that have an edge that exists the subgraph
        for agent in self.true_agents:
            for edge in agent.edges:
                if edge not in self.true_ids:
                    self.graph.add_edge(self.dummy_id, agent.id)
                    break

    def neighbourhood_mask(self, center, k):
        neighbourhood_mask = np.zeros(len(self.parent_graph), dtype=int)
        neighbourhood_mask[center] = 1
        queue = np.array([center], dtype=int)
        while k > 0 and len(queue) > 0:
            iteration_mask = np.zeros(len(self.parent_graph))
            for idx in queue:
                edges = np.array(list(nx.neighbors(self.parent_graph, idx)), dtype=int)
                iteration_mask[edges] = 1
            new_column_indices = np.where(
                iteration_mask * (1 - neighbourhood_mask) != 0
            )[0]

            if new_column_indices.any():
                queue = new_column_indices
                neighbourhood_mask[new_column_indices] = 1

            k -= 1
        return np.array(neighbourhood_mask, dtype=bool)

    def __len__(self):
        return len(self.graph)

        