import networkx as nx
import numpy as np

from ant.market import Market

class VariableSubMarket:
    def __init__(self, market: Market, center, k=1):
        self.parent_market = market
        self.parent_graph = self.parent_market.graph
        self.submask = self.neighbourhood_mask(center, k)
        self.ids = np.array(list(range(len(self.parent_graph))))[self.submask]
        self.agents = np.array([self.parent_market.agents[idx] for idx in self.ids])
        self.graph = nx.subgraph(self.parent_graph, self.ids)
        self.subgraph_center = list(self.ids).index(center)
        self.adjacency_mask = nx.to_numpy_array(self.graph, dtype=int)
        self.adjacency_mask_without_center = nx.to_numpy_array(self.graph, dtype=bool)
        self.adjacency_mask_without_center[:, self.subgraph_center] = False
        self.adjacency_mask_without_center[self.subgraph_center, :] = False

        self._resource_values = np.zeros(len(self.parent_graph))
        self._endowments = np.zeros(len(self.parent_market))
        self._post_market_construction_init_flag = False

        if len(self.graph) > 0.1 * len(self.parent_graph):
            print(f"WARNING: Subgraph is over 10% of parent graph ({len(self.graph) / len(self.parent_graph) * 100}%)")

    @property
    def most_recent_allocation_matrix(self):
        return self.parent_market.allocation_matrix
    

    @property
    def sub_market_allocation_matrix(self):
        return self.most_recent_allocation_matrix[:, self.submask][self.submask, :]
    
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

    def neighbourhood_mask(self, center, k):
        neighbourhood_mask = np.zeros(len(self.parent_graph), dtype=int)
        neighbourhood_mask[center] = 1
        queue = np.array([center], dtype=int)
        while k > 0 and len(queue) > 0:
            iteration_mask = np.zeros(len(self.parent_graph))
            for idx in queue:
                edges = np.array(list(nx.neighbors(self.parent_graph, idx)), dtype=int)
                iteration_mask[edges] = 1
            new_column_indices = np.where(iteration_mask * (1 - neighbourhood_mask) != 0)[0]

            if new_column_indices.any():
                queue = new_column_indices
                neighbourhood_mask[new_column_indices] = 1

            k -= 1
        return np.array(neighbourhood_mask, dtype=bool)
    

    def __len__(self):
        return len(self.graph)
