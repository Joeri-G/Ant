import pytest
import numpy as np
from ant.decentralised.neighbours import Subgraph
from ant.market import Market
from ant.centralised import P4


def test_basic():
    n = 10
    seed = 10
    M = Market(n, seed=seed, agent_type=OptimizerAgent)

    agent_under_test = M.agents[0]

    optimal_allocation_matrix, optimal_allocation_vector = P4(M)

    market_state = M.simulate(1)
