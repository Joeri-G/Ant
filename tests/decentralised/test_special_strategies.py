import pytest
import numpy as np
import networkx as nx
from ant.market import Market, BaseAgent
from ant.decentralised.special_strategies import EgalitarianAgent, PettyAgent, ImitationAgent
from ant.decentralised.direct import GreedyAgent, ProportionalAgent

def test_egalitarian_agent():
    G = nx.star_graph(3)
    agents = [EgalitarianAgent(id=0)] + [BaseAgent(id=i) for i in range(1, 4)]
    M = Market(4, graph=G, agents=agents)
    for a in M.agents:
        a.market = M
        a.production_timeline = np.ones(10) * 150 # Production of 150
        a.received = np.zeros(4)
        
    M.sharing_ratio_calculation = lambda t: np.array([0.0, 0.5, 1.0, 1.5])
    
    # Ratios for neighbours [1, 2, 3] = [0.5, 1.0, 1.5]
    # Max ratio = 1.5
    # Deficits = [1.0, 0.5, 0.0]
    # Total deficit = 1.5
    # Fractions = [1.0/1.5, 0.5/1.5, 0.0]
    # Allocation = [100, 50, 0]
    alloc = agents[0].allocate(0)
    
    assert np.isclose(alloc[1], 100.0)
    assert np.isclose(alloc[2], 50.0)
    assert np.isclose(alloc[3], 0.0)

def test_petty_agent():
    G = nx.star_graph(2)
    agents = [PettyAgent(id=0, delta=5.0)] + [BaseAgent(id=i) for i in range(1, 3)]
    M = Market(3, graph=G, agents=agents)
    for a in M.agents:
        a.market = M
        a.production_timeline = np.ones(10) * 50
        a.received = np.zeros(3)

    # Received from 1: 10, 2: 20
    agents[0].received[1] = 10.0
    agents[0].received[2] = 20.0
    # Targets: 10+5=15, 20+5=25 (Total target: 40)
    # Available prep: 50 -> enough to exact reciprocate
    alloc_exact = agents[0].allocate(0)
    assert np.isclose(alloc_exact[1], 15.0)
    assert np.isclose(alloc_exact[2], 25.0)
    
    # Case 2: Not enough resources
    agents[0].production_timeline = np.ones(10) * 20
    # Targets: 15, 25 (Total: 40). Fractions: 15/40, 25/40. Available: 20
    alloc_prop = agents[0].allocate(0)
    assert np.isclose(alloc_prop[1], 20 * (15/40))
    assert np.isclose(alloc_prop[2], 20 * (25/40))

def test_imitation_agent():
    G = nx.star_graph(2)
    agents = [ImitationAgent(id=0), GreedyAgent(id=1), ProportionalAgent(id=2)]
    M = Market(3, graph=G, agents=agents)
    for a in M.agents:
        a.market = M
        a.production_timeline = np.ones(10) * 100
        a.received = np.zeros(3)
        
    M.sharing_ratio_calculation = lambda t: np.array([1.0, 3.0, 2.0])
    
    # Needs to run with time > 0 to imitate
    alloc = agents[0].allocate(1)
    
    # It should have copied GreedyAgent because agent 1 (GreedyAgent) has the max ratio (3.0)
    assert agents[0].copied_class == GreedyAgent
