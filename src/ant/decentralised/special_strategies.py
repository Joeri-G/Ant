import numpy as np
from ant.agent import BaseAgent
from typing import Optional

from ant.market import Market

class EgalitarianAgent(BaseAgent):
    def allocate(self, time: int) -> np.ndarray:
        allocation_vector = np.zeros(len(self.market))

        sharing_ratios = self.market.sharing_ratio_calculation(time)
        edges = self.edges()
        
        neighbour_ratios = sharing_ratios[edges]
        max_ratio = np.max(neighbour_ratios)
        
        deficits = max_ratio - neighbour_ratios
        total_deficit = np.sum(deficits)
        
        if total_deficit > 0:
            fractions = deficits / total_deficit
        else:
            fractions = np.ones(len(edges)) / len(edges)
            
        allocation_vector[edges] = fractions * self.production_timeline[time]
        return allocation_vector


class PettyAgent(BaseAgent):
    def __init__(self, id: int, market: Optional[Market] = None, seed: Optional[int] = None, delta: float = 0.0, **kwargs):
        super().__init__(id, market=market, seed=seed, **kwargs)
        self.delta = delta

    def allocate(self, time: int) -> np.ndarray:
        allocation_vector = np.zeros(len(self.market))
        edges = self.edges()
        
        if len(edges) == 0:
            return allocation_vector
            
        available_resources = self.production_timeline[time]
        
        target_amounts = self.received[edges] + self.delta
        
        target_amounts = np.maximum(0, target_amounts)
        
        total_target = np.sum(target_amounts)
        
        if total_target == 0:
            # equal division if we receive nothing
            fractions = np.ones(len(edges)) / len(edges)
            allocation_vector[edges] = fractions * self.production_timeline[time]
            return allocation_vector

        if total_target > available_resources:
            # not enough to reciprocate exactly, so proportionally
            actual_amounts = (target_amounts / total_target) * available_resources
        else:
            # enough to reciprocate exactly
            actual_amounts = target_amounts 
            
        allocation_vector[edges] = actual_amounts
        
        return allocation_vector
    

class ImitationAgent(BaseAgent):
    def __init__(self, id: int, market: Optional[Market] = None, seed: Optional[int] = None, **kwargs):
        super().__init__(id, market=market, seed=seed, **kwargs)
        self.copied_agent = BaseAgent(id=id, market=market, seed=seed)

    def _resolve_target_agent(self, agent: BaseAgent) -> BaseAgent:
        """
        Recursively drills down through wrappers to find the pure strategy currently being executed.
        """
        if isinstance(agent, ImitationAgent) and hasattr(agent, "copied_agent"):
            return self._resolve_target_agent(agent.copied_agent)
            
        if hasattr(agent, "internal_agents") and agent.internal_agents:
            weights = getattr(agent, "weights", [])
            # Pick the highest weighted strategy, or default to the first one
            idx = int(np.argmax(weights)) if len(weights) == len(agent.internal_agents) else 0
            return self._resolve_target_agent(agent.internal_agents[idx])
            
        # Base case: we hit a pure strategy
        return agent

    def allocate(self, time: int) -> np.ndarray:
        if time > 0:
            sharing_ratios = self.market.sharing_ratio_calculation(time)
            
            edges = self.edges()
            neighbours_and_self = np.append(edges, self.id)
            
            best_ratio_idx = np.argmax(sharing_ratios[neighbours_and_self])
            best_agent_id = neighbours_and_self[best_ratio_idx]
            best_ratio = sharing_ratios[best_agent_id]
            
            # Only change if someone else is strictly better
            if sharing_ratios[self.id] < best_ratio and best_agent_id != self.id:
                best_agent = self.market.agents[best_agent_id]
                
                target_agent = self._resolve_target_agent(best_agent)
                target_type = type(target_agent).__name__
                
                if type(self.copied_agent) is not type(target_agent):
                    if target_type == "PettyAgent":
                        self.copied_agent = target_agent.__class__(
                            id=self.id, market=self.market, delta=getattr(target_agent, "delta", 0.0)
                        )
                    elif target_type == "SatisficingAgent":
                        self.copied_agent = target_agent.__class__(
                            id=self.id, market=self.market, aspiration_level=getattr(target_agent, "aspiration_level", 1.0)
                        )
                    else:
                        self.copied_agent = target_agent.__class__(id=self.id, market=self.market)
                else:
                    if target_type == "PettyAgent":
                        self.copied_agent.delta = getattr(target_agent, "delta", 0.0)
                    elif target_type == "SatisficingAgent":
                        self.copied_agent.aspiration_level = getattr(target_agent, "aspiration_level", 1.0)
                        
        self.copied_agent.received = self.received
        self.copied_agent.received_history = self.received_history
        self.copied_agent.production_timeline = self.production_timeline
        self.copied_agent.market = self.market
        
        if hasattr(self, 'last_allocation_fractions'):
            self.copied_agent.last_allocation_fractions = self.last_allocation_fractions
            
        allocation = self.copied_agent.allocate(time)
        
        if hasattr(self.copied_agent, 'last_allocation_fractions'):
            self.last_allocation_fractions = self.copied_agent.last_allocation_fractions
            
        return allocation



class MixedAgent(BaseAgent):
    def __init__(self, id: int, market: Optional[Market] = None, seed: Optional[int] = None, strategies: list = None, weights: list = None, **kwargs):
        super().__init__(id, market=market, seed=seed, **kwargs)
        self.strategies = strategies if strategies is not None else []
        self.weights = weights if weights is not None else []
        self.internal_agents = []
        
        for strat_class in self.strategies:
            self.internal_agents.append(strat_class(id=id, market=market, seed=seed, **kwargs))

    def allocate(self, time: int) -> np.ndarray:
        allocation_vector = np.zeros(len(self.market))
        
        if not self.internal_agents:
            return allocation_vector
            
        for agent in self.internal_agents:
            agent.received = self.received
            agent.received_history = self.received_history
            agent.production_timeline = self.production_timeline
            agent.market = self.market
        
        for agent, weight in zip(self.internal_agents, self.weights):
            allocation_vector += weight * agent.allocate(time)
            
        return allocation_vector


class SatisficingAgent(BaseAgent):
    def __init__(self, id: int, market: Optional[Market] = None, seed: Optional[int] = None, aspiration_level: float = 1.0, **kwargs):
        super().__init__(id, market=market, seed=seed, **kwargs)
        self.aspiration_level = aspiration_level
        self.last_allocation_fractions = None

    def allocate(self, time: int) -> np.ndarray:
        allocation_vector = np.zeros(len(self.market))
        edges = self.edges()

        if len(edges) == 0:
            return allocation_vector

        if time > 0:
            sharing_ratios = self.market.sharing_ratio_calculation(time)
            my_ratio = sharing_ratios[self.id]

            if my_ratio >= self.aspiration_level and self.last_allocation_fractions is not None:
                # Satisfied: maintain the same allocation fractions
                return self.last_allocation_fractions * self.production_timeline[time]

        # Not satisfied or first round: use Greedy strategy
        if time > 0:
            sharing_ratios = self.market.sharing_ratio_calculation(time)
            min_sharing_ratio_edge = np.argmin(sharing_ratios[edges])
            neighbour = edges[min_sharing_ratio_edge]
        else:
            neighbour = self.random.choice(edges)

        allocation_vector[neighbour] = self.production_timeline[time]
        
        self.last_allocation_fractions = np.zeros(len(self.market))
        self.last_allocation_fractions[neighbour] = 1.0

        return allocation_vector
