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
        
        # Exact reciprocity: return what was received + delta
        target_amounts = self.received[edges] + self.delta
        
        # negative delta check
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
        self.copied_class = BaseAgent

    def allocate(self, time: int) -> np.ndarray:
        if time > 0:
            sharing_ratios = self.market.sharing_ratio_calculation(time)
            
            edges = self.edges()
            neighbours_and_self = np.append(edges, self.id)
            
            best_ratio_idx = np.argmax(sharing_ratios[neighbours_and_self])
            best_agent_id = neighbours_and_self[best_ratio_idx]
            
            # Keep own strategy if we are among the highest earning strategies
            best_ratio = sharing_ratios[best_agent_id]
            if sharing_ratios[self.id] >= best_ratio:
                # We are mathematically in the max set
                pass
            elif best_agent_id != self.id:
                best_agent = self.market.agents[best_agent_id]
                if isinstance(best_agent, ImitationAgent):
                    self.copied_class = best_agent.copied_class
                else:
                    self.copied_class = best_agent.__class__
                
                if hasattr(best_agent, 'delta'):
                    self.delta = best_agent.delta
                    
        return self.copied_class.allocate(self, time)


