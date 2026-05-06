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