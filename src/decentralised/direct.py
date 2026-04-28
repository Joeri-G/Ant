from src.decentralised.utility import k_highest_indices
from typing import List
'''
This module contains decentralised strategies that use information about the agents own node.
'''

'''
    Divide the resources proportional to the resources received in the previous round.
    The remainder is divided over the top contributers
'''
def proportional(received: List[int], surplus: int) -> List[int]:
    total_received = sun(received)
    fractions = [v / total_received for i, v in enumerate(received)]
    remainder = total_received - sum(fractions)
    k_highest_indices(fractions, remainder)
    for i in remainder:
        fractions[i] += 1
    return fractions
