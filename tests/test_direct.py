import pytest
import numpy as np
from ant.decentralised.direct import ProportionalAgent


# Testing multiple cases in one go (parametrization)
@pytest.mark.parametrize(
    "received, surplus, expected",
    [
        ([10, 2], 1, [10.0/12.0, 2.0/12.0]),
        ([2, 2, 1], 4, [2.0/5.0*4.0, 2.0/5.0*4.0, 1.0/5.0*4.0]),
        ([1, 1, 1, 1], 3, [1.0/4.0*3.0, 1.0/4.0*3.0, 1.0/4.0*3.0, 1.0/4.0*3.0]),
    ],
)
def test_basic(received, surplus, expected):
    agent = ProportionalAgent(0)

    agent.received = np.array(received)
    agent.resource_count = surplus

    assert (agent.allocate(0) == expected).all()
