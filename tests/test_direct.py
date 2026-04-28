import pytest
from ant.decentralised.direct import proportional

# Testing multiple cases in one go (parametrization)
@pytest.mark.parametrize("received, surplus, expected", [
    ([10, 2], 1, [1, 0]),
    ([2, 2, 1], 4, [2, 2, 0]),
    ([1, 1, 1, 1], 3, [1, 1, 1, 0]),
])
def test_basic(received, surplus, expected):
    assert proportional(received, surplus) == expected
    # assert testfunction() == "Foo"
