import numpy as np

from voxcity.simulator.common.raytracing import calculate_transmittance


def test_calculate_transmittance_monotonic():
    t1 = calculate_transmittance(0.0, tree_k=0.6, tree_lad=1.0)
    t2 = calculate_transmittance(1.0, tree_k=0.6, tree_lad=1.0)
    t3 = calculate_transmittance(2.0, tree_k=0.6, tree_lad=1.0)
    assert 0.0 < t3 < t2 < t1 <= 1.0

